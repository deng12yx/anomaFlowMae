import argparse
import json
import multiprocessing

from concurrent.futures import as_completed, ProcessPoolExecutor
from datasets import Dataset

from multiprocessing import Queue
from typing import Tuple, Union

from dataset import config
from preprocess import Ether, IP, TCP, UDP, ARP, DNS

from pathlib import Path
import shutil

from datasets import load_from_disk, DatasetDict

import pickle

import jsonlines
from preprocess import DATASETS_FILE
from preprocess.pyspark.datasets_adaptor import process_parquet_to_datasetdict, print_label_counts
from preprocess.pyspark.spark_aggregator import PySparkAggregator

import pandas as pd
from scapy.all import *


def read_and_fetch_packets(packet_queue, pcap_path, output_batch_size, max_batch):
    print(f"Reading from file: {pcap_path}")
    packet_reader = PcapReader(str(pcap_path))
    batch_count = 0

    while True:
        batch = list(itertools.islice(packet_reader, output_batch_size))
        if not batch or (max_batch and batch_count >= max_batch):
            packet_queue.put(None)
            break
        packet_queue.put(batch)
        batch_count += 1


def should_omit_packet(packet):
    # if len(bytes(packet)) < 800:
    #     return True

    # SYN, ACK or FIN flags set to 1 and no payload
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    if UDP in packet and not packet[UDP].payload:
        return True

    # DNS segment
    if DNS in packet or ARP in packet:
        return True

    return False


def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"

    return packet


def mask_udp(packet):
    if UDP in packet:
        packet[UDP].sport = 0
        packet[UDP].dport = 0

    return packet


def mask_tcp(packet):
    if TCP in packet:
        packet[TCP].sport = 0
        packet[TCP].dport = 0

    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = "\x00" * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet


def crop_and_pad(packet, max_length=1024) -> Tuple[bytes, int]:
    packet_bytes = bytearray(raw(packet))
    origin_len = len(packet_bytes)

    if origin_len < max_length:
        packet_bytes.extend(b'\x00' * (max_length - origin_len))
    elif origin_len > max_length:
        packet_bytes = packet_bytes[:max_length]

    return bytes(packet_bytes), min(origin_len, max_length)


def transform_packet(packet) -> Union[Tuple[bytes, int], None]:
    # 判断是否应该省略数据包
    if should_omit_packet(packet):
        return None

    # 移除以太网头部
    packet = remove_ether_header(packet)

    # 对 IP 地址进行掩码处理
    packet = mask_ip(packet)

    # 对 UDP 协议部分进行掩码处理
    packet = mask_udp(packet)

    # 对 TCP 协议部分进行掩码处理
    packet = mask_tcp(packet)

    # 对数据包进行截断和填充处理
    return crop_and_pad(packet)


def preprocess_function(packet, label):
    try:
        feature, feature_len = transform_packet(packet)
        if feature is None or feature_len is None:
            return None
        return {"x": feature, "feature_len": feature_len, "labels": label}
    except Exception as e:
        # 捕获异常，并在发生异常时执行其他操作，例如记录错误信息
        # print(f"An error occurred while preprocessing packet: {e}")
        return None


def transform_pcap(packet_queue: Queue, num_producers: int,result_queue):
    """
    从数据包队列中获取数据，处理后保存为数据集。

    Args:
        packet_queue (Queue): 存放数据包的队列。
        num_producers (int): 生产者数量，用于检测结束条件。
        output_path (str): 输出路径，用于保存处理后的数据集。

    Returns:
        None
    """
    end_count = 0  # 计数器，用于检测结束条件
    rows = []  # 存放处理后的数据行

    while True:
        item = packet_queue.get()  # 从队列中获取数据
        if item is None:  # 检测到结束标志
            end_count += 1
            if end_count == num_producers:  # 所有生产者都结束时退出循环
                break
        else:
            batch = item  # 解包数据
            for packet in batch:  # 遍历数据包批次
                row = preprocess_function(packet, "1")  # 预处理数据包并生成数据行
                if row is not None:  # 如果数据行不为空
                    rows.append(row)  # 将数据行添加到列表中
    print("ok！")
    # 从字典列表创建数据集
    result = Dataset.from_dict({k: [dic[k] for dic in rows] for k in rows[0]})
    result_queue.put(result)
    # return dataset



def clean_dirs(*dirs):
    for cur_dir in dirs:
        if cur_dir.exists():
            shutil.rmtree(cur_dir)
        cur_dir.mkdir(parents=True)


class PcapDict(dict):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = Path(root_dir)
        self._load_data()

    def _load_data(self):
        # Iterate through all subdirectories in the root directory
        for label_dir in self.root_dir.iterdir():
            # Skip if it's not a directory
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            label = re.sub('[^0-9a-zA-Z]+', '_', str(label))
            # Iterate through all pcap files in the subdirectory
            pcap_files = [
                pcap_file
                for pcap_file in label_dir.iterdir()
                if pcap_file.name.endswith(".pcap")
            ]

            # Store the label and the list of pcap files
            self[label] = pcap_files

    def __repr__(self):
        return f"PcapDict(root_dir={self.root_dir})"


def save_id2label(target_dir_path, id2label):
    with (target_dir_path / "id2label.json").open("w") as f:
        json.dump(id2label, f, indent=4)


def get_args():
    parser = argparse.ArgumentParser(description="PCAP Preprocessing")
    parser.add_argument(
        "-s",
        "--source",
        # default="/mnt/data2/ISCX-VPN-NonVPN-2016/ISCX-VPN-NonVPN-App",
        default='/root/autodl-tmp/Flow-MAE/data/cicIdsEncrypted',
        help="path to the directory containing raw pcap files",
    )
    parser.add_argument(
        "-d",
        "--dest",
        # default="train_test_data/ISCX-VPN-NonVPN-2016-App",
        default='/root/autodl-tmp/Flow-MAE/data/cicIdsEncryptedtest',
        help="path to the directory for persisting preprocessed files",
    )
    parser.add_argument(
        "-n",
        "--njob",
        type=int,
        default=8,
        help="num of executors",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=5,
        help="maximum batch size for processing packets",
    )
    parser.add_argument(
        "--output-batch-size",
        type=int,
        default=5000,
        help="maximum batch for processing packets",
    )
    parser.add_argument(
        "-t"
        "--transform-type",
        choices=["adaptor", "AdaptorSpark"],
        default="AdaptorSpark",
        help="specify the type of transform_pcap to use",
    )
    parser.add_argument(
        "-a",
        "--aggregator",
        choices=["adaptor", "pysparkaggregator"],
        default="pysparkaggregator",
        help="Aggregator type to use, e.g., 'pyspark'"
    )

    args = parser.parse_args()
    return args

def createParquet(dataset):
    # path = '/root/autodl-tmp/Flow-MAE/data/encryptedtestData2/dataset'
    # dataset = load_from_disk(path)
    print(dataset)
    processed_data = {
        key: [row[key] for row in dataset]
        for key in ('x', 'feature_len', 'labels')
    }
    print("startdf")
    df = pd.DataFrame(processed_data)
    print("ok")
    return df


def createTrainAndTest(table):
    aggregator = PySparkAggregator()
    source_directory = "/root/autodl-tmp/Flow-MAE/data/testdata/evalDataProbe/datasetParquet"
    target_directory = "/root/autodl-tmp/Flow-MAE/data/testdata/evalDataProbe/dataParquetused"
    min_feature_len = 100  # 可选，最小特征长度
    sample_per_label = 10000  # 可选，每个标签要保留的样本数量
    test_ratio = 0.1  # 可选，测试集比例
    print("start")
    train_df, test_df=aggregator.aggregator2(table, source_directory, target_directory, min_feature_len, sample_per_label, test_ratio)
    print("ok")
    return train_df, test_df


def lowercase_condition(example):
    return {'x': example['x'], 'feature_len': example['feature_len'], "labels": example['labels']}


def remove_outer_array(arr):
    return arr[0]  # 假设每个元素都是形如 [array[]] 的结构，直接返回内层的数组


def process_dataset(path, mode):
    dataset = load_from_disk(path)
    print(dataset)
    # dataset.to_csv(f'/root/autodl-tmp/Flow-MAE/data/encryptedtestData2/' + mode + 'data.csv')
    # processed_data = dataset.map(lowercase_condition,batched=True)
    dataset.set_format(type="pandas")
    df = dataset[:]
    columns = df.columns.tolist()
    columns[0], columns[1], columns[2] = columns[1], columns[2], columns[0]
    df = df[columns]
    print("start remove")
    df['x'] = df['x'].apply(remove_outer_array)
    print("remove ok")
    array_data = df.values
    # 将 NumPy 数组转换为列表
    data_list = array_data.tolist()
    num_samples = int(1 * len(data_list))
    use_data_list = random.sample(data_list, num_samples)
    print("use_data_list大小：", len(use_data_list))

    return use_data_list


def save_to_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def createPKL():
    train_path = '/root/autodl-tmp/Flow-MAE/data/testdata/evalDataProbe/dataParquetused/dataset_dict/train'
    test_path = '/root/autodl-tmp/Flow-MAE/data/testdata/evalDataProbe/dataParquetused/dataset_dict/test'
    train_file_path = '/root/autodl-tmp/Flow-MAE/data/testdata/evalDataProbe/pretrain4096_0.75train.pkl'
    test_file_path = '/root/autodl-tmp/Flow-MAE/data/testdata/evalDataProbe/pretrain4096_0.75test.pkl'

    try:
        processed_data_train = process_dataset(train_path, mode="train")
        processed_data_test = process_dataset(test_path, mode="test")
        with open(train_file_path, 'wb') as train_file:
            pickle.dump(processed_data_train, train_file)
        with open(test_file_path, 'wb') as test_file:
            pickle.dump(processed_data_test, test_file)
        print("Data saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    # train_path = '/root/autodl-tmp/Flow-MAE/data/testdata/dataParquetused/dataset_dict/train'
    # test_path = '/root/autodl-tmp/Flow-MAE/data/testdata/dataParquetused/dataset_dict/test'
    # train_file_path = '/root/autodl-tmp/Flow-MAE/data/IDSTestBeforeTrain/pretrain4096_0.75test.pkl'
    # test_file_path = '/root/autodl-tmp/Flow-MAE/data/IDSTestBeforeTrain/pretrain4096_0.75train.pkl'
    #
    # with Pool() as pool:
    #     processed_data_train = list(tqdm(pool.imap(process_dataset, [train_path]), total=1))
    #     processed_data_test = list(tqdm(pool.imap(process_dataset, [test_path]), total=1))
    #
    # save_to_pickle(processed_data_train, train_file_path)
    # save_to_pickle(processed_data_test, test_file_path)

    print("Data saved successfully.")


def loadPKL(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        print("长度为：", len(data))


def normalizeData(train_df, test_df):
    train_parquet_path = "/root/autodl-tmp/Flow-MAE/data/testdata/evalDataProbe/dataParquetused/train.parquet"
    test_parquet_path = "/root/autodl-tmp/Flow-MAE/data/testdata/evalDataProbe/dataParquetused/test.parquet"

    dataset_dict = process_parquet_to_datasetdict(train_parquet_path=train_parquet_path,
                                                  test_parquet_path=test_parquet_path, feature_len_max=1024)
    print(dataset_dict)
    print_label_counts(dataset_dict)

    dataset_dict = DatasetDict.load_from_disk(Path(train_parquet_path).with_name(DATASETS_FILE).as_posix())
    print(dataset_dict)
    return dataset_dict


def loadJSONL(path):
    with open(path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            print(item)
            print("--------------")


def convert_pcapng_to_pcap(input_dir, output_dir):
    # 获取指定目录下所有以 .pcapng 结尾的文件
    pcapng_files = [file for file in os.listdir(input_dir) if file.endswith(".pcapng")]

    # 遍历每个文件并执行转换命令
    for pcapng_file in pcapng_files:
        input_file_path = os.path.join(input_dir, pcapng_file)
        output_file_path = os.path.join(output_dir, pcapng_file[:-7] + ".pcap")  # 修改文件后缀为 .pcap
        command = f"tshark -F pcap -r {input_file_path} -w {output_file_path}"
        os.system(command)


def organize_pcap_files(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    for file in files:
        # 检查文件是否为pcap文件
        if file.endswith('.pcap'):
            # 获取文件名（不带扩展名）
            file_name = os.path.splitext(file)[0]

            # 创建文件夹
            dest_folder = os.path.join(folder_path, file_name)
            os.makedirs(dest_folder, exist_ok=True)

            # 移动pcap文件到对应文件夹
            shutil.move(os.path.join(folder_path, file), os.path.join(dest_folder, file))

def main(pcap_path,args):
    # target_dir_path = Path(dest)
    # clean_dirs(target_dir_path)
    dataset=None

    with multiprocessing.Manager() as manager:
        packet_queue = manager.Queue(maxsize=100)
        result_queue = manager.Queue(maxsize=100)
        with ProcessPoolExecutor(max_workers=args.njob) as executor:
            futures = []
            future = executor.submit(
                read_and_fetch_packets,
                packet_queue, pcap_path,
                args.output_batch_size, args.max_batch)
            futures.append(future)

            # 创建消费者进程
            consumer_process = multiprocessing.Process(
                target=transform_pcap,
                args=(packet_queue, len(futures),result_queue)
            )
            consumer_process.start()

            # 等待生产者进程结束
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Failed to read pcap file: {e}")

            # 通知消费者进程结束
            packet_queue.put(None)

        # 等待消费者进程结束
        consumer_process.join()
        result=result_queue.get()
    return result
    # PySparkAggregator.aggregator(tmp_dir, target_dir_path, 1024, 5000)



if __name__ == "__main__":
    # # 可以根据需求自定义这些值
    t1 = time.time()
    args = get_args()
    # args={}
    # args['source']=""
    config.HF_DATASETS_CACHE = '/root/autodl-tmp/cache/huggingface/datasets'
    pcap_path="/root/autodl-tmp/Flow-MAE/data/testdata/evalDataProbe/merge.pcap"

    dataset=main(pcap_path,args)
    table=createParquet(dataset)
    train_df, test_df=createTrainAndTest(table)
    dataset_dict=normalizeData(train_df, test_df)
    createPKL()

    t2 = time.time()
    print(f"duration: {t2 - t1:.2f} seconds")
