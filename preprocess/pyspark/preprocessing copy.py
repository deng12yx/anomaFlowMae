import argparse
import json
import multiprocessing
import queue
import re
import shutil
import time
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
from ..datasets_engine.adaptor import transform_pcap
from preprocess.pyspark.spark_adapter import read_and_fetch_packets
from preprocess.pyspark.spark_aggregator import PySparkAggregator
def transform_pcap(packet_queue: Queue, num_producers: int, output_path: str):
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
            batch, label = item  # 解包数据
            for packet in batch:  # 遍历数据包批次
                row = preprocess_function(packet, label)  # 预处理数据包并生成数据行
                if row is not None:  # 如果数据行不为空
                    rows.append(row)  # 将数据行添加到列表中
    print("ok！")
    # 从字典列表创建数据集
    dataset = Dataset.from_dict({k: [dic[k] for dic in rows] for k in rows[0]})
    dataset.to_csv("/root/autodl-tmp/Flow-MAE/data/testdata/dataset.csv", index=False)  # 如果不想保存行索引，可以设置
    # index=False

    # 将转换后的数据集保存为 Arrow 文件
    dataset_path = os.path.join(output_path, "dataset")
    dataset.save_to_disk(dataset_path)

    print("Dataset processed and saved.")


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
        default='/root/autodl-tmp/Flow-MAE/data/FinetuneData',
        help="path to the directory containing raw pcap files",
    )
    parser.add_argument(
        "-d",
        "--dest",
        # default="train_test_data/ISCX-VPN-NonVPN-2016-App",
        default='/root/autodl-tmp/Flow-MAE/data/FinetunetestData',
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


def main(args):
    data_dir_path = Path(args.source)
    target_dir_path = Path(args.dest)
    tmp_dir = Path("/tmp/spark_parquet")
    clean_dirs(target_dir_path)
    # clean_dirs(target_dir_path, tmp_dir)

    pcap_dict = PcapDict(str(data_dir_path))

    # 使用 Manager().Queue() 替换 multiprocessing.Queue()
    with multiprocessing.Manager() as manager:
        packet_queue = manager.Queue(maxsize=100)

        # 创建生产者进程
        with ProcessPoolExecutor(max_workers=args.njob) as executor:
            futures = []
            for label, label_path in pcap_dict.items():
                for pcap_path in label_path:
                    future = executor.submit(
                        read_and_fetch_packets,
                        packet_queue, pcap_path,
                        args.output_batch_size, args.max_batch,
                        label, tmp_dir)
                    futures.append(future)

            # 创建消费者进程
            consumer_process = multiprocessing.Process(
                target=transform_pcap,
                args=(packet_queue, len(futures),target_dir_path)
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
    # PySparkAggregator.aggregator(tmp_dir, target_dir_path, 1024, 5000)


def main2(args):
    data_dir_path = Path(args.source)
    target_dir_path = Path(args.dest)
    tmp_dir = Path("/tmp/spark_parquet")
    clean_dirs(target_dir_path)
    # clean_dirs(target_dir_path, tmp_dir)

    pcap_dict = PcapDict(str(data_dir_path))

    # 使用 Manager().Queue() 替换 multiprocessing.Queue()

    packet_queue = queue.Queue()
    futures = []
    for label, label_path in pcap_dict.items():
        for pcap_path in label_path:
            future = read_and_fetch_packets(
                packet_queue, pcap_path,
                args.output_batch_size, args.max_batch,
                label, tmp_dir)
            futures.append(future)
    transform_pcap(packet_queue, len(futures),target_dir_path)


if __name__ == "__main__":
    # # 可以根据需求自定义这些值
    t1 = time.time()
    args = get_args()
    # args={}
    # args['source']=""
    main(args)

    t2 = time.time()
    print(f"duration: {t2 - t1:.2f} seconds")
