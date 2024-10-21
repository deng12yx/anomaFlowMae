import os
import pyarrow as pa
from datasets import Dataset
from multiprocessing import Queue
from typing import Tuple

# from preprocess.factory import register_factory, transform_pcap_factory
from preprocess.factory import  transform_pcap_factory
from preprocess.pyspark.process_packet import transform_packet


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
    # dataset.to_csv("/root/autodl-tmp/Flow-MAE/data/testdata/dataset.csv", index=False)  # 如果不想保存行索引，可以设置
    # index=False

    # 将转换后的数据集保存为 Arrow 文件
    dataset_path = os.path.join(output_path, "dataset")
    dataset.save_to_disk(dataset_path)

    print("Dataset processed and saved.")


# @register_factory(transform_pcap_factory, "datasets")
# class TransformPcapAdaptor:
#     def __call__(self, *args, **kwargs):
#         # Your adaptor implementation
#         ...


if __name__ == "__main__":
    pass
