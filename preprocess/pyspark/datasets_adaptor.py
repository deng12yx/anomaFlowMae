import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Optional, Union, Dict, List, Set

import numpy as np
import torch
from datasets import DatasetDict, load_dataset, ClassLabel, Features, Value, Sequence

from preprocess import FEATURE_COL, LABEL_COL, FEATURE_LEN_COL, FEATURE_LEN_MAX, DATASETS_FILE, LABEL_MAPPING_FILE

from datasets import config


def preprocess_function(
        record: Union[List[Dict], Dict],  # 输入的数据记录，可以是单个字典或字典列表
        feature_col: Optional[str] = FEATURE_COL,  # 特征列的名称，默认为 FEATURE_COL
        label_col: Optional[str] = LABEL_COL,  # 标签列的名称，默认为 LABEL_COL
        feature_len_col: Optional[str] = FEATURE_LEN_COL,  # 特征长度列的名称，默认为 FEATURE_LEN_COL
        feature_len_max: Optional[int] = FEATURE_LEN_MAX,  # 特征长度的最大值，默认为 FEATURE_LEN_MAX
) -> dict:  # 返回类型为字典
    # 获取特征列数据
    byte_data = record[feature_col]
    # 将单个特征数据转换为列表形式
    if not isinstance(byte_data, list):
        byte_data = [byte_data]

    # 将特征数据转换为 numpy 数组列表
    np_data_list = [
        np.frombuffer(data, dtype=np.uint8)[:feature_len_max] / 255.0 for data in byte_data
    ]

    # 创建一个填充后的数据数组，用于存储特征数据
    padded_data_array = np.zeros((len(np_data_list), 1, feature_len_max), dtype=np.float32)
    # 遍历每个特征数据，进行填充
    for i, np_data in enumerate(np_data_list):
        padded_data_array[i, 0, :len(np_data)] = np_data

    # 将填充后的数据数组转换为 PyTorch 张量
    padded_data_tensor = torch.from_numpy(padded_data_array)
    # 如果张量的第一个维度大小为1，则压缩该维度
    if padded_data_tensor.size(0) == 1:
        padded_data_tensor = padded_data_tensor.squeeze(0)

    # 将填充后的数据张量更新到输入记录中的特征列
    record[feature_col] = padded_data_tensor

    return record  # 返回更新后的数据记录字典


def read_parquet_to_datasetdict(train_parquet_path: str, test_parquet_path: str = None) -> DatasetDict:
    if test_parquet_path is None:
        # Load the dataset from the train_parquet_path
        dataset_dict = load_dataset(train_parquet_path)
    else:
        # Load the datasets from Parquet files
        train_dataset = load_dataset(train_parquet_path)["train"]
        test_dataset = load_dataset(test_parquet_path)["train"]
        # Add the datasets to the DatasetDict
        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    return dataset_dict


    return dataset_dict
def get_label_mapping(all_labels: Set):
    # Check if all labels are numeric
    all_numeric = all(isinstance(label, (int, float)) for label in all_labels)

    if not all_numeric:
        # Replace special characters with underscores
        all_labels = {
            re.sub('[^0-9a-zA-Z]+', '_', str(label))
            for label in all_labels
        }

        # Ignore case and sort
        all_labels = sorted(all_labels, key=lambda label: label.lower())
    else:
        # Sort numeric labels
        all_labels = sorted(all_labels)

    label_to_id = {label: i for i, label in enumerate(all_labels)}

    return label_to_id, all_labels


def get_label_mapping_from_dataset_dict(dataset_dict, label_col: Optional[str] = LABEL_COL):
    # Create the label name to ID mapping
    all_labels = set()
    for dataset in dataset_dict.values():
        all_labels.update(set(dataset.unique(label_col)))

    return get_label_mapping(all_labels)


def process_parquet_to_datasetdict(
        train_parquet_path: str,  # 训练数据的 Parquet 文件路径
        test_size: float = 0,  # 测试集的比例，默认为0，表示不创建测试集
        test_parquet_path: Optional[str] = None,  # 测试数据的 Parquet 文件路径，可选
        save_path: Optional[str] = None,  # 数据集保存路径，可选
        feature_col: Optional[str] = FEATURE_COL,  # 特征列名称，默认为 FEATURE_COL
        label_col: Optional[str] = LABEL_COL,  # 标签列名称，默认为 LABEL_COL
        feature_len_col: Optional[str] = FEATURE_LEN_COL,  # 特征长度列名称，默认为 FEATURE_LEN_COL
        feature_len_max: Optional[int] = FEATURE_LEN_MAX,  # 特征长度最大值，默认为 FEATURE_LEN_MAX
) -> DatasetDict:  # 返回类型为 DatasetDict 对象
    # 读取 Parquet 文件到 DatasetDict 对象
    dataset_dict = read_parquet_to_datasetdict(train_parquet_path, test_parquet_path)
    # 对数据集进行预处理
    dataset_dict = dataset_dict.map(preprocess_function, batched=True)

    # 获取标签映射和标签列表
    label_to_id, label_list = get_label_mapping_from_dataset_dict(dataset_dict)
    # 创建标签对象
    class_label = ClassLabel(num_classes=len(label_list), names=label_list)

    # 创建包含数据集所有列的 Features 对象
    features = Features({
        label_col: class_label,  # 标签列
        feature_col: Sequence(feature=Sequence(
            feature=Value(dtype='float32', id=None), length=feature_len_max, id=None),
            length=1, id=None),  # 特征列
        feature_len_col: Value('int64'),  # 特征长度列
    })
    # 将数据集转换为指定特征类型的数据集
    dataset_dict = dataset_dict.cast(features)

    # 分割训练集和测试集
    if "test" not in dataset_dict and test_size > 0:
        dataset_dict = dataset_dict["train"].train_test_split(test_size=test_size, shuffle=True,
                                                              stratify_by_column=label_col)
    # 设置保存路径
    if save_path is None:
        save_path = Path(train_parquet_path).with_name(DATASETS_FILE)
    # 删除已存在的保存路径
    if save_path.exists():
        shutil.rmtree(save_path)

    # 将数据集保存到磁盘
    dataset_dict.save_to_disk(save_path)
    # 将标签映射保存到磁盘
    with open(save_path / LABEL_MAPPING_FILE, "w") as f:
        json.dump(label_to_id, f, indent=4)

    return dataset_dict  # 返回处理后的 DatasetDict 对象


def process_parquet_to_datasetdict2(
        train_parquet_path: str,  # 训练数据的 Parquet 文件路径
        test_size: float = 0,  # 测试集的比例，默认为0，表示不创建测试集
        test_parquet_path: Optional[str] = None,  # 测试数据的 Parquet 文件路径，可选
        save_path: Optional[str] = None,  # 数据集保存路径，可选
        feature_col: Optional[str] = FEATURE_COL,  # 特征列名称，默认为 FEATURE_COL
        label_col: Optional[str] = LABEL_COL,  # 标签列名称，默认为 LABEL_COL
        feature_len_col: Optional[str] = FEATURE_LEN_COL,  # 特征长度列名称，默认为 FEATURE_LEN_COL
        feature_len_max: Optional[int] = FEATURE_LEN_MAX,  # 特征长度最大值，默认为 FEATURE_LEN_MAX
) -> DatasetDict:  # 返回类型为 DatasetDict 对象
    # 读取 Parquet 文件到 DatasetDict 对象
    dataset_dict = read_parquet_to_datasetdict(train_parquet_path, test_parquet_path)
    # 对数据集进行预处理
    dataset_dict = dataset_dict.map(preprocess_function, batched=True)

    # 获取标签映射和标签列表
    label_to_id, label_list = get_label_mapping_from_dataset_dict(dataset_dict)
    # 创建标签对象
    class_label = ClassLabel(num_classes=len(label_list), names=label_list)

    # 创建包含数据集所有列的 Features 对象
    features = Features({
        label_col: class_label,  # 标签列
        feature_col: Sequence(feature=Sequence(
            feature=Value(dtype='float32', id=None), length=feature_len_max, id=None),
            length=1, id=None),  # 特征列
        feature_len_col: Value('int64'),  # 特征长度列
    })
    # 将数据集转换为指定特征类型的数据集
    dataset_dict = dataset_dict.cast(features)

    # 分割训练集和测试集
    if "test" not in dataset_dict and test_size > 0:
        dataset_dict = dataset_dict["train"].train_test_split(test_size=test_size, shuffle=True,
                                                              stratify_by_column=label_col)
    # 设置保存路径
    if save_path is None:
        save_path = Path(train_parquet_path).with_name(DATASETS_FILE)
    # 删除已存在的保存路径
    if save_path.exists():
        shutil.rmtree(save_path)

    # 将数据集保存到磁盘
    dataset_dict.save_to_disk(save_path)
    # 将标签映射保存到磁盘
    with open(save_path / LABEL_MAPPING_FILE, "w") as f:
        json.dump(label_to_id, f, indent=4)

    return dataset_dict  # 返回处理后的 DatasetDict 对象


def print_label_counts(dataset_dict: DatasetDict, label_col: Optional[str] = LABEL_COL):
    for dataset_name, dataset in dataset_dict.items():
        label_counts = Counter(dataset[label_col])

        print(f"Dataset {dataset_name}:")
        for label_name, count in label_counts.items():
            print(f"Label {label_name}: {count}")
        print()


if __name__ == "__main__":
    config.HF_DATASETS_CACHE = '/root/autodl-tmp/cache/huggingface/datasets'
    train_parquet_path = "/root/autodl-tmp/Flow-MAE/data/testdata/dataParquetused/train.parquet"
    test_parquet_path = "/root/autodl-tmp/Flow-MAE/data/testdata/dataParquetused/test.parquet"

    dataset_dict = process_parquet_to_datasetdict(train_parquet_path=train_parquet_path,
                                                  test_parquet_path=test_parquet_path, feature_len_max=1024,
                                                  test_size=0.1)

    print(dataset_dict)
    print_label_counts(dataset_dict)

    dataset_dict = DatasetDict.load_from_disk(Path(train_parquet_path).with_name(DATASETS_FILE).as_posix())
    print(dataset_dict)
