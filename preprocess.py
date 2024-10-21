import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Tuple
from datasets import Dataset, Features, Value, Array2D
import torch
from datasets import DatasetDict, Dataset, Features
from torchvision.transforms import Normalize
from transformers import HfArgumentParser

from dataset.dataset_functions import multiprocess_merge, SEQ_LENGTH_BYTES, SEQ_HEIGHT, STREAM, PAYLOAD, \
    multiprocess_merge_continues
from dataset.dataset_dicts import DatasetDicts
from dataset.files import GlobFiles
from preprocess.pcap import copy_folder, FileRecord, extract_tcp_udp, \
    split_seesionns, trim_seesionns, traverse_folder_recursive, rm_small_pcap, json_seesionns
from preprocess.pipeline import CopyStage, Pipeline, TraverseStage
from pretrain.functions import tensor_pad
from utils.arguments import PreprocessArguments, DataTrainingArguments, StageArguments
from utils.file_utils import str2path
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_args() -> Tuple[PreprocessArguments, DataTrainingArguments]:
    parser = HfArgumentParser((PreprocessArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()
    return args


def tshark_extract_pipeline(pre_args):
    """
    该函数根据预先设定的参数，执行一系列网络数据处理步骤，主要用于提取网络数据并进行预处理。

    参数:
    pre_args: 预先设定的参数，包括数据集的源文件夹路径、目标文件夹路径和输出文件夹路径等。

    使用方式:
    tshark_extract_pipeline(pre_args)

    输入参数:
    pre_args: 预先设定的参数对象，其中应包含以下属性:
        - dataset_src_root_dir: 数据集源文件夹路径
        - dataset_dst_root_dir: 数据集目标文件夹路径
        - output_dir: 输出文件夹路径
        - num_workers: 并行处理的工作线程数量
        - trim_folder: 要处理的数据集的文件夹名称
        - tcp_udp_folder: 存储提取的TCP/UDP数据的文件夹名称
        - min_packet_num: 最小数据包数量阈值
        - min_file_size: 最小文件大小阈值
        - split_session_folder: 分割会话的文件夹名称
        - trim_time_folder: 存储时间戳修剪后数据的文件夹名称
        - split_packets_folder: 分割数据包的文件夹名称
        - json_folder: 存储转换为JSON格式的数据的文件夹名称
        - packet_length: 数据包长度
        - time_window: 时间窗口大小
        - max_packet_num: 最大数据包数量阈值
        - packet_num: 每个文件中的数据包数量阈值
        - splitcap_path: 分割文件的路径

    """
    # 将输入的文件路径字符串转换为路径对象
    src_root, dst_root, output_root = str2path(
        pre_args.dataset_src_root_dir,
        pre_args.dataset_dst_root_dir,
        pre_args.output_dir,
    )

    # 确保源文件夹存在
    assert src_root.is_dir()
    # 创建目标文件夹，如果不存在则递归创建
    dst_root.mkdir(parents=True, exist_ok=True)
    # 创建输出文件夹，如果不存在则递归创建
    output_root.mkdir(parents=True, exist_ok=True)

    # 使用偏函数设置部分参数
    stage_args = partial(
        StageArguments,
        output_dir=output_root,
        src_file=output_root.joinpath("src.txt"),
        dst_file=output_root.joinpath("dst.txt"),
        num_workers=pre_args.num_workers
    )

    pipeline_args = [
        # 第一个阶段，复制文件并调用外部脚本处理数据
        (
            CopyStage,  # 复制阶段类
            stage_args(  # 阶段参数设置
                name="trim_sessions",  # 阶段名称
                category="CopyStage",  # 阶段类别
                src_folder=src_root,  # 源文件夹路径
                dst_folder=dst_root.joinpath(pre_args.trim_folder),  # 目标文件夹路径
                cmd="bash preprocess/trim_length.sh {1} {2} "  # 外部命令，使用bash脚本处理数据
                    f"{pre_args.packet_length}"  # 传递的参数：数据包长度
            )
        ),
        (
            CopyStage,  # 复制阶段类
            stage_args(  # 阶段参数设置
                name="extract_tcp_udp",  # 阶段名称：提取TCP和UDP数据包
                category="CopyStage",  # 阶段类别：复制阶段
                src_folder=dst_root.joinpath(pre_args.trim_folder),  # 源文件夹路径：预处理后的数据存储路径与trim_folder相结合
                dst_folder=dst_root.joinpath(pre_args.tcp_udp_folder),  # 目标文件夹路径：存储提取的TCP和UDP数据包的文件夹路径
                cmd="bash preprocess/extract.sh ",  # 外部命令，用于提取TCP和UDP数据包的bash脚本路径
                num_workers=8  # 并行处理的工作线程数：指定同时执行的工作线程数量
            )
        ),
        (
            TraverseStage,  # 遍历阶段类
            stage_args(  # 阶段参数设置
                name="rm_small_pcap",  # 阶段名称：移除小型PCAP文件
                category="TraverseStage",  # 阶段类别：遍历阶段
                src_folder=dst_root.joinpath(pre_args.tcp_udp_folder),  # 源文件夹路径：存储提取的TCP和UDP数据包的文件夹路径
                cmd="bash preprocess/remove.sh {1} "  # 外部命令，用于移除小型PCAP文件的bash脚本路径
                    f"{pre_args.min_packet_num} "  # 传递的参数：最小数据包数量
            )
        ),

        (
            CopyStage,  # 复制阶段类
            stage_args(  # 阶段参数设置
                name="split_sessions",  # 阶段名称：拆分会话
                category="CopyStage",  # 阶段类别：复制阶段
                src_folder=dst_root.joinpath(pre_args.trim_folder),  # 源文件夹路径：预处理后的数据存储路径与trim_folder相结合
                dst_folder=dst_root.joinpath(pre_args.split_session_folder),  # 目标文件夹路径：存储拆分后会话的文件夹路径
                file2folder=True,  # 文件转文件夹标志：表示是否将文件按照会话拆分为多个文件夹
                cmd="bash preprocess/split_sessions.sh {1} {2} "  # 外部命令，用于拆分会话的bash脚本路径
                    f"{pre_args.min_packet_num} "  # 传递的参数：最小数据包数量
                    f"{pre_args.min_file_size} "  # 传递的参数：最小文件大小
                    f"flow "  # 拆分方式：按流拆分
                    f"{pre_args.splitcap_path} ",  # 传递的参数：拆分工具的路径
            )
        ),

        (
            CopyStage,  # 复制阶段类
            stage_args(  # 阶段参数设置
                name="trim_sessions",  # 阶段名称：缩减会话时间
                category="CopyStage",  # 阶段类别：复制阶段
                src_folder=dst_root.joinpath(pre_args.split_session_folder),  # 源文件夹路径：存储拆分后会话的文件夹路径
                dst_folder=dst_root.joinpath(pre_args.trim_time_folder),  # 目标文件夹路径：存储缩减后会话的文件夹路径
                cmd="bash preprocess/trim.sh {1} {2} "  # 外部命令，用于缩减会话时间的bash脚本路径
                    f"{pre_args.time_window} "  # 传递的参数：时间窗口大小
            )
        ),

        (
            CopyStage,  # 复制阶段类
            stage_args(  # 阶段参数设置
                name="split_packets",  # 阶段名称：拆分数据包
                category="CopyStage",  # 阶段类别：复制阶段
                src_folder=dst_root.joinpath(pre_args.tcp_udp_folder),  # 源文件夹路径：存储提取的TCP和UDP数据包的文件夹路径
                dst_folder=dst_root.joinpath(pre_args.split_packets_folder),  # 目标文件夹路径：存储拆分后数据包的文件夹路径
                cmd="bash preprocess/split_pkt.sh {1} {2} "  # 外部命令，用于拆分数据包的bash脚本路径
                    f"{pre_args.min_packet_num} "  # 传递的参数：最小数据包数量
                    f"{pre_args.max_packet_num} "  # 传递的参数：最大数据包数量
                    f"{pre_args.packet_num} "  # 传递的参数：指定的数据包数量
            )
        ),

        (
            TraverseStage,  # 遍历阶段类
            stage_args(  # 阶段参数设置
                name="rm_small_pcap",  # 阶段名称：移除小型PCAP文件
                category="TraverseStage",  # 阶段类别：遍历阶段
                src_folder=dst_root.joinpath(pre_args.split_packets_folder),  # 源文件夹路径：存储拆分后数据包的文件夹路径
                cmd="bash preprocess/remove.sh {1} "  # 外部命令，用于移除小型PCAP文件的bash脚本路径
                    f"{pre_args.min_packet_num} "  # 传递的参数：最小数据包数量
            )
        ),

        (
            CopyStage,  # 复制阶段类
            stage_args(  # 阶段参数设置
                name="json_sessions",  # 阶段名称：转换为JSON格式
                category="CopyStage",  # 阶段类别：复制阶段
                src_folder=dst_root.joinpath(pre_args.trim_folder),  # 源文件夹路径：预处理后的数据存储路径与trim_folder相结合
                dst_folder=dst_root.joinpath(pre_args.json_folder),  # 目标文件夹路径：存储转换为JSON格式后的文件夹路径
                cmd="bash preprocess/pcap2json.sh {1} {2} "  # 外部命令，用于将PCAP文件转换为JSON格式的bash脚本路径
                    f"{pre_args.min_packet_num} "  # 传递的参数：最小数据包数量
                    f"{pre_args.min_file_size} "  # 传递的参数：最小文件大小
            )
        ),

    ]

    pipeline = Pipeline(pipeline_args)  # 创建数据处理管道
    pipeline.run()  # 执行数据处理管道


# def csv2datasets_pipeline():
#     tcp_files = GlobFiles(
#         root="/mnt/data/IDS2018all/json_sessions",
#         file_pattern="*TCP.csv",
#         threshold=200
#     )
#     pprint(tcp_files.files)
#
#     dataset = DatasetDicts.from_csv_parallel(tcp_files.files)
#     pprint(dataset.shape)
#
#     dataset = dataset.map_parallel(multiprocess_merge, num_proc=16, batch_size=16,
#                                    columns_to_keep=[STREAM, PAYLOAD])
#     pprint(dataset.shape)
#
#     flatten = dataset.flatten_to_dataset_dict(axis=1)
#     pprint(flatten.shape)
#     flatten.save_to_disk("/mnt/data2/IDS2018Train")
#     json.dump(
#         obj={
#             "shape": flatten.shape,
#             "rows": flatten.num_rows
#         },
#         fp=open("/mnt/data2/IDS2018Train/record.json", "w")
#     )


# 定义函数：将CSV文件转换为数据集并进行一系列处理和保存操作
def csv2datasets_pipeline():
    # 选择所有符合指定文件模式的CSV文件
    tcp_files = GlobFiles(
        root="./data3/FlowTrans/IDS2018_Finetune/json_sessions",  # CSV文件所在的根目录
        file_pattern="*.csv",  # 文件名的通配符表达式
        threshold=100  # 文件大小阈值，仅选择大小超过该阈值的文件
    )
    # 打印选1
    # 中的CSV文件列表
    # pprint(tcp_files.files)

    # 从CSV文件中读取数据并生成数据集字典
    dataset = DatasetDicts.from_csv_parallel(tcp_files.files)
    # 打印数据集的形状信息
    pprint(dataset.shape)

    # 并行地对数据集进行映射操作
    dataset = dataset.map_parallel(
        multiprocess_merge_continues,  # 映射函数，对数据集进行处理
        num_proc=48,  # 并行处理的进程数
        batch_size=9,  # 每个处理批次的大小
        columns_to_keep=[STREAM, PAYLOAD]  # 要保留的列
    )
    # 打印处理后的数据集的形状信息
    pprint(dataset.shape)

    # 将数据集展平为一个数据集字典
    flatten = dataset.flatten_to_dataset_dict(axis=1)
    # 将展平后的数据集保存到指定路径
    flatten.save_to_disk("./data3/FlowTrans/IDS2018_FinetuneData")
    # 打印展平后的数据集的形状信息
    pprint(flatten.shape)

    # 将数据集的形状信息和行数信息以JSON格式写入文件
    json.dump(
        obj={
            "shape": flatten.shape,  # 形状信息
            "rows": flatten.num_rows  # 行数信息
        },
        fp=open("./data3/FlowTrans/IDS2018_FinetuneData/record.json", "w")  # 写入文件的路径
    )


def preprocess_images(example):
    example["pixel_values"] = [
        tensor_pad(example[PAYLOAD]).div(255)
    ]
    return example


if __name__ == "__main__":
    t1 = time.time()

    # pre_args = PreprocessArguments()
    # tshark_extract_pipeline(pre_args)

    csv2datasets_pipeline()

    t2 = time.time()
    print('run time: %s sec' % time.strftime("%H:%M:%S", time.gmtime(t2 - t1)))
    print('hello')
