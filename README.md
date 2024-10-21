
"/root/autodl-tmp/Flow-MAE/yxWork/preprocessPcapEval.py" 数据预处理，直接运行，当前只针对单个pcap包的预处理，可自行更改函数
pretrain.py运行：sh pretrain.sh
finetune.py运行：sh finetune.sh
具体模型参数，文件路径详见sh文件并修改

Here's a breakdown of the structure and the key components of the repository:

- **dataset/**: This directory contains common data operation codes and scripts. It serves as the primary hub for managing and processing data used across various stages of the project.
- **eval.py**: A Python script dedicated to evaluating the performance of trained models.
- **eval.sh**: Shell script facilitating the execution of the evaluation process.
- **finetune.py**: Python script that provides functionalities for fine-tuning the models.
- **finetune.sh**: Shell script facilitating the fine-tuning process.
- **model/**: Contains the architecture and model-related files that serve as the backbone for the `Flow-MAE` system.
- **notebooks/**: Directory for Jupyter notebooks related to data visualization and model interpretability.
- **preprocess/**: This directory hosts scripts and utilities for data preprocessing and transformation.
- **preprocess.py**: Main preprocessing script to get the data ready for training and evaluation.
- **pretrain/**: A directory containing resources and scripts dedicated to the pre-training phase of models.
- **pretrain.py**: The primary script guiding the pre-training phase.
- **pretrain.sh**: Shell script aiding in executing the pre-training process.
- **tools/**: A utility directory storing tools and scripts that aid in various tasks throughout the project.
- **utils/**: Contains utility functions and helper scripts commonly used across the repository.
- **student.py/**: 蒸馏后的学生模型
## Building and Running the Code

### System Requirements

The experiments included in this repository have been tested on a testbed equipped with an i7-12700K CPU (8 P-cores @4.7GHz and 4 E-cores @3.6GHz), 64 GB DDR5 DRAM (@6000MT/s), and two NVIDIA GeForce 3090Ti GPUs (24 GB of GDDR6X memory each). The software environment of the testbed includes Ubuntu 22.04.1 LTS (kernel 5.15.0-50), Python 3.8.13, PyTorch 1.12.0, and CUDA 11.6.
