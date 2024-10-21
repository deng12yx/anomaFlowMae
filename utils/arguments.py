from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Callable, Any


@dataclass
class StageArguments:
    """

    """
    name: str = field(
        default=None,
        metadata={"help": "stage name"}
    )
    category: str = field(
        default=None,
        metadata={"help": "stage class"}
    )
    src_folder: Path = field(
        default=None,
        metadata={"help": "source folder of current stage"}
    )
    dst_folder: Path = field(
        default=None,
        metadata={"help": "dst folder of current stage"}
    )
    file2folder: bool = field(
        default=False,
        metadata={"help": "make new folder for file while traverse the folder"}
    )
    output_dir: Path = field(
        default=None,
        metadata={"help": "root directory for output"}
    )
    src_file: Path = field(
        default=None
    )
    dst_file: Path = field(
        default=None
    )
    cmd: str = field(
        default=None
    )
    num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    # run_func: Callable[..., Any] = field(
    #     default=None,
    #     metadata={"help": "make new folder for file while traverse the folder"}
    # )
    kwargs: Dict = field(
        default=None,
        metadata={"help": "other kw args"}
    )


@dataclass
class PreprocessArguments:
    """
    Arguments Preprocess to pcap files to generate dataset_dict.
    """
    name: str = field(
        default="CIC-IDS2018-Finetune",
        metadata={"help": "Preprocess dataset_dict name"}
    )
    output_dir: str = field(
        default="/mnt/data3/FlowTrans/IDS2018_Finetune",
        metadata={"help": "root directory for output"}
    )
    dataset_src_root_dir: str = field(
        # default="/mnt/data2/IDS2018",
        default="../DATA/TrafficClasification/data/IDS2018black",
        metadata={"help": "Source dataset_dict (.pcap) folder directory path"}
    )
    dataset_dst_root_dir: str = field(
        # default="/mnt/data/IDS2018all",
        default="/mnt/data3/FlowTrans/IDS2018_Finetune",
        metadata={"help": "Generated dataset_dict folder directory path"}
    )
    tcp_udp_folder: str = field(
        default="tcp_udp",
        metadata={"help": "folder to store the udp and tcp pcap"}
    )
    split_session_folder: str = field(
        default="split_sessions",
        metadata={"help": "folder to store the split pcap session"}
    )
    splitcap_path: str = field(
        default="./tools/SplitCap.exe",
        metadata={"help": "path to splitcap.exe (https://www.netresec.com/?page=SplitCap), "
                          "which split pcap to sessions"}
    )
    trim_folder: str = field(
        default="trim_sessions",
        metadata={"help": "folder to store the trimmed sessions"}
    )
    trim_time_folder: str = field(
        default="trim_time",
        metadata={"help": "folder to store the trimmed sessions"}
    )
    split_packets_folder: str = field(
        default="split_packets",
        metadata={"help": "folder to store the split packets"}
    )
    time_window: int = field(
        default=3600,
        metadata={"help": "time length (seconds) of the trim slice"}
    )
    packet_length: int = field(
        default=222,
        metadata={"help": "max packet length (bytes) of the trim slice, "
                          "the input sequence length"}
    )
    max_packet_num: int = field(
        default=8,
        metadata={"help": "max packet num (quantity) of the trim slice"}
    )
    min_packet_num: int = field(
        default=4,
        metadata={"help": "min packet num (quantity) of the trim slice"}
    )
    packet_num: int = field(
        default=8,
        metadata={"help": "packet num (quantity) of the input, "
                          "the input sequence num"}
    )
    min_file_size: int = field(
        default=200,
        metadata={"help": "min file size"}
    )
    json_folder: str = field(
        default="json_sessions",
        metadata={"help": "folder to store the json sessions"}
    )
    num_workers: Optional[int] = field(
        default=48,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset_dict to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset_dict to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    # def __post_init__(self):
    #     if self.dataset_name is None and self.train_file is None and self.validation_file is None:
    #         raise ValueError("Need either a dataset_dict name or a training/validation file.")
    #     else:
    #         if self.train_file is not None:
    #             extension = self.train_file.split(".")[-1]
    #             assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #         if self.validation_file is not None:
    #             extension = self.validation_file.split(".")[-1]
    #             assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    #     self.task_name = self.task_name.lower()
