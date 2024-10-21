#!/usr/bin/env python3
# coding=utf-8
"""This is a sample Python script. """
import json
import os
import sys
from functools import partial

import datasets
import numpy as np
import pandas as pd
import torch.distributed as dist
import transformers
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    confusion_matrix)
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint, set_seed
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import torch
from dataset.session_dataset import dataset_collate_function, SessionDataSet
from model.mae import ViTMAEConfig, ViTMAEForPreTraining, ViTMAEForImageClassification, \
    ViTMAEForStudentImageClassification_Relation, ViTMAEForStudentImageClassification_Feature, \
        ViTMAEForStudentImageClassification_Response
from pretrain.arguments import ModelArguments, DataTrainingArguments, CustomTrainingArguments
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")


def logger_setup(log_level):
    # Setup logging
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def get_last_ckpt(training_args: TrainingArguments):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def generate_config(model_args):
    # Load pretrained model and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    config = ViTMAEConfig(
        image_size=model_args.image_size,
        patch_size=model_args.patch_size,
        num_channels=model_args.num_channels,
        num_attention_heads=6,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
        attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
        hidden_size=384,  # 缩小 hidden_size
        num_hidden_layers=4,  # 减少隐藏层数量
        intermediate_size=1536,  # 缩小 intermediate_size
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        qkv_bias=True,
        decoder_num_attention_heads=6,  # 缩小 decoder_num_attention_heads
        decoder_hidden_size=128,  # 缩小 decoder_hidden_size
        decoder_num_hidden_layers=4,  # 减少解码器隐藏层数量
        decoder_intermediate_size=512,  # 缩小 decoder_intermediate_size
        mask_ratio=0.75,
        norm_pix_loss=False,
    )
    logger.warning("You are instantiating a new config instance from scratch.")
    if model_args.config_overrides is not None:
        logger.info(f"Overriding config: {model_args.config_overrides}")
        config.update_from_string(model_args.config_overrides)
        logger.info(f"New config: {config}")

    # adapt config
    config.update(
        {
            "mask_ratio": model_args.mask_ratio,
            "norm_pix_loss": model_args.norm_pix_loss,
            "num_labels": model_args.num_labels,
        }
    )

    return config


def create_model(model_args, config, teacher_model = None):
    if model_args.model_type == "response":
        model = ViTMAEForStudentImageClassification_Response(config, teacher_model)
    elif model_args.model_type == "feature":
        model = ViTMAEForStudentImageClassification_Feature(config, teacher_model)
    elif model_args.model_type == "relation":
        model = ViTMAEForStudentImageClassification_Relation(config, teacher_model)
    else:
        model = ViTMAEForImageClassification(config)
    return model


# Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics(p, print_entity=False):
    """Computes accuracy on a batch of predictions"""
    labels = p.label_ids
    # preds = np.argmax(p.predictions, axis=1)

    predictions = np.array(p.predictions, dtype=object)    
    # print(f"Predictions Type: {type(predictions)}")
    # print(f"Predictions Shape: {predictions.shape}")
    # print(f"Predictions Example: {predictions[:2]}")  # 查看前两个样本

    if isinstance(predictions, list):
        preds = [np.argmax(pred) if isinstance(pred, np.ndarray) else pred for pred in predictions]
    else:
        preds = np.argmax(predictions, axis=1)
    # Check if the current process is the main process
    is_main_process = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

    if print_entity and is_main_process:
        # Calculate and print confusion matrix
        conf_matrix = confusion_matrix(labels, preds)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Create classification report dictionary and add class-wise accuracy
        class_report_dict = classification_report(labels, preds, output_dict=True, zero_division=0)
        # Convert dictionary to DataFrame and format columns
        class_report_df = pd.DataFrame(class_report_dict).transpose()
        class_report_df['support'] = class_report_df['support'].astype(int)
        # Print classification report with class-wise accuracy
        print("Classification Report with Class-wise Accuracy:")
        print(class_report_df)

    return {
        "accuracy_score": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='weighted'),
        "recall": recall_score(labels, preds, average='weighted'),
        "f1": f1_score(labels, preds, average='weighted'),
    }


# 生成并保存图表
def plot_metrics(train_results, eval_results, metric_name, output_path):
    epochs = [res['epoch'] for res in train_results]
    train_metrics = [res['metrics'][metric_name] for res in train_results]
    eval_metrics = [res['metrics'][metric_name] for res in eval_results]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, label=f"Train {metric_name}")
    plt.plot(epochs, eval_metrics, label=f"Eval {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Train and Eval {metric_name} over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger_setup(log_level)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    dataset = {}
    dataset['train'] = SessionDataSet(data_args.train_dir, 1024, 64, 16, mode="finetune")
    dataset['test'] = SessionDataSet(data_args.validation_dir, 1024, 64, 16, mode="finetune")
    model_args.num_labels = dataset['train'].num_labels

    # Count the number of labels.
    # dataset = datasets.load_from_disk(data_args.dataset_dir)
    # model_args.num_labels = dataset['train'].features['label'].num_classes
    config = generate_config(model_args)
    last_checkpoint = get_last_ckpt(training_args)
    print(f"model_args.model_type is {model_args.model_type}")
    if model_args.model_type != "None":
        teacher_model = ViTMAEForImageClassification.from_pretrained("./vit-mae-finetuneWithOutProbe/checkpoint-70600")
        model = create_model(model_args, config, teacher_model)
    else:
        model = create_model(model_args, config)
    pacth_size = model.vit.embeddings.patch_size
    pixels_per_patch = pacth_size[0] * pacth_size[1]
    num_patches = model.vit.embeddings.num_patches

    # train_ds = None
    # test_ds = None
    # if training_args.do_train:
    #     dataset_dict = datasets.load_dataset(str(data_args.train_dir))
    #     train_ds = dataset_dict[list(dataset_dict.keys())[0]]
    # if training_args.do_eval:
    #     dataset_dict = datasets.load_dataset(str(data_args.validation_dir))
    #     test_ds = dataset_dict[list(dataset_dict.keys())[0]]

    # Compute absolute learning rate
    total_train_batch_size = (
            training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = training_args.base_learning_rate * total_train_batch_size / 256

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'] if training_args.do_train else None,
        eval_dataset=dataset['test'] if training_args.do_eval else None,
        # data_collator=partial(dataset_collate_function, data_args=data_args),
        compute_metrics=compute_metrics,
    )
    # 初始化列表存储每个 epoch 的结果
    train_epochs_results = []
    eval_epochs_results = []
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        # # 保存每个 epoch 的训练结果
        # for epoch, metrics in enumerate(train_result.metrics['epoch']):
        #     train_epochs_results.append({
        #         "epoch": epoch + 1,
        #         "metrics": metrics
        #     })

    # Evaluation
    if training_args.do_eval:
        trainer.compute_metrics = partial(compute_metrics, print_entity=True)
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)
        # 保存每个 epoch 的评估结果
        for epoch, metrics in enumerate(eval_result['epoch']):
            eval_epochs_results.append({
                "epoch": epoch + 1,
                "metrics": metrics
            })
    # 将结果保存到 JSON 文件
    with open("student_train_results.json", "w") as f:
        json.dump(train_epochs_results, f, indent=4)
    with open("student_eval_results.json", "w") as f:
        json.dump(eval_epochs_results, f, indent=4)
    # 示例：绘制并保存准确度图表
    plot_metrics(train_epochs_results, eval_epochs_results, "accuracy", "accuracy.png")


if __name__ == "__main__":
    main()
