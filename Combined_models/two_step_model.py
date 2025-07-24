import subprocess
import argparse
import configparser
import numpy as np
import pandas as pd
from accelerate import Accelerator
from accelerate.logging import get_logger
import json
from inf_rebel import test_model
from datetime import datetime
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from binary_filter import run_filter
from st2_combine import main_st2
from st1_combine import main_st1
from LLM_run import run_LLM
import os
logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

#print('--------')
available_llms = {
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "dpo": "yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B",
    "una": "fblgit/UNA-TheBeagle-7b-v1",
    "solar": "bhavinjawade/SOLAR-10B-OrcaDPO-Jawade",
    "gpt4": "OpenAI-GPT4"  # Added GPT-4
}
def parse_args():
    
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    ''''''
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default='Joined_data/News_data/test.csv', 
        help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of "
            "training examples to this value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker validation, truncate the number of "
            "validation examples to this value if set."
        ),
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker test, truncate the number of "
            "test examples to this value if set."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the test dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_model_id", 
        type=str, 
        help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--hub_token", 
        type=str, 
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    # Custom Arguments
    parser.add_argument(
        "--add_signal_bias",
        action="store_true",
        help="Whether or not to add signal bias",
    )
    parser.add_argument(
        "--signal_bias_on_top_of_lm",
        action="store_true",
        help="Whether or not to add signal bias",
    )
    parser.add_argument(
        "--postprocessing_position_selector",
        action="store_true",
        help="Whether or not to use postprocessing position selector to control overlap problem.",
    )
    parser.add_argument(
        "--mlp",
        action="store_true",
        help="Whether or not to add MLP layer on top of the pretrained LM.",
    )
    parser.add_argument(
        "--signal_classification",
        action="store_true",
        help="Conduct signal classification to verify whether we need to detect signal span.",
    )
    parser.add_argument(
        "--pretrained_signal_detector",
        action="store_true",
        help="Whether to use pretrained signal detector",
    )
    parser.add_argument( #"outs_test/signal_cls"
        "--signal_model_and_tokenizer_path",
        type=str,
        help="Path to pretrained signal detector model.",
    )
    parser.add_argument(
        "--beam_search",
        action="store_true",
        help="Whether to do bean search selection.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="classifier dropout rate",
    )
    parser.add_argument(
        "--use_best_model",
        action="store_true",
        help="Activate to use model with Highest Overall F1 score, else defaults to Last model.",
    )
    parser.add_argument(
        "--load_checkpoint_for_test",
        type=str,
        default=None,
        help="classifier dropout rate",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to train models from scratch.",
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Whether to use model to predict on test set.",
    )
    parser.add_argument(
        "--augmentation_file",
        type=str,
        default=None,
        help="Whether to use pretrained signal detector",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Whether to use pretrained signal detector",
    )
    #parser.add_argument('--filter_threshold', type=float, required=True, help='Threshold for classification')
    
    
    
    
    
    
    parser.add_argument('--use_cpu', action="store_true", help='To tell that the model should only use cpu')
    
    
    
    #rebel
    parser.add_argument(
        "--rebel_inf_model_name_or_path",
        type=str,
        default='pretrained_models/st3/rebel_st3/model_our_data_gpt_augmented.pth',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    
    #rebel
    
    
    #parser.add_argument('--filter_model_path', type=str, help='Path to model')
    
    
    #st1
    
    
    
    
    '''
    parser.add_argument(
        "--st1_get_process_log_level",
        type=str,
        default=None,
        help="output for csv file",
    )
    
    parser.add_argument(
        "--st1_do_train",
        action="store_true",
        help="Whether to train models from scratch.",
    )
    
    parser.add_argument(
        "--st1_output_dir",
        type=str,
        default=None,
        help="output for csv file",
    )
    
    
    parser.add_argument(
        "--st1_do_predict",
        action="store_true",
        help="sets the model to predict",
    )
    parser.add_argument('--st1_use_cpu', action="store_true", help='To tell that the model should only use cpu')
    parser.add_argument('--st1_main_process_first', action="store_true", help='To tell that the model should only use cpu')
    
    '''
    parser.add_argument(
        "--st1_do_predict",
        action="store_true",
        help="sets the model to predict",
    )
    parser.add_argument('--st1_use_cpu', action="store_true", help='To tell that the model should only use cpu')
    
    
    parser.add_argument(
        "--st1_output_dir",
        type=str,
        default='outs/2sft_st1_base_new',
        help="output for csv file",
    )
    parser.add_argument(
        "--st1_task_name",
        type=str,
        default='cola',
        help="The name of the task to train on: "
    )
    parser.add_argument(
        "--st1_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--st1_dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--st1_max_seq_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument(
        "--st1_overwrite_cache",
        action='store_true',
        help="Overwrite the cached preprocessed datasets or not."
    )
    parser.add_argument(
        "--st1_pad_to_max_length",
        action='store_true',
        default=True,
        help="Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch."
    )
    parser.add_argument(
        "--st1_max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value if set."
    )
    parser.add_argument(
        "--st1_max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
    )
    parser.add_argument(
        "--st1_max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this value if set."
    )
    parser.add_argument(
        "--st1_model_name_or_path",
        type=str,
        default='pretrained_models/st1/roberta_st1/best_model',
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--st1_config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--st1_tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name"
    )
    parser.add_argument(
        "--st1_cache_dir",
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co"
    )
    parser.add_argument(
        "--st1_use_fast_tokenizer",
        action='store_true',
        default=True,
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
    )
    parser.add_argument(
        "--st1_model_revision",
        type=str,
        default="main",
        help="The specific model version to use (can be a branch name, tag name or commit id)."
    )
    parser.add_argument(
        "--st1_use_auth_token",
        action='store_true',
        help="Will use the token generated when running `transformers-cli login` (necessary to use this script with private models)."
    )
    parser.add_argument(
        "--st1_train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--st1_validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--st1_test_file",
        type=str,
        default=None,
        help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--st1_is_regression",
        action='store_true',
        default=False,
        help="If the model to use with predictions is a regression model."
    )
    
    parser.add_argument(
        "--st1_seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training."
    )
    
    
    
    
    
    
    
    
    
    #st1
    
    
    
    
    #st2
    
    
    parser.add_argument(
        "--st2_pretrained_path",
        type=str,
        default='pretrained_models/st2/roberta_st2/epoch_9',
        help="The path to the folder that has the pretrained model.",
    )
    parser.add_argument(
        "--st2_load_checkpoint_for_test",
        type=str,
        default='pretrained_models/st2/roberta_st2/epoch_9/pytorch_model.bin',
        help="specific path to the model",
    )
    parser.add_argument(
        "--st2_model_name_or_path",
        type=str,
        default='albert-base-v2',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--st2_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--st2_dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--st2_train_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--st2_validation_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--st2_test_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--st2_max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of "
            "training examples to this value if set."
        ),
    )
    parser.add_argument(
        "--st2_max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker validation, truncate the number of "
            "validation examples to this value if set."
        ),
    )
    parser.add_argument(
        "--st2_max_test_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker test, truncate the number of "
            "test examples to this value if set."
        ),
    )
    parser.add_argument(
        "--st2_max_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--st2_pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--st2_pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    
    parser.add_argument(
        "--st2_config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--st2_tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--st2_per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--st2_per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--st2_per_device_test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the test dataloader.",
    )
    parser.add_argument(
        "--st2_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--st2_weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--st2_num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--st2_max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--st2_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--st2_lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--st2_num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--st2_output_dir", 
        type=str, 
        default="outs/baseline", 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--st2_seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--st2_model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--st2_label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--st2_return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--st2_task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--st2_debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--st2_push_to_hub", 
        action="store_true", 
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--st2_hub_model_id", 
        type=str, 
        help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--st2_hub_token", 
        type=str, 
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--st2_checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--st2_resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--st2_with_tracking",
        action="store_true",
        default=False,
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--st2_report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--st2_with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--st2_ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    # Custom Arguments
    parser.add_argument(
        "--st2_add_signal_bias",
        action="store_true",
        help="Whether or not to add signal bias",
    )
    parser.add_argument(
        "--st2_signal_bias_on_top_of_lm",
        action="store_true",
        help="Whether or not to add signal bias",
    )
    parser.add_argument(
        "--st2_postprocessing_position_selector",
        action="store_true",
        help="Whether or not to use postprocessing position selector to control overlap problem.",
    )
    parser.add_argument(
        "--st2_mlp",
        action="store_true",
        help="Whether or not to add MLP layer on top of the pretrained LM.",
    )
    parser.add_argument(
        "--st2_signal_classification",
        action="store_true",
        help="Conduct signal classification to verify whether we need to detect signal span.",
    )
    parser.add_argument(
        "--st2_pretrained_signal_detector",
        action="store_true",
        help="Whether to use pretrained signal detector",
    )
    parser.add_argument( #"outs_test/signal_cls"
        "--st2_signal_model_and_tokenizer_path",
        type=str,
        help="Path to pretrained signal detector model.",
    )
    parser.add_argument(
        "--st2_beam_search",
        action="store_true",
        help="Whether to do bean search selection.",
    )
    parser.add_argument(
        "--st2_dropout",
        type=float,
        default=0.3,
        help="classifier dropout rate",
    )
    parser.add_argument(
        "--st2_use_best_model",
        action="store_true",
        help="Activate to use model with Highest Overall F1 score, else defaults to Last model.",
    )
   
    parser.add_argument(
        "--st2_do_train",
        action="store_true",
        help="Whether to train models from scratch.",
    )
    parser.add_argument(
        "--st2_do_test",
        action="store_true",
        help="Whether to use model to predict on test set.",
    )
    parser.add_argument(
        "--st2_augmentation_file",
        type=str,
        default=None,
        help="Whether to use pretrained signal detector",
    )
    parser.add_argument(
        "--st2_topk",
        type=int,
        default=5,
        help="Whether to use pretrained signal detector",
    )
    
    
    #st2
    
    
    
    
    #filter
    
    
    
    
    parser.add_argument('--filter_train_file', type=str, help='Path to the training data file')
    parser.add_argument('--filter_val_file', type=str, help='Path to the validation data file')
    parser.add_argument('--filter_test_file', type=str, help='Path to the test data file')
    parser.add_argument('--filter_threshold', type=float, default=0.8, help='Threshold for classification')
    parser.add_argument('--filter_model_path', type=str, default='pretrained_models/st0/roberta_st0/best_model_st1.pt', help='Path to model')
    
    
    
    
    
    #filter
    
    
    
    #llms
    
    
    
    parser.add_argument('--llms_task', help='Task to perform', choices=['test'], default='test')
    parser.add_argument('--llms_news_dataset', help='Path to the news dataset CSV file', default='news.csv')
    parser.add_argument('--llms_test_dataset', help='Path to the test dataset CSV file', default='test.csv')
    parser.add_argument('--llms_num_examples', type=int, help='Number of examples per relation', default=2)
    parser.add_argument('--llms_llm', help='LLM to use', default='zephyr', choices=available_llms)
    parser.add_argument('--llms_template', help='Path to the prompt template YAML file', default='prompt_template.yml')
    parser.add_argument('--llms_output', default='LLM_pred', help='Path to save the output predictions CSV file')
    parser.add_argument('--llms_api_key', help='API key for GPT-4', required=False)
    parser.add_argument('--llms_verbose', help='Print the full prompt', default=False, action='store_true')
    parser.add_argument("--llms_log", type=int, choices=[10, 20, 30, 40, 50], action="store", default=20,
                        help="Verbosity (default: INFO) : DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50")
    
    
    #llms
    
    
    #flags
    parser.add_argument(
        "--st2_roberta_flag",
        default='False',
        help="Tells the pipeline not to use this model",
    )
    parser.add_argument(
        "--st1_roberta_flag",
        default='False',
        help="Tells the pipeline not to use this model",
    )
    parser.add_argument(
        "--rebel_flag",
        action="store_true",
        help="Tells the pipeline to use this model",
    )
    parser.add_argument(
        "--llm_flag",
        action="store_true",
        help="Tells the pipeline to use this model",
    )
    
    parser.add_argument(
        "--subtask1_flag",
        default='False',
        help="Tells the pipeline not to do subtask 1",
    )
    parser.add_argument(
        "--subtask2_flag",
        default='False',
        help="Tells the pipeline not to do subtask 2",
    )
    parser.add_argument(
        "--subtask3_flag",
        default='False',
        help="Tells the pipeline not to do subtask 3",
    )
    
    
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to a configuration file"
    )
    
    
    
    parser.add_argument(
        "--split_st3_flag",
        action="store_true",
        help="Tells the pipeline to split st3",
    )
    parser.add_argument(
        "--rebel_st1_flag",
        action="store_true",
        help="Tells the pipeline to use st1 from rebel",
    )
    parser.add_argument(
        "--rebel_st2_flag",
        action="store_true",
        help="Tells the pipeline to use st2 from rebel",
    )
    
    
    parser.add_argument(
        "--llm_st1_flag",
        default='False',
        help="Tells the pipeline to use st1 from llm",
    )
    
    parser.add_argument(
        "--llm_st2_flag",
        default='False',
        help="Tells the pipeline to use st2 from llm",
    )
    
    
    parser.add_argument(
        "--llm_st1_mod",
        default='None',
        help="Tells the pipeline to use st2 from llm",
    )
    parser.add_argument(
        "--llm_st2_mod",
        default='None',
        help="Tells the pipeline to use st2 from llm",
    )
    
    parser.add_argument(
        "--rebel_st1_mod",
        default='None',
        help="Tells the pipeline to use st2 from llm",
    )
    parser.add_argument(
        "--rebel_st2_mod",
        default='None',
        help="Tells the pipeline to use st2 from llm",
    )
    
    
    
    #flags
    
    parser.add_argument(
        "--pipeline_config_name",
        type=str,
        default='None',
        help="Name of the output file that will be used as a reference"
    )
    parser.add_argument(
        "--st0_preset",
        type=str,
        default='None',
        help="Name of the file that is using a model that is already done for the filter"
    )
    parser.add_argument(
        "--st1_preset",
        type=str,
        default='None',
        help="Name of the file that is using a model that is already done for st1"
    )
    parser.add_argument(
        "--st2_preset",
        type=str,
        default='None',
        help="Name of the file that is using a model that is already done for st2"
    )
    
    parser.add_argument(
        "--preset_cache_dir",
        type=str,
        default='saved_app_outs/',
        help="Name of the output file that will be used as a reference"
    )
    
    #text from user
    parser.add_argument('--text_from_user', type=str, help='this is user submitted text to be evaluated')
    #text from user
    
    #
    parser.add_argument(
        "--user_config_file_path",
        type=str,
        default='False',
        help="Name of the output file that will be used as a reference"
    )
    
    '''
    parser.add_argument(
        "--st1_mod",
        type=str,
        default='None',
        help="model used for st1"
    )
    parser.add_argument(
        "--st2_mod",
        type=str,
        default='None',
        help="model used for st2",
    )
    '''
    #parser.add_argument('--llms_api_key', help='API key for GPT-4', required=False)
    parser.add_argument(
        "--config_path",
        default='None',
        help="Tells the pipeline to use st2 from llm",
    )
    #parser.add_argument('--text_from_user', default='None', required=False)
    
    parser.add_argument('--skip_st1', default='False', required=False)
    parser.add_argument('--skip_st2', default='False', required=False)
    
    parser.add_argument('--override_preset', default='False', required=False)
    
    
    #
    args = parser.parse_args()
    if args.config_file:
        config = configparser.ConfigParser()
        config.read(args.config_file)
        
        # Override command line arguments with those from the config file
        for key in config['TEMP']:
            value = config['TEMP'].get(key)
            if hasattr(args, key):
                #attr_type = type(getattr(args, key))
                attr_type = type(value)
                #print(attr_type)
                #print(value)
                #print(type(value))
                setattr(args, key, attr_type(config['TEMP'][key]))

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None  and args.test_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def split_list_last(lst):
    s = lst.split('  ')
    return s[-1]

def split_list_rest(lst):
    s = lst.split('  ')
    r = s[:-1]
    rest = "  ".join(s[:-1])
    if r:
        #print(r)
        r[0] = r[0].replace(" ", "", 1)
    
    return r

def para_into_df(s):
    parts = s.split('. ')
    parts = [part + '. ' for part in parts if part != '']
    df = pd.DataFrame()
    df['text'] = parts
    return df

def load_config(config_path, args):
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    
def run_pipeline(config):
    args = parse_args()
    st0_mod = 'n/a'
    st1_mod = 'n/a'
    st2_mod = 'n/a'

    if isinstance(config, str) and config != 'False':
        config = load_config(config, args)
    # Override command line arguments with those from the config file
    if config != 'False':
        for key in config['TEMP']:
                value = config['TEMP'].get(key)
                if hasattr(args, key):
                    attr_type = type(value)
                    setattr(args, key, attr_type(config['TEMP'][key]))

    args.filter_threshold = float(args.filter_threshold)
    args.st1_do_predict = True
    args.st2_do_test = True
    args.st1_use_cpu = True
    
    user_flag = False
    
    if args.text_from_user != None:
        print('*** Processed text: ***')
        print(args.text_from_user)
        base_df = para_into_df(args.text_from_user)
        user_flag = True
    else:
        base_df = pd.read_csv(args.test_file)
        st0_path = args.filter_model_path.split('/')
        st0_preset_name = args.preset_cache_dir + 'tf-' + args.test_file[9:] + '-filter-roberta-' + st0_path[-1]
    
    args.st1_test_file = args.test_file
    args.base_df = base_df
    
    if not user_flag and os.path.exists(args.st0_preset) and args.override_preset == 'False':
        only_causal_df = pd.read_csv(args.st0_preset)
        args.only_causal = only_causal_df
        df_final = only_causal_df.copy()
        st0_mod = 'roberta'
    else:
        only_causal_df = run_filter(args)    
        if len(only_causal_df) < 1:
            return []
        
        only_causal_df = only_causal_df.drop(columns=['label'])
        only_causal_df = only_causal_df.drop(columns=['triplets'])
        only_causal_df = only_causal_df.drop(columns=['causal_text_w_pairs'])
        args.only_causal = only_causal_df
        
        df_final = only_causal_df.copy()
        st0_path = args.filter_model_path.split('/')
        st0_mod = 'roberta'
        if user_flag == False:
            df_final.to_csv(args.preset_cache_dir + 'tf-' + args.test_file[9:] + '-filter-roberta-' + st0_path[-1] + '.csv')
            print(st0_preset_name)
            print('above is something to check')
    
    if args.subtask1_flag == 'True':
        print('*** SUBTASK 1 (st1) ***')
        if user_flag == False and os.path.exists(args.st1_preset) and args.override_preset == 'False':
            st1_df = pd.read_csv(args.st1_preset)
            df_final['label'] = st1_df['label']
            st1_mod = 'roberta'
        else:
            if not args.st1_roberta_flag:
                print('running default model')
                args.st1_roberta_flag = 'True'
                
            if args.st1_roberta_flag == 'True':
                args.st1_test_file = 't'
                st1_df = main_st1(args)
                mapping = {0: 'cause', 1: 'enable', 2: 'prevent', 3: 'intend'}
                df_final['label'] = st1_df
                df_final['label'] = df_final['label'].replace(mapping)
                st1_mod = 'roberta'
                if user_flag == False:
                    st1_path = args.st1_model_name_or_path.split('/')
                    st1_preset_name = st0_preset_name + '-st1-roberta-' + st1_path[-1]
                    print(st1_preset_name)
                    print('above is something to check')
                    df_final['label'].to_csv(st1_preset_name + '.csv')

    if args.subtask2_flag == 'True':
        print('*** SUBTASK 2 (st2) ***')
        if user_flag == False and os.path.exists(args.st2_preset) and args.override_preset == 'False':
            st2_df = pd.read_csv(args.st2_preset)
            df_final['num_rs'] = st2_df['num_rs']
            df_final['span_pred'] = st2_df['span_pred']
            st2_mod = 'roberta'
        else:
            if not args.st2_roberta_flag:
                print('running default model')
                args.st2_roberta_flag = 'True'
            if args.st2_roberta_flag == 'True':
                args.st2_test_file = 't'
                st2_result = main_st2(args)
                if len(st2_result)>0:
                    # for some reason, roberta export twice the same sentence.
                    # taking only the first one
                    st2_result = [s[0] for s in st2_result]
                df_final['num_rs'] = [1 for s in st2_result]
                df_final['span_pred'] = st2_result
                st2_mod = 'roberta'
                if user_flag == False:
                    st2_path = args.st2_pretrained_path.split('/')
                    st2_preset_name = st0_preset_name + '-st2-roberta-' + st2_path[-1]
                    print(st2_preset_name)
                    print('above is something to check')
                    df_final[['span_pred', 'num_rs']].to_csv(st2_preset_name + '.csv')


    if args.subtask3_flag == 'True':
        print('*** SUBTASK 3 (st3) ***')

        if args.rebel_flag == 'False' and args.llm_flag == 'False':
            print('running default model')
            args.rebel_flag = 'True'
            
        if args.rebel_flag == 'True':
            print('rebel')
            
            same_model = False
            st_switch = ''
            if args.rebel_st1_mod == 'None':
                same_model = True
                st_switch = 'st2'
                #rebel_df = test_model(args.only_causal, args.rebel_st2_mod)
            elif args.rebel_st1_mod == 'None':
                same_model = True
                st_switch = 'st1'
                #rebel_df = test_model(args.only_causal, args.rebel_st1_mod)
            elif args.rebel_st1_mod == args.rebel_st2_mod:
                same_model = True
                #rebel_df = test_model(args.only_causal, args.rebel_st1_mod)
                
            if user_flag:
                if same_model:
                    if st_switch == 'st1':
                        rebel_df = test_model(args.only_causal, args.rebel_st1_mod)
                    if st_switch == 'st2':
                        rebel_df = test_model(args.only_causal, args.rebel_st2_mod)
                    else:
                        rebel_df = test_model(args.only_causal, args.rebel_st2_mod)
                        
                
                if args.rebel_st1_flag == 'True':
                    st1_mod = 'rebel'
                    if not same_model:
                        rebel_df = test_model(args.only_causal, args.rebel_st1_mod)
                    df_final['label'] = rebel_df['prediction'].map(split_list_last)
                if args.rebel_st2_flag == 'True':
                    st2_mod = 'rebel'
                    if not same_model:
                        rebel_df = test_model(args.only_causal, args.rebel_st2_mod)
                    df_final['span_pred'] = rebel_df['prediction'].map(split_list_rest)
            else:
                if 'rebel' in args.st1_preset and os.path.exists(args.st1_preset) and args.override_preset == 'False':
                    st1_mod = 'rebel'
                    
                    rebel_df = pd.read_csv(args.st1_preset)
                    print(rebel_df['label'].head())
                    print(rebel_df.columns.tolist())
                    print('change above')
                    print(len(rebel_df))
                    print(len(df_final))
                    #print(df_final['label_rebel'].head())
                    x = rebel_df['label']
                    df_final['label'] = x
                elif 'rebel' in args.st1_preset:
                    st1_mod = 'rebel'
                    
                    rebel_df = test_model(args.only_causal, args.rebel_st1_mod)
                    df_final['label'] = rebel_df['prediction'].map(split_list_last)
                    
                    
                    rebel_path = args.rebel_st1_mod.split('/')
                    rebel_st1_preset_name = {}
                    rebel_st2_preset_name = {}
                    rebel_st1_preset_name['label'] = st0_preset_name + '-st1-rebel-' + rebel_path[-1]
                    rebel_st2_preset_name['span_pred'] = st0_preset_name + '-st2-rebel-' + rebel_path[-1]
                    
                    
                    df_rebel_preset = df_final
                    df_rebel_preset['label'] = rebel_df['prediction'].map(split_list_last)
                    df_rebel_preset['span_pred'] = rebel_df['prediction'].map(split_list_rest)
                    df_rebel_preset['label'].to_csv(rebel_st1_preset_name['label'] + '.csv')
                    df_rebel_preset['span_pred'].to_csv(rebel_st2_preset_name['span_pred'] + '.csv')
                    
                if 'rebel' in args.st2_preset and os.path.exists(args.st2_preset) and args.override_preset == 'False':
                    st2_mod = 'rebel'
                    
                    print(args.st2_preset)
                    rebel_df = pd.read_csv(args.st2_preset)
                    #x = rebel_df['span_pred_rebel']
                    df_final['span_pred'] = rebel_df['span_pred']
                elif 'rebel' in args.st2_preset:
                    st2_mod = 'rebel'
                    
                    
                    rebel_df = test_model(args.only_causal, args.rebel_st2_mod)
                    df_final['span_pred'] = rebel_df['prediction'].map(split_list_rest)
                    
                    
                    rebel_path = args.rebel_st2_mod.split('/')
                    rebel_st1_preset_name = {}
                    rebel_st2_preset_name = {}
                    rebel_st1_preset_name['label'] = st0_preset_name + '-st1-rebel-' + rebel_path[-1]
                    rebel_st2_preset_name['span_pred'] = st0_preset_name + '-st2-rebel-' + rebel_path[-1]
                    
                    
                    df_rebel_preset = df_final
                    df_rebel_preset['label'] = rebel_df['prediction'].map(split_list_last)
                    df_rebel_preset['span_pred'] = rebel_df['prediction'].map(split_list_rest)
                    df_rebel_preset['label'].to_csv(rebel_st1_preset_name['label'] + '.csv')
                    df_rebel_preset['span_pred'].to_csv(rebel_st2_preset_name['span_pred'] + '.csv')
                    
        if args.llm_flag == 'True':
            print('LLM')    
            args.llms_output = args.llms_output + '/' + args.llms_llm +'/' + args.llms_llm + f'_pred-{datetime.now()}.csv'
            same_model = False
            if args.llm_st1_flag == 'False':
                same_model = True
                args.llms_llm = args.llm_st2_mod
                llm_df = run_LLM(args)
            elif args.llm_st2_flag == 'False':
                same_model = True
                args.llms_llm = args.llm_st1_mod
                llm_df = run_LLM(args)
            elif args.llm_st2_mod == args.llm_st1_mod:
                same_model = True
                args.llms_llm = args.llm_st2_mod
                llm_df = run_LLM(args)
                
            if args.llm_st1_flag == 'True':
                if not same_model:
                    args.llms_llm = args.llm_st1_mod
                    llm_df = run_LLM(args)
                st1_mod = 'llm_' + args.llms_llm
                df_final['label'] = llm_df.apply(lambda row: [ row['relation']], axis=1)
                
            if args.llm_st2_flag == 'True':
                
                if not same_model:
                    args.llms_llm = args.llm_st2_mod
                    llm_df = run_LLM(args)
                st2_mod = 'llm_' + args.llms_llm
                df_final['span_pred'] = llm_df.apply(lambda row: [ row['subject'], row['object']], axis=1)
            

            llm_df['subj-obj-rel-LLM-' + args.llms_llm] = llm_df.apply(lambda row: [row['subject'], row['object'], row['relation']], axis=1)
            
    df_final['st0_model'] = st0_mod
    df_final['st1_model'] = st1_mod
    df_final['st2_model'] = st2_mod
   
    if args.pipeline_config_name != 'None':
        df_final.to_csv(args.preset_cache_dir + args.pipeline_config_name + '.csv')
    df_json = df_final.to_dict(orient='records')
    return df_json

if __name__ == "__main__":
    
    args = parse_args()
    config_path = args.user_config_file_path
    
    json_dict = run_pipeline(config_path)
    
    df = pd.DataFrame(json_dict)
    os.makedirs('out', exist_ok = True)
    df.to_csv('out/'f'final-combined_pred-{datetime.now()}.csv')