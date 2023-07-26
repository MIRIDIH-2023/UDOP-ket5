#!/usr/bin/env python
# coding = utf-8

import logging
import os
import random
import sys
import pickle
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import torch
import transformers
from transformers import (AutoConfig, AutoModelForTokenClassification,
                          AutoTokenizer, HfArgumentParser, Trainer,
                          TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers import T5Tokenizer

from core.common.utils import random_split, visualize_text_layout_task, visualize_text_task, visualize_layout_task, img_trans_torchvision, get_visual_bbox
from core.datasets import RvlCdipDatasetForVisualization
from core.models import (UdopConfig, UdopTokenizer,
                         UdopUnimodelForConditionalGeneration)
from core.trainers import DataCollator
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CLASSES = {
    'UdopUnimodel': (UdopConfig, UdopUnimodelForConditionalGeneration, UdopTokenizer),
}

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "local dataset stored location"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
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
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    image_size: Optional[int] = field(
    default=512,
    metadata={
        "help": "image size"
        "value if set."
    },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            'help':
            'The maximum total input sequence length after tokenization. Sequences longer '
            'than this will be truncated, sequences shorter will be padded.'
        },
    )   
    max_seq_length_decoder: int = field(
        default=16,
        metadata={
            'help':
            'The maximum total input sequence length after tokenization. Sequences longer '
            'than this will be truncated, sequences shorter will be padded.'
        },
    )    
    do_save_visualize: bool = field(
        default=False,
        metadata={
            'help':'Whether to save visualizations in predict'
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        default=None, metadata={'help': 'Model type selected in the list.'})
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
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )    
    attention_type: str = field(
        default="original_full",
        metadata={"help": "Attention type: BigBird configuruation only. Choices: block_sparse (default) or original_full"},
    )


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.logging_dir = os.path.join(training_args.output_dir, 'runs')
    if model_args.cache_dir is None:
        model_args.cache_dir = os.path.join(training_args.output_dir, 'cache')
    if training_args.do_train:
        os.makedirs(model_args.cache_dir, exist_ok=True)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Model arguments: {model_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

 
    #if 'local' in model_args.model_name_or_path:
    if model_args.model_type in MODEL_CLASSES:
        config_type, model_type, tokenizer_type = MODEL_CLASSES[model_args.model_type]
    else:
        config_type, model_type, tokenizer_type = AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = UdopConfig.from_pretrained("finetuning/checkpoint-180000")
    tokenizer = AutoTokenizer.from_pretrained("finetuning/checkpoint-180000")
    model = UdopUnimodelForConditionalGeneration.from_pretrained("finetuning/checkpoint-180000").to(device)


   # Get datasets
    json_path = '../data/json_data'
    image_path = '../data/image'
    indexmap = None
    with open('./no_outside_lst_new.pickle','rb') as f:
        indexmap = pickle.load(f)
        f.close()
    random.shuffle(indexmap)

    test_dataset = (RvlCdipDatasetForVisualization(json_path = json_path,
                                    image_path = image_path,
                                    index_map = indexmap,
                                    data_args=data_args, 
                                    tokenizer=tokenizer,  
                                    mode='test')
                     if (training_args.do_eval or training_args.do_predict) else None) 


    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        while True:
            
            idx = input(f"Enter idx (or 'quit') in range 0 ~ {len(test_dataset)-1}: ")
            if idx.lower() == "quit":
                break
            
            sample, page_size, image_size = test_dataset.__getitem__(int(idx))

            input_ids = torch.unsqueeze(sample['input_ids'], dim=0).to(device)
            labels = sample['labels'].to(device)
            seg_data = torch.unsqueeze(sample['seg_data'], dim=0).to(device)
            im = torch.unsqueeze(sample['image'], dim=0).to(device)
            visual_seg_data = torch.unsqueeze(sample['visual_seg_data'], dim=0).to(device)
            


            output_ids = model.generate(
                    input_ids,
                    seg_data=seg_data,
                    image=im,
                    visual_seg_data=visual_seg_data,
                    use_cache=True,
                    decoder_start_token_id=tokenizer.pad_token_id,
                    num_beams=1,
                    max_length=512,
                )
            
            input_text = tokenizer.decode(input_ids[0])
            prediction_text = tokenizer.decode(output_ids[0][1:-1])
            label_text = tokenizer.decode(labels)

            if input_text.startswith("Layout Modeling"):
                visualize_layout_task(sample, label_text, prediction_text, input_text, data_args.do_save_visualize, training_args.output_dir, page_size, image_size)
            elif input_text.startswith("Visual Text Recognition"):
                visualize_text_task(sample, label_text, prediction_text, input_text, data_args.do_save_visualize, training_args.output_dir, page_size, image_size)
            elif input_text.startswith("Joint Text-Layout Reconstruction"):
                visualize_text_layout_task(sample, label_text, prediction_text, data_args.do_save_visualize, training_args.output_dir, page_size, image_size)

            print("input: ", input_text)
            print("\nLabel: ", label_text)
            print("\nPrediction: ", prediction_text)


if __name__ == "__main__":
    main()
