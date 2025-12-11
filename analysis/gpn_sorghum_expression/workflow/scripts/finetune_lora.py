#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import numpy as np
import os
import sys
from dataclasses import dataclass, field
import torch
from typing import Dict, List, Optional

import datasets
from datasets import load_dataset

# import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from peft import get_peft_model, LoraConfig

import gpn.model
from modeling_species_expression import GPNForSpeciesExpression


def standardize(x):
    return (x - x.mean(axis=0, dtype=np.float64)) / x.std(axis=0, dtype=np.float64)


def batched_pearsonr(x, y):
    return np.mean(standardize(x) * standardize(y), axis=0, dtype=np.float64)


def parse_hidden_dims(value: Optional[str]) -> List[int]:
    if value is None:
        return []
    value = value.strip()
    if value == "" or value.lower() == "none":
        return []
    dims: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        dims.append(int(token))
    return dims


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.26.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    do_test: bool = field(default=False)
    min_lr_rate: Optional[float] = field(default=None)
    problem_type: Optional[str] = field(
        default=None,
    )
    classification_head: Optional[str] = field(
        default="standard",
    )
    regression_softplus: bool = field(default=False)
    seq_column_name: Optional[str] = field(
        default=None,
    )
    species_column_name: Optional[str] = field(
        default="species_id",
        metadata={
            "help": "Column containing the species identifier. Defaults to 'species_id'."
        },
    )
    species_projection_hidden_dims: Optional[str] = field(
        default="1024",
        metadata={
            "help": (
                "Comma-separated hidden dimensions for the species-specific projection head. "
                "Use an empty string to disable additional hidden layers."
            )
        },
    )
    species_projection_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate within the species-specific projection head."},
    )
    species_projection_activation: str = field(
        default="gelu",
        metadata={"help": "Activation function used inside the species-specific head."},
    )
    species_projection_pooling: str = field(
        default="mean",
        metadata={
            "help": "Pooling strategy over sequence representations before the projection head."
        },
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "labels" column for single/multi-label classification task'
            )
        },
    )
    pos_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Positive weight for binary token classification"},
    )
    token_classification: bool = field(
        default=False,
        metadata={
            "help": "Whether to run token classification or sequence classification."
        },
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use streaming datasets."},
    )
    subsample_train: Optional[float] = field(
        default=None,
        metadata={"help": "Subsample the training dataset to this proportion."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.min_lr_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr_rate"] = data_args.min_lr_rate

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Downloading and loading a dataset from the hub.
    # Also works for local dataset
    d = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
        streaming=data_args.streaming,
    )
    print(d)

    num_labels = len(d["train"][0]["labels"])
    print(f"{num_labels=}")

    if data_args.seq_column_name is not None and data_args.seq_column_name != "seq":
        for key in d.keys():
            d[key] = d[key].rename_column(data_args.seq_column_name, "seq")

    if (
        data_args.label_column_name is not None
        and data_args.label_column_name != "labels"
    ):
        for key in d.keys():
            d[key] = d[key].rename_column(data_args.label_column_name, "labels")

    if data_args.streaming:
        raise ValueError(
            "Species-specific projection currently does not support streaming datasets."
        )

    species_column = data_args.species_column_name or "species_id"
    if species_column not in d["train"].column_names:
        raise ValueError(
            f"Expected column '{species_column}' in the dataset to identify species, "
            f"but available columns are {d['train'].column_names}."
        )

    all_species = set()
    for split_name, split_dataset in d.items():
        if species_column not in split_dataset.column_names:
            raise ValueError(
                f"Column '{species_column}' missing from split '{split_name}'. "
                "Ensure the dataset includes the species identifier for every split."
            )
        all_species.update(split_dataset.unique(species_column))

    species_list = sorted(all_species)
    species_to_idx = {species: idx for idx, species in enumerate(species_list)}
    logger.info(
        "Identified %d species for swappable projection layers: %s",
        len(species_list),
        species_list,
    )

    def add_species_idx(batch):
        return {
            "species_idx": [species_to_idx[s] for s in batch[species_column]],
        }

    d = d.map(
        add_species_idx,
        batched=True,
        desc="Indexing species identifiers",
        remove_columns=[species_column],
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "num_labels": num_labels,
        "problem_type": data_args.problem_type,
        "pos_weight": data_args.pos_weight,
        "classification_head": data_args.classification_head,
        "regression_softplus": data_args.regression_softplus,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type](**config_kwargs)
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    species_hidden_dims = parse_hidden_dims(data_args.species_projection_hidden_dims)
    config.species_to_idx = species_to_idx
    config.species_projection_hidden_dims = species_hidden_dims
    config.species_projection_dropout = float(data_args.species_projection_dropout)
    config.species_projection_activation = data_args.species_projection_activation
    config.species_projection_pooling = data_args.species_projection_pooling

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = GPNForSpeciesExpression.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = GPNForSpeciesExpression(config)

    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        # target_modules="all-linear",
        target_modules=r"^model\.encoder\.\d+\.conv\.1$|^model\.encoder\.\d+\.ffn\.0$",
        modules_to_save=["species_projection"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if training_args.do_train:
        d["train"] = d["train"].shuffle(seed=training_args.seed)

    def tokenize_function(examples):
        res = {}
        res["input_ids"] = tokenizer(
            examples["seq"],
            return_special_tokens_mask=False,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )["input_ids"]
        res["labels"] = examples["labels"]
        res["species_idx"] = examples["species_idx"]
        return res

    d.set_transform(tokenize_function)

    train_dataset = d["train"] if training_args.do_train else None
    eval_dataset = d["validation"] if training_args.do_eval else None
    test_dataset = d["test"] if data_args.do_test else None

    def compute_metrics(eval_pred):
        y_pred = eval_pred.predictions
        y_true = eval_pred.label_ids
        if num_labels == 1:
            y_true = np.expand_dims(y_true, axis=1)
        mean_pearson_across_rows = np.mean(batched_pearsonr(y_pred, y_true))
        res = {"mean_pearson_across_rows": mean_pearson_across_rows}
        if num_labels > 1:
            mean_pearson_across_cols = np.nanmean(batched_pearsonr(y_pred.T, y_true.T))
            res["mean_pearson_across_cols"] = mean_pearson_across_cols
        return res

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    # https://github.com/huggingface/peft/issues/1881
    trainer.can_return_loss = True

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Testing
    if data_args.do_test:
        logger.info("*** Test ***")

        test_output = trainer.predict(test_dataset=test_dataset)
        metrics = test_output.metrics
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = (
                f"{data_args.dataset_name} {data_args.dataset_config_name}"
            )
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
