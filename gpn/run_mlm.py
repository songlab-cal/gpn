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
import math
import numpy as np
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import datasets
from datasets import load_dataset, DatasetDict, concatenate_datasets

import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from Bio.Seq import Seq
import gpn.model
import numpy as np
import pandas as pd
from scipy.stats import geom
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class GenomeSamplerDataset(IterableDataset):
    def __init__(
        self,
        dataset=None,
        tokenizer_path=None,
        window_size=None,
        random_seed=None,
        min_contig_size=None,
        soft_masked_weight=None,
    ):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.window_size = window_size
        self.random_seed = random_seed
        self.soft_masked_weight = soft_masked_weight

        print("Loading parquet.")
        self.contigs = dataset
        self.contigs["contig_len"] = self.contigs.seq.str.len()
        print(self.contigs.shape)
        if min_contig_size is not None:
            self.contigs = self.contigs[self.contigs.contig_len >= self.min_contig_size]
            print(self.contigs.shape)
        if not "contig_weight" in self.contigs.columns:
            print("Setting contig weights according to lengths.")
            self.contigs["contig_weight"] = (1 + self.contigs.contig_len - self.window_size).clip(lower=1)
        else:
            print("Using predefined contig weights.")
        self.contigs["contig_prob"] = self.contigs.contig_weight / self.contigs.contig_weight.sum()
        print("Done.")

    def __iter__(self):
        print("Loading tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        print("Done.")

        seed = self.random_seed
        worker_info = get_worker_info()
        if worker_info is not None:
            seed = seed * (worker_info.id + 1)
        rs = np.random.RandomState(seed=seed)

        while True:
            contig_index = rs.choice(len(self.contigs), p=self.contigs.contig_prob.values)
            contig = self.contigs.iloc[contig_index]
            if contig.contig_len > self.window_size:
                start = rs.randint(contig.contig_len - self.window_size)
            else:
                start = 0
            end = start + self.window_size
            seq = contig.seq[start:end]
            strand = rs.choice(["+", "-"])
            if strand == "-":
                seq = str(Seq(seq).reverse_complement())

            x = tokenizer(
                seq,
                return_token_type_ids=False,
                return_attention_mask=False,
                return_tensors="pt",
            )
            x["input_ids"] = x["input_ids"].flatten()
            x["loss_weight"] = np.ones_like(x["input_ids"], dtype=float)
            x["loss_weight"][np.char.islower(list(seq))] = self.soft_masked_weight
            yield x


class DataCollatorForLanguageModelingSimplified(DataCollatorForLanguageModeling):
    # Simplified to skip padding since we'll assume all sequences have the same length
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = {
            key: torch.stack([torch.tensor(example[key]) for example in examples], dim=0)
            for key in examples[0].keys()
        }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


#rv = geom(0.1)
#probs = np.array([rv.pmf(i) for i in range(1, 6)])
#probs = probs / sum(probs)
#probs = torch.tensor(probs).float()
#values = torch.range(1, 5).float()
#span_mean = torch.dot(probs, values)
#print("span_mean: ", span_mean)
#
#N_NON_SPECIAL_TOKENS = 4
#
#
#
#class DataCollatorForLanguageModelingSpan(DataCollatorForLanguageModeling):
#    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
#        """
#        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
#        """
#        import torch
#
#        labels = inputs.clone()
#        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
#        probability_matrix = torch.full(labels.shape, self.mlm_probability / span_mean)  # approximate, doesn't count collisions not borders
#        if special_tokens_mask is None:
#            special_tokens_mask = [
#                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
#            ]
#            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
#        else:
#            special_tokens_mask = special_tokens_mask.bool()
#
#        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
#        masked_indices = torch.bernoulli(probability_matrix).bool()
#
#        mask_idx = torch.nonzero(masked_indices)
#        span = 1 + torch.multinomial(probs, len(mask_idx), replacement=True)
#        for (i, j), s in zip(mask_idx, span):
#            masked_indices[i, j:min(j+s, masked_indices.shape[1])] = True
#
#        labels[~masked_indices] = -100  # We only compute loss on masked tokens
#
#        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
#
#        # 10% of the time, we replace masked input tokens with random word
#        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#        # modification introduced by gbenegas:
#        # only replace with non-special tokens
#        random_words = torch.randint(len(self.tokenizer)-N_NON_SPECIAL_TOKENS, len(self.tokenizer), labels.shape, dtype=torch.long)
#        inputs[indices_random] = random_words[indices_random]
#
#        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
#        return inputs, labels
#
#    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
#        batch = {key: torch.stack([torch.tensor(example[key]) for example in examples], dim=0) for key in examples[0].keys()}
#
#        # If special token mask has been preprocessed, pop it from the dict.
#        special_tokens_mask = batch.pop("special_tokens_mask", None)
#        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
#            batch["input_ids"], special_tokens_mask=special_tokens_mask,
#        )
#        return batch


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.26.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

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
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
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
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
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
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
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
    soft_masked_loss_weight_train: Optional[float] = field(default=1.0)
    soft_masked_loss_weight_evaluation: Optional[float] = field(default=1.0)

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mlm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Downloading and loading a dataset from the hub.
    # Also works for local dataset
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
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
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names

    def tokenize_function(examples, soft_masked_weight):
        res = tokenizer(
            examples["seq"],
            return_special_tokens_mask=True,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        res["loss_weight"] = np.ones_like(res["input_ids"], dtype=float)
        res["loss_weight"][
            np.char.islower([list(x) for x in examples["seq"]])
        ] = soft_masked_weight
        return res

    soft_masked_weight = {
        "train": data_args.soft_masked_loss_weight_train,
        "validation": data_args.soft_masked_loss_weight_evaluation,
    }

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = DatasetDict()
        for split, w in soft_masked_weight.items():
            if split == "train": continue  # will be tokenized on-the-fly by GenomeSamplerDataset
            tokenized_datasets[split] = raw_datasets[split].map(
                lambda examples: tokenize_function(examples, w),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Running tokenizer in {split} dataset with soft masked loss weight {w}",
            )
    
    if training_args.do_train:
        train_dataset = GenomeSamplerDataset(
            dataset=pd.DataFrame(raw_datasets["train"]),
            tokenizer_path=model_args.tokenizer_name,
            window_size=512,
            random_seed=training_args.seed,
            soft_masked_weight=data_args.soft_masked_loss_weight_train,
        )
        #if data_args.max_train_samples is not None:
        #    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        #    train_dataset = train_dataset.select(range(max_train_samples))


    if training_args.do_eval:
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Data collator
    # This one will take care of randomly masking the tokens.

    data_collator = DataCollatorForLanguageModelingSimplified(
    #data_collator = DataCollatorForLanguageModelingSpan(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

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

        #max_train_samples = (
        #    data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        #)
        #metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
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
