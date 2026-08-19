# Copyright 2020 The HuggingFace Team. All rights reserved.
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

"""Train GPN on a prepared nucleotide-sequence dataset."""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from Bio.Seq import Seq
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    set_seed,
)

from gpn import register_auto_classes
from gpn.ss.model import GPNConfig, GPNForMaskedLM
from gpn.training import (
    configure_training_logging,
    evaluate_and_save,
    find_training_checkpoint,
    load_training_dataset,
    parse_training_arguments,
    train_and_save,
)

logger = logging.getLogger(__name__)


class DataCollatorForLanguageModelingSimplified(DataCollatorForLanguageModeling):
    """Stack equal-length examples directly instead of padding them."""

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = {
            key: torch.stack([torch.tensor(example[key]) for example in examples])
            for key in examples[0]
        }
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch


@dataclass
class ModelArguments:
    """Model and tokenizer inputs owned by the GPN trainer."""

    model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Optional GPN checkpoint to continue training."},
    )
    tokenizer_name: str | None = field(
        default=None,
        metadata={"help": "Tokenizer name or path; defaults to the model checkpoint."},
    )
    config_overrides: str | None = field(
        default=None,
        metadata={
            "help": "Comma-separated GPNConfig overrides for training from scratch."
        },
    )
    cache_dir: str | None = field(
        default=None, metadata={"help": "Optional Hugging Face cache directory."}
    )
    model_revision: str = field(
        default="main", metadata={"help": "Model/tokenizer revision."}
    )

    def __post_init__(self) -> None:
        if self.config_overrides and self.model_name_or_path:
            raise ValueError(
                "config_overrides cannot be combined with model_name_or_path."
            )
        if not (self.tokenizer_name or self.model_name_or_path):
            raise ValueError("Set tokenizer_name or model_name_or_path.")


@dataclass
class DataTrainingArguments:
    """Prepared dataset and GPN-specific masking controls."""

    dataset_name: str = field(metadata={"help": "Prepared dataset name or path."})
    dataset_config_name: str | None = field(default=None)
    dataset_revision: str | None = field(
        default=None,
        metadata={"help": "Immutable Hugging Face dataset revision."},
    )
    mlm_probability: float = field(default=0.15)
    soft_masked_loss_weight_train: float = field(default=1.0)
    soft_masked_loss_weight_evaluation: float = field(default=1.0)
    map_batch_size: int = field(
        default=2048, metadata={"help": "Examples per tokenization map batch."}
    )


def main(argv: list[str] | None = None) -> None:
    register_auto_classes("ss")
    model_args, data_args, training_args = parse_training_arguments(
        ModelArguments, DataTrainingArguments, argv
    )
    configure_training_logging(training_args, logger)
    checkpoint = find_training_checkpoint(training_args)
    set_seed(training_args.seed)
    np.random.seed(training_args.seed)

    raw_datasets = load_training_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        dataset_revision=data_args.dataset_revision,
        cache_dir=model_args.cache_dir,
        streaming=True,
    )

    model_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
    }
    if model_args.model_name_or_path:
        config = GPNConfig.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
        model = GPNForMaskedLM.from_pretrained(
            model_args.model_name_or_path, config=config, **model_kwargs
        )
    else:
        config = GPNConfig()
        if model_args.config_overrides:
            config.update_from_string(model_args.config_overrides)
        model = GPNForMaskedLM(config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        **model_kwargs,
    )

    def tokenize(
        examples: dict[str, list[str]],
        soft_masked_weight: float,
        *,
        augment_strand: bool = False,
    ) -> dict[str, Any]:
        sequences = examples["seq"]
        if augment_strand:
            reverse = np.random.choice([False, True], len(sequences))
            sequences = [
                str(Seq(sequence).reverse_complement()) if use_reverse else sequence
                for sequence, use_reverse in zip(sequences, reverse, strict=True)
            ]
        result = tokenizer(
            sequences,
            return_special_tokens_mask=True,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        result["loss_weight"] = np.ones_like(result["input_ids"], dtype=float)
        lowercase = np.char.islower([list(sequence) for sequence in sequences])
        result["loss_weight"][lowercase] = soft_masked_weight
        return result

    remove_columns = list(next(iter(raw_datasets["train"])).keys())
    train_dataset = None
    if training_args.do_train:
        train_dataset = (
            raw_datasets["train"]
            .shuffle(seed=training_args.seed)
            .map(
                lambda examples: tokenize(
                    examples,
                    data_args.soft_masked_loss_weight_train,
                    augment_strand=True,
                ),
                batched=True,
                batch_size=data_args.map_batch_size,
                remove_columns=remove_columns,
                drop_last_batch=True,
            )
        )

    eval_dataset = None
    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"].map(
            lambda examples: tokenize(
                examples, data_args.soft_masked_loss_weight_evaluation
            ),
            batched=True,
            remove_columns=remove_columns,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModelingSimplified(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
        ),
    )
    train_and_save(trainer, training_args, checkpoint)
    evaluate_and_save(trainer, training_args)


if __name__ == "__main__":
    main()
