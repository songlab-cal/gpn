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

"""Train GPN-Star on prepared genomic intervals and a local whole-genome MSA."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor
from transformers import Trainer, set_seed

from gpn.data import Tokenizer
from gpn.star.data import GenomeMSA
from gpn.star.model import GPNStarConfig, GPNStarForMaskedLM
from gpn.star.utils import find_directory_sum_paths, get_all_species_mask, max_smooth
from gpn.training import (
    configure_training_logging,
    evaluate_and_save,
    find_training_checkpoint,
    load_training_dataset,
    parse_training_arguments,
    train_and_save,
)

logger = logging.getLogger(__name__)


class DataCollatorForLanguageModelingSimplified:
    """Apply clade-aware masking to equal-length GPN-Star examples."""

    def __init__(self, tokenizer, clades, mlm_probability=0.15):
        self.gpn_tokenizer = tokenizer
        self.clades = clades
        self.mlm_probability = mlm_probability

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        return self.torch_call(examples)

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Tensor]:
        field_dtypes = {
            "input_ids": torch.long,
            "source_ids": torch.long,
            "target_species": torch.long,
            "loss_weight": torch.float32,
        }
        batch = {
            key: torch.stack([torch.as_tensor(example[key]) for example in examples])
            for key in examples[0]
        }
        for key, dtype in field_dtypes.items():
            if key in batch:
                batch[key] = batch[key].to(dtype)

        batch["input_ids"], batch["labels"], batch["source_ids"] = (
            self.torch_mask_tokens(
                batch["input_ids"],
                batch["source_ids"],
                batch["target_species"],
            )
        )
        return batch

    def torch_mask_tokens(
        self,
        inputs: Int[Tensor, "batch position target"],
        source_ids: Int[Tensor, "batch position species"],
        target_species: Int[Tensor, "batch target"],
    ) -> tuple[
        Int[Tensor, "batch position target"],
        Int[Tensor, "batch position target"],
        Int[Tensor, "batch position species"],
    ]:
        """Mask the same positions within clades and prevent in-clade copying."""

        clades = torch.as_tensor(
            self.clades,
            dtype=torch.long,
            device=inputs.device,
        )
        batch_size, length, _ = inputs.shape
        labels = inputs.clone()

        probability_matrix = torch.full(
            (batch_size, length, clades.unique().size(0)),
            float(self.mlm_probability),
            dtype=torch.float32,
            device=inputs.device,
        )
        masked_clades = torch.bernoulli(probability_matrix).bool()
        target_clades = clades[target_species]
        masked = torch.gather(
            masked_clades,
            dim=2,
            index=target_clades[:, None, :].expand(-1, length, -1),
        )
        masked &= labels != 0

        replaced = (
            torch.bernoulli(
                torch.full(
                    labels.shape,
                    0.9,
                    dtype=torch.float32,
                    device=labels.device,
                )
            ).bool()
            & masked
        )
        inputs[replaced] = self.gpn_tokenizer.mask_token_id()

        masked_sources = get_all_species_mask(masked, target_clades, clades)
        source_ids[masked_sources] = self.gpn_tokenizer.mask_token_id()
        labels[~masked] = -100
        return inputs, labels, source_ids


@dataclass
class ModelArguments:
    """Model inputs owned by the GPN-Star trainer."""

    model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Optional GPN-Star checkpoint to continue training."},
    )
    config_overrides: str | None = field(
        default=None,
        metadata={
            "help": "Comma-separated GPNStarConfig overrides for training from scratch."
        },
    )
    cache_dir: str | None = field(
        default=None, metadata={"help": "Optional Hugging Face cache directory."}
    )
    model_revision: str = field(default="main", metadata={"help": "Model revision."})

    def __post_init__(self) -> None:
        if self.config_overrides and self.model_name_or_path:
            raise ValueError(
                "config_overrides cannot be combined with model_name_or_path."
            )


@dataclass
class DataTrainingArguments:
    """Prepared interval, alignment, and phylogenetic inputs for GPN-Star."""

    dataset_name: str = field(metadata={"help": "Prepared interval dataset."})
    msa_path: str = field(metadata={"help": "Local prepared MSA directory."})
    phylo_dist_path: str = field(
        metadata={"help": "Directory containing pairwise.npy and in_clade.npy."}
    )
    dataset_config_name: str | None = field(default=None)
    dataset_revision: str | None = field(
        default=None,
        metadata={"help": "Immutable Hugging Face dataset revision."},
    )
    mlm_probability: float = field(default=0.15)
    soft_masked_loss_weight_train: float = field(default=1.0)
    soft_masked_loss_weight_evaluation: float = field(default=1.0)


def main(profile: Path) -> None:
    model_args, data_args, training_args = parse_training_arguments(
        ModelArguments, DataTrainingArguments, profile
    )
    configure_training_logging(training_args, logger)
    checkpoint = find_training_checkpoint(training_args)
    set_seed(training_args.seed)

    raw_datasets = load_training_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        dataset_revision=data_args.dataset_revision,
        cache_dir=model_args.cache_dir,
    )

    model_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
    }
    if model_args.model_name_or_path:
        model = GPNStarForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            phylo_dist_path=data_args.phylo_dist_path,
            **model_kwargs,
        )
    else:
        config = GPNStarConfig(phylo_dist_path=data_args.phylo_dist_path)
        if model_args.config_overrides:
            config.update_from_string(model_args.config_overrides)
        model = GPNStarForMaskedLM(config)

    msa_paths = find_directory_sum_paths(data_args.msa_path)
    genome_msas = [
        GenomeMSA(path, n_species=n_species, in_memory=False)
        for n_species, path in msa_paths.items()
    ]
    clade_dict = model.model.phylo_info.clade_dict
    clades = model.model.phylo_info.clade_labels

    def tokenize(
        examples: dict[str, Any], soft_masked_weight: float
    ) -> dict[str, np.ndarray]:
        msa = np.concatenate(
            [
                genome_msa.get_msa_batch(
                    examples["chrom"],
                    examples["start"],
                    examples["end"],
                    examples["strand"],
                    tokenize=True,
                )
                for genome_msa in genome_msas
            ],
            axis=-1,
        )
        batch_size, length, _ = msa.shape

        phast_cons = max_smooth(
            np.nan_to_num(np.asarray(examples["phastCons"], dtype=float), nan=0.0),
            7,
        )
        phylo_p = np.asarray(examples["phyloP"], dtype=float)

        num_targets = 20
        first_clade_is_human_only = len(clade_dict[0]) == 1
        all_clades = np.arange(1 if first_clade_is_human_only else 0, len(clade_dict))
        sampled_clades = np.stack(
            [
                np.random.choice(
                    all_clades,
                    size=num_targets - 1,
                    replace=len(all_clades) < num_targets - 1,
                )
                for _ in range(batch_size)
            ]
        )

        target_species = np.zeros((batch_size, num_targets), dtype=np.int32)
        for batch_index in range(batch_size):
            for target_index, clade in enumerate(sampled_clades[batch_index], start=1):
                candidates = clade_dict[clade] - {0}
                target_species[batch_index, target_index] = np.random.choice(
                    list(candidates)
                )

        input_ids = np.take_along_axis(msa, target_species[:, np.newaxis, :], axis=2)
        loss_weight = np.ones((batch_size, length), dtype=float)
        loss_weight[np.asarray(examples["lowercase"])] *= soft_masked_weight
        loss_weight *= np.fmax(phylo_p, 1.0)
        loss_weight *= 0.1 + phast_cons

        return {
            "input_ids": input_ids,
            "loss_weight": np.repeat(loss_weight[:, :, None], num_targets, axis=2),
            "target_species": target_species,
            "source_ids": msa,
        }

    weights = {
        "train": data_args.soft_masked_loss_weight_train,
        "validation": data_args.soft_masked_loss_weight_evaluation,
    }
    for split, weight in weights.items():
        raw_datasets[split].set_transform(
            lambda examples, weight=weight: tokenize(examples, weight)
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=(raw_datasets["validation"] if training_args.do_eval else None),
        data_collator=DataCollatorForLanguageModelingSimplified(
            Tokenizer(), clades=clades, mlm_probability=data_args.mlm_probability
        ),
    )
    train_and_save(trainer, training_args, checkpoint)
    evaluate_and_save(trainer, training_args)
