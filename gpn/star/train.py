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
import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import datasets
from datasets import load_dataset, DatasetDict, concatenate_datasets, disable_caching

# import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    # AutoTokenizer,
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

import gpn.star.model
from gpn.star.utils import *
from gpn.star.data import GenomeMSA, Tokenizer
import numpy as np

disable_caching()

class DataCollatorForLanguageModelingSimplified(DataCollatorForLanguageModeling):
    # gbenegas: Simplified to skip padding since we'll assume all sequences have the same length
    def __init__(self, tokenizer, clades, mlm=True, mlm_probability=0.15):
        super().__init__(tokenizer, mlm, mlm_probability)
        self.clades = clades

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = {
            key: torch.stack(
                [torch.tensor(example[key]) for example in examples], dim=0
            )
            for key in examples[0].keys()
        }

        # For calculating col-wise MSA dropout rates
        L = batch["input_ids"].size(-1)

        flip_p = batch.pop("flip_p", None)
        if self.mlm:
            batch["input_ids"], batch["labels"], batch["source_ids"] = self.torch_mask_tokens(
                batch["input_ids"],
                batch["source_ids"],
                batch["target_species"],
                flip_p=flip_p,
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        
        return batch

    def torch_mask_tokens(
        self,
        inputs: torch.Tensor, # (B, L, T)
        source_ids: torch.Tensor, # (B, L, N)
        target_species: torch.Tensor, # (B, T)
        flip_p=None,
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """        
        # these come from the tokenizer, for now hardcoding
        nuc_min, nuc_max = (
            self.tokenizer.nucleotide_token_id_start(),
            self.tokenizer.nucleotide_token_id_end(),
        )
        clades = self.clades
        C = clades.unique().size(0)
        B, L, T = inputs.shape
        labels = inputs.clone()
        
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        # Mask the same positions in each clade
        probability_matrix = torch.full((B, L, C), self.mlm_probability)

        # Bernoulli sampling of masked positions
        masked_indices_clades = torch.bernoulli(probability_matrix).bool() # (B, L, C)
        
        # Get masks for the target species
        target_clades = clades[target_species] # (B, T)
        masked_indices = torch.gather(masked_indices_clades, 
                                      dim=2, 
                                      index=target_clades[:, None, :].expand(-1, L, -1))
        
        # Do not mask gap tokens
        gap_mask = labels == 0
        masked_indices = masked_indices & ~gap_mask

        # 90% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.9)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id()

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        # For non-conserved columns, random flip the token into other nucleotides
        if flip_p is not None:
            indices_random_labels = (torch.bernoulli(flip_p).bool()[:, :, None] & masked_indices)
            random_nucs_labels = torch.randint(
                nuc_min, nuc_max, labels.shape, dtype=labels.dtype
            )
            labels[indices_random_labels] = random_nucs_labels[indices_random_labels]

        # We also mask the tokens in the corresponding clades in source_ids to avoid in-clade copying
        target_clades = clades[target_species] # (B, T) # each species in T is from a unique clade
        masked_indices_all_species = get_all_species_mask(masked_indices, target_clades, clades)
        source_ids[masked_indices_all_species] = self.tokenizer.mask_token_id()
        
        
        # We only compute loss on masked (thus also non-gap) tokens
        labels[~masked_indices] = -100

        return inputs, labels, source_ids


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
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    msa_path: Optional[str] = field(default=None)
    phylo_dist_path: Optional[str] = field(default=None)
    soft_masked_loss_weight_train: Optional[float] = field(default=1.0)
    soft_masked_loss_weight_evaluation: Optional[float] = field(default=1.0)
    use_aux_features: Optional[bool] = field(default=True)
    weight_conserved: Optional[str] = field(default='True')
    flip_nonconserved: Optional[float] = field(default=0)


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
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print(raw_datasets)

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
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.config_overrides is not None:
        logger.info(f"Overriding config: {model_args.config_overrides}")
        config.update_from_string(model_args.config_overrides)
        logger.info(f"New config: {config}")

    print('CONFIG: ', config)
    
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
        config.phylo_dist_path = data_args.phylo_dist_path
        model = AutoModelForMaskedLM.from_config(config)

    # Read MSA file
    msa_paths = find_directory_sum_paths(data_args.msa_path)
    print(msa_paths)
    genome_msa_list = [GenomeMSA(path, n_species=n_species, in_memory=False) for n_species, path in msa_paths.items()]

    # Get clade info from the phylo info module in the initialized model
    clade_dict = model.model.phylo_info.clade_dict
    clades = model.model.phylo_info.clade_labels
    print('Num clades', len(clade_dict))
    print(clade_dict)
    
    def tokenize_function(examples, soft_masked_weight, split):
        msa = [genome_msa.get_msa_batch(
            examples["chrom"],
            examples["start"],
            examples["end"],
            examples["strand"],
            tokenize=True,  # n_jobs=0,
        ) for genome_msa in genome_msa_list]
        msa = np.concatenate(msa, axis=-1)

        # Sample species in MSA
        B, L, N = msa.shape

        phastCons = np.array(examples["phastCons"], dtype=float)
        phastCons = max_smooth(np.nan_to_num(phastCons, nan=0.0), 7)
        phyloP = np.array(examples["phyloP"], dtype=float)
        
        # Sample target species
        # Sample 19 clades
        # Sample 1 species per selected clade
        # Always include human
        # In total 20 species
        num_targets = 20
        
        if len(list(clade_dict[0])) == 1: # if the first clade has only human
            all_clades = np.arange(1, len(clade_dict)) # do not sample human again
        else:
            all_clades = np.arange(len(clade_dict))
        

        target_clades = []
        with_replacement = len(all_clades) < num_targets-1
        for i in range(B):
            target_clades.append(np.random.choice(all_clades, size=num_targets-1, replace=with_replacement))
        target_clades = np.stack(target_clades) # (B, T-1)
        
        target_species = np.zeros((B, num_targets), dtype=np.int32)
        for i in range(B):
            for j in range(1, num_targets):
                clade = target_clades[i, j-1]
                species = clade_dict[clade] - {0} # do not sample human again
                target_species[i, j] = np.random.choice(list(species), 1)[0]
        
        # subsample
        input_ids = np.take_along_axis(msa, target_species[:, np.newaxis, :], axis=2)

        # position-wise loss weight based on conservation
        loss_weight = np.ones((B, L), dtype=float)
        loss_weight[np.array(examples["lowercase"])] *= soft_masked_weight

        if data_args.weight_conserved == 'True':
            loss_weight *= np.fmax(phyloP, 1.0)  # ignores nan
            loss_weight *= 0.1 + phastCons
        elif data_args.weight_conserved == 'neutral':
            loss_weight *= np.fmax(1.0+5*(1.0-np.abs(phyloP)), 1.0)  # ignores nan
            loss_weight *= 0.1 + (1-phastCons)

        # species-wise loss
        loss_weight_species = np.ones(num_targets, dtype=float)
        
        loss_weight = loss_weight[:, :, None] * loss_weight_species[None, None, :] # (B, L, C)
        
        res = dict(input_ids=input_ids,
                   loss_weight=loss_weight, 
                   target_species = target_species,
                   )
        
        res['source_ids'] = msa


        if data_args.flip_nonconserved > 0:
            flip_p = 0.5 * (phastCons < data_args.flip_nonconserved)
            res["flip_p"] = flip_p
                
        return res

    soft_masked_weight = {
        "train": data_args.soft_masked_loss_weight_train,
        "validation": data_args.soft_masked_loss_weight_evaluation,
    }

    # with training_args.main_process_first(desc="dataset map tokenization"):
    # tokenized_datasets = DatasetDict()
    for split, w in soft_masked_weight.items():
        raw_datasets[split].set_transform(
            # this trick is needed to pass the weight to the function
            # so w is a copy and not a reference which gets overriden
            lambda examples, w=w, split=split: tokenize_function(examples, w, split),
        )
    # Should evaluation be done without sampling MSAs?
    
    tokenized_datasets = raw_datasets

    if training_args.do_train:
        train_dataset = tokenized_datasets["train"]

    if training_args.do_eval:
        eval_dataset = tokenized_datasets["validation"]

    # Data collator
    # This one will take care of randomly masking the tokens.

    data_collator = DataCollatorForLanguageModelingSimplified(
        Tokenizer(),
        mlm_probability=data_args.mlm_probability,
        clades = clades
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
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
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

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
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
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
