from Bio.Seq import Seq
from datasets import load_dataset, IterableDatasetDict
import numpy as np
import torch
from transformers import AutoTokenizer
from typing import Any, Dict, Optional, Tuple


def load_dataset_training(
    path: str,
    tokenizer_path: str = "songlab/tokenizer-dna-mlm",
    soft_masked_loss_weight_train: float = 0.1,
    soft_masked_loss_weight_evaluation: float = 0.0,
    seed: int = 42,
    batch_size: int = 2048,  # unfortunately, needed for now, see comment below
) -> IterableDatasetDict:
    np.random.seed(seed)
    raw_datasets = load_dataset(path, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def tokenize_function(examples, soft_masked_weight, data_augmentation=False):
        seq = examples["seq"]
        if data_augmentation:
            n = len(seq)
            strand = np.random.choice(["+", "-"], n)
            seq = [
                seq[i] if strand[i] == "+" else str(Seq(seq[i]).reverse_complement())
                for i in range(n)
            ]

        input_ids = tokenizer(
            seq,
            padding=False,
            truncation=False,
            return_special_tokens_mask=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )["input_ids"]
        input_ids = torch.tensor(input_ids, dtype=torch.uint8)
        loss_weight = torch.ones_like(input_ids, dtype=torch.float16)
        loss_weight[np.char.islower([list(x) for x in seq])] = soft_masked_weight
        return dict(input_ids=input_ids, loss_weight=loss_weight)

    remove_columns = list(list(raw_datasets["train"].take(1))[0].keys())

    train_dataset = raw_datasets["train"].shuffle(seed=seed)
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(
            examples, soft_masked_loss_weight_train, data_augmentation=True,
        ),
        batched=True, remove_columns=remove_columns,
        # This takes care of some issues when using torch_compile
        # I think it's a bug in IterableDataset in the datasets library
        # When the last batch is smaller than the batch size
        # Hopefully it will be fixed soon
        drop_last_batch=True, batch_size=batch_size,
    )
    validation_dataset = raw_datasets["validation"].map(
        lambda examples: tokenize_function(
            examples, soft_masked_loss_weight_evaluation
        ),
        batched=True, remove_columns=remove_columns,
    )
    test_dataset = raw_datasets["test"].map(
        lambda examples: tokenize_function(
            examples, soft_masked_loss_weight_evaluation
        ),
        batched=True, remove_columns=remove_columns,
    )

    return IterableDatasetDict(
        train=train_dataset,
        validation=validation_dataset,
        test=test_dataset,
    )