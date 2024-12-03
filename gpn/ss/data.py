from Bio.Seq import Seq
from datasets import load_dataset, IterableDatasetDict
import numpy as np
import torch
from transformers import AutoTokenizer, set_seed
from typing import Any, Dict, Optional, Tuple


# TODO: should be modular since this should be used for both GPN and GPN-MSA
# this should be moved to gpn/data.py
# also, it could be swapped to a span masker, for example
def create_masked_input_and_labels(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,  # TODO: in the future, we can use our own tokenizer class
    mlm_probability: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    assert input_ids.dtype == torch.uint8
    labels = input_ids.clone().to(torch.int8)  # need to handle -100
    # We sample a few tokens in each sequence for MLM training (with probability `mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id()

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer.vocab_size(), labels.shape, dtype=torch.uint8)
    input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return input_ids, labels


def load_dataset_training(
    path: str,
    tokenizer_path: str = "songlab/tokenizer-dna-mlm",
    soft_masked_loss_weight_train: float = 0.1,
    soft_masked_loss_weight_evaluation: float = 0.0,
    seed: int = 42,
    batch_size: int = 2048,  # unfortunately, needed for now, see comment below
    mask_probability: float = 0.15,
) -> IterableDatasetDict:
    set_seed(seed)
    raw_datasets = load_dataset(path, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def tokenize_function(
        examples, soft_masked_weight, data_augmentation=False,
        do_create_masked_input_and_labels=True,
    ):
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
        input_ids, labels = create_masked_input_and_labels(
            input_ids, tokenizer, mask_probability
        )
        return dict(input_ids=input_ids, labels=labels, loss_weight=loss_weight)

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