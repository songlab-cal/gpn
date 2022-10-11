from Bio.AlignIO.MafIO import MafIndex
from Bio.Seq import Seq
from collections.abc import Mapping
from dataclasses import dataclass
from einops import rearrange, reduce, repeat
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


@dataclass
class DataCollatorForLanguageModelingMSA(DataCollatorForLanguageModeling):
    mlm_prob_target_only : float = 0.1
    mlm_prob_col : float = 0.1
    no_mask_flank : int = 8

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        B, R, C = inputs.shape
        labels = inputs.clone()
        # gbenegas: important note - ignoring special tokens mask for now

        probability_matrix_target_only = torch.full(labels.shape, self.mlm_prob_target_only)
        probability_matrix_target_only[:, 1:, :] = 0.0
        probability_matrix_target_only[:, :, :self.no_mask_flank] = 0.0
        probability_matrix_target_only[:, :, -self.no_mask_flank:] = 0.0
        masked_indices_target_only = torch.bernoulli(probability_matrix_target_only).bool()

        probability_matrix_col = torch.full((B, C), self.mlm_prob_col)
        probability_matrix_col[:, :self.no_mask_flank] = 0.0
        probability_matrix_col[:, -self.no_mask_flank:] = 0.0
        masked_indices_col = repeat(torch.bernoulli(probability_matrix_col).bool(), "B C -> B R C", R=R)

        masked_indices = masked_indices_target_only | masked_indices_col
        #print(masked_indices_target_only.float().mean(), masked_indices_col.float().mean(), masked_indices.float().mean())

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        #print(labels)
        #raise Exception("debug")

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

# d = GenomeMSASamplerDataset(
#    intervals_path="intervals/Homo_sapiens.train.tsv.gz",
#    data_path=".",
#    tokenizer_path="tokenizer",
#    window_size=32,
#    random_seed=42,
# )
# i = 0
# for x in d:
#    print(x)
#    i += 1
#    if i > 100: break

# d = GenomeMSAFixedDataset(
#    intervals_path="intervals/genome.test.tsv.gz",
#    data_path=".",
#    tokenizer_path="tokenizer",
#    window_size=64,
#    step_size=32,
# )
# print(d[0])

# i = 0
# dl = DataLoader(d, batch_size=4, num_workers=2)
# for batch in dl:
#    if i % 100 == 0: print(i)
#    #print(batch)
#    i += 1
#    if i > 10000: break
