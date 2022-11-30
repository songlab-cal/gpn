import numpy as np
from scipy.stats import geom
import torch
from transformers import DataCollatorForLanguageModeling
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from collections.abc import Mapping


rv = geom(0.1)
probs = np.array([rv.pmf(i) for i in range(1, 6)])
probs = probs / sum(probs)
probs = torch.tensor(probs).float()
values = torch.range(1, 5).float()
span_mean = torch.dot(probs, values)
print("span_mean: ", span_mean)

N_NON_SPECIAL_TOKENS = 4


class DataCollatorForLanguageModelingSpan(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability / span_mean)  # approximate, doesn't count collisions not borders
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
            #print("mean: ", special_tokens_mask.float().mean())
            #raise Exception("collator debug")

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        mask_idx = torch.nonzero(masked_indices)
        span = 1 + torch.multinomial(probs, len(mask_idx), replacement=True)
        for (i, j), s in zip(mask_idx, span):
            masked_indices[i, j:min(j+s, masked_indices.shape[1])] = True

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # modification introduced by gbenegas:
        # only replace with non-special tokens
        random_words = torch.randint(len(self.tokenizer)-N_NON_SPECIAL_TOKENS, len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = {key: torch.stack([torch.tensor(example[key]) for example in examples], dim=0) for key in examples[0].keys()}
        # Handle dict or lists with proper padding and conversion to tensor.
        #if isinstance(examples[0], Mapping):
        #    batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        #else:
        #    batch = {
        #        "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        #    }

        # If special token mask has been preprocessed, pop it from the dict.
        #no_loss_mask = batch.pop("no_loss_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"],
            #no_loss_mask=no_loss_mask,
        )
        return batch
