import numpy as np
import pandas as pd
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import PreTrainedTokenizerFast


dataset = ["acgt"]

batch_size = 1000

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]

special_tokens = ["[MASK]", "[PAD]"]

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.Lowercase()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # should not be used at all
trainer = trainers.BpeTrainer(
    vocab_size=5,
    special_tokens=special_tokens,
    initial_alphabet=["a", "c", "g", "t"],
)


tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, mask_token="[MASK]", pad_token="[PAD]")
print(len(tokenizer))
tokenizer.save_pretrained(f"./tokenizer_bare/")