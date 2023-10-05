import numpy as np
import pandas as pd
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import PreTrainedTokenizerFast


dataset = ["acgt"]
special_tokens = [
    "[PAD]",
    "[MASK]",
    "[UNK]",
]

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Lowercase()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # should not be used at all
trainer = trainers.BpeTrainer(
    vocab_size=7,
    special_tokens=special_tokens,
    initial_alphabet=list(dataset[0]),
)
tokenizer.train_from_iterator(dataset, trainer=trainer, length=len(dataset))
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token="[PAD]",
    mask_token="[MASK]",
    unk_token="[UNK]",
)
print(len(tokenizer))
tokenizer.save_pretrained("tokenizer_ss")