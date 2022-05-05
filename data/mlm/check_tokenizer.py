import pandas as pd
import sys
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], use_fast=True)
#tokenizer._tokenizer.model.dropout = 0.5
print(len(tokenizer))
with open(sys.argv[2]) as f:
    seqs = f.read().splitlines()
    seqs = seqs[:10000]
    print(len(seqs))
    print(len(seqs[0]))
tokenized_length = pd.Series([len(s) for s in tokenizer(seqs)["input_ids"]])
print(tokenized_length.describe())
