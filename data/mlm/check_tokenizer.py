import numpy as np
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


def insert_spaces(seq, prob):
    new_seq = []
    for x in seq:
        if np.random.random() < prob:
            new_seq += [" ", x, " "]
        else:
            new_seq.append(x)
    new_seq = "".join(new_seq)
    return new_seq


if False:
    prob = 0.015
    seqs = [insert_spaces(seq, prob) for seq in seqs]


input_ids = tokenizer(seqs)["input_ids"]
all_input_ids = np.concatenate(input_ids)
nuc_token_ids = [tokenizer.get_vocab()[nuc] for nuc in ["a", "c", "g", "t"]]
proportion_singleton_tokens = np.isin(all_input_ids, nuc_token_ids).mean()
tokenized_length = pd.Series([len(s) for s in input_ids])
print(tokenized_length.describe())
print("proportion_singleton tokens: ", proportion_singleton_tokens)
