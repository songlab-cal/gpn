import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer

np.set_printoptions(threshold=100000)

#seqs_path = "./windows/val/1000/1000/seqs.txt"
seqs_path = "./seqs_tokenizer_training_1k_50k.txt"
with open(seqs_path) as f:
    seqs = f.read().splitlines()
    seqs = seqs[:100]
    print(len(seqs))
    print(len(seqs[0]))
    window_size = len(seqs[0])

tokenizer_paths = [
    "tokenizer_bpe_256_v5",
    "tokenizer_bpe_256_v5_regularized",
    "tokenizer_bpe_1024_v4",
    "tokenizer_bpe_1024_v4_regularized",
    "tokenizer_bpe_4096_v4",
    "tokenizer_bpe_4096_v4_regularized",
    "tokenizer_bpe_8192_v5",
    "tokenizer_bpe_8192_v5_regularized",
    "tokenizer_spc_4096",
    "tokenizer_spc_4096_regularized",
    "tokenizer_spc_8192",
    "tokenizer_spc_8192_regularized",
]

rows = []
for tokenizer_path in tokenizer_paths:
    print(tokenizer_path)
    regularized = False
    sp_model_kwargs = None
    use_fast = True
    
    if "spc" in tokenizer_path:
        use_fast = False
        sp_model_kwargs = dict(enable_sampling=False)


    if tokenizer_path.endswith("_regularized"):
        regularized = True
        if "spc" in tokenizer_path:
            sp_model_kwargs = dict(enable_sampling=True, nbest_size=-1, alpha=1.0)
            
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.replace("_regularized", ""), use_fast=use_fast, sp_model_kwargs=sp_model_kwargs)
    if regularized and "spc" not in tokenizer_path:
        tokenizer._tokenizer.model.dropout = 0.5
    input_ids = tokenizer(seqs)["input_ids"]
    all_input_ids = np.concatenate(input_ids)
    tokenized_length = pd.Series([len(s) for s in input_ids])
    #tokenized_length.describe()
    compression = window_size / tokenized_length.mean()
    nuc_token_ids = [tokenizer.get_vocab()[nuc] for nuc in ["a", "c", "g", "t"]]
    proportion_singleton_tokens = np.isin(all_input_ids, nuc_token_ids).mean()
    token_counts = str(pd.Series(all_input_ids).value_counts().iloc[2:].values)  # ignoring CLS and SEP
    rows.append([tokenizer_path, compression, proportion_singleton_tokens, token_counts])
df = pd.DataFrame(rows, columns=["tokenizer", "compression", "proportion_singleton_tokens", "token_counts"])
df.index = df.tokenizer.values
df.to_csv("tokenizer_statistics.tsv", sep="\t") 