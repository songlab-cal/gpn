import numpy as np
import pandas as pd
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import PreTrainedTokenizerFast


# make sure lowercase is well handled
# [UNK] should be avoided since it might convert N into a whole token, breaking the kmer structure

# adapted from https://classes.cs.uchicago.edu/archive/2020/fall/30121-1/lecture-examples/Kmers/Kmers.html
def generate_dna_kmers(k):
    '''
    Return a list of all possible substrings of
    length k using only characters A, C, G, and T
    '''
    bases = ["a", "c", "g", "t"]

    last = bases
    current = []
    for i in range(k-1):
        for b in bases:
            for l in last:
                current.append(l+b)
        last = current
        current= []
    return last


K = 4

vocab = ["[PAD]", "[UNK]"] + generate_dna_kmers(K)
vocab_size = len(vocab)
assert(vocab_size==2+4**K)

vocab = dict(zip(vocab, range(vocab_size)))
tokenizer = Tokenizer(models.WordLevel(vocab, "[UNK]"))
tokenizer.normalizer = normalizers.Lowercase()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token="[PAD]",
    unk_token="[UNK]",
    padding_side="left",
    truncation_side="left",
)

#print(tokenizer("AAAA CAAA CGAT"))
#seq = "AAAACAAACGAT"
#seq_kmers = " ".join([seq[x:x+K] for x in range(0, len(seq), K)])
#print(seq_kmers)

tokenizer.save_pretrained(f"tokenizer_ss_clm_K{K}")