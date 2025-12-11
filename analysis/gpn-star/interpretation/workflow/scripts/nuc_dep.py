from gpn.data import GenomeMSA, Tokenizer
import gpn.star.model
from gpn.star.utils import find_directory_sum_paths

from datasets import Dataset
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForMaskedLM, TrainingArguments, Trainer
from tqdm import tqdm
import tempfile

import sys


CHROM = sys.argv[1]
START = int(sys.argv[2])
END = int(sys.argv[3])
STRAND = sys.argv[4]
MODEL_PATH = sys.argv[5]
MSA_PATH = sys.argv[6]
PHYLO_INFO_PATH = sys.argv[7]
WINDOW_SIZE = int(sys.argv[8])
OUTPUT_PATH = sys.argv[9]


assert END > START
assert END - START <= WINDOW_SIZE


def f(msa):
    B = msa.shape[0]
    input_ids, source_ids = msa[:, :, :1], msa
    target_species = np.full((B, 1), target_species_index, dtype=int)
    logits = model(
        input_ids=input_ids, source_ids=source_ids, target_species=target_species
    ).logits[:, :, target_species_index, NUCLEOTIDES_IDX]
    logits = F.log_softmax(logits, dim=-1)
    return logits.cpu().float()


def get_categorical_jacobian(msa, target_only=True):
    all_tokens = torch.tensor(NUC_TOKENS)
    num_tokens = len(NUC_TOKENS)
    L = len(msa)
    S = msa.shape[1]

    with torch.no_grad():
        x = torch.clone(msa).unsqueeze(0)
        fx = f(x)[0]
        fx_h = torch.zeros((L, num_tokens, L, num_tokens), dtype=torch.float32)
        x = torch.tile(x, [num_tokens, 1, 1])
        for i in tqdm(range(L)):
            x_h = torch.clone(x)
            if target_only:
                x_h[:, i, 0] = all_tokens
            else:
                x_h[:, i] = all_tokens.unsqueeze(1).expand(-1, S)
            fx_h[i] = f(x_h)
        jac = fx_h - fx
    return jac.numpy()


def nucleotide_dependencies(jac, ord=None, symmetric=False):
    L = jac.shape[0]
    x = jac.transpose(0, 2, 1, 3).reshape(L, L, -1)
    x = np.linalg.norm(x, ord=ord, axis=2)
    np.fill_diagonal(x, 0)
    if symmetric:
        x = (x + x.T) / 2
    return x


center = (START + END) // 2
context_start = center - WINDOW_SIZE // 2
context_end = center + WINDOW_SIZE // 2
assert context_end - context_start == WINDOW_SIZE

start_idx = START - context_start
end_idx = END - context_start

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
target_species_index = 0

msa_paths = find_directory_sum_paths(MSA_PATH)
genome_msa_list = [
    GenomeMSA(path, n_species=n_species, in_memory=False)
    for n_species, path in msa_paths.items()
]
config = AutoConfig.from_pretrained(MODEL_PATH)
config.phylo_dist_path = PHYLO_INFO_PATH
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, config=config)
model.to(device).eval()

tokenizer = Tokenizer()
NUCLEOTIDES = list("ACGT")
NUCLEOTIDES_IDX = [tokenizer.vocab.index(nc) for nc in NUCLEOTIDES]
NUC_TOKENS = NUCLEOTIDES_IDX
NUCLEOTIDES_IDX


msa = [
    genome_msa.get_msa(CHROM, context_start, context_end, strand="+", tokenize=True)
    for genome_msa in genome_msa_list
]
msa = np.concatenate(msa, axis=-1)
msa = torch.tensor(msa.astype(np.int64)).to(device)


full_jac = get_categorical_jacobian(msa)
jac = full_jac[start_idx:end_idx, :, start_idx:end_idx, :]
contact = nucleotide_dependencies(
    jac,
    ord=np.inf,
    symmetric=True,
)
coordinates = range(START, END)
contact = pd.DataFrame(contact, index=coordinates, columns=coordinates)

if STRAND == "-":
    contact = contact.iloc[::-1, ::-1]

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
pd.DataFrame(contact).to_parquet(OUTPUT_PATH)
