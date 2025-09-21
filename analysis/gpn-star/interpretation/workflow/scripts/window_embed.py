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
from transformers import AutoConfig, AutoModel, TrainingArguments, Trainer
from tqdm import tqdm
import tempfile

import sys


WINDOWS_PATH = sys.argv[1]
MODEL_PATH = sys.argv[2]
MSA_PATH = sys.argv[3]
PHYLO_INFO_PATH = sys.argv[4]
WINDOW_SIZE = int(sys.argv[5])
CENTER_WINDOW_SIZE = int(sys.argv[6]) # we average embeddings over this central sub-window
OUTPUT_PATH = sys.argv[7]


msa_paths = find_directory_sum_paths(MSA_PATH)
genome_msa_list = [GenomeMSA(path, n_species=n_species, in_memory=False) for n_species, path in msa_paths.items()]
config = AutoConfig.from_pretrained(MODEL_PATH)
config.phylo_dist_path = PHYLO_INFO_PATH
_model = AutoModel.from_pretrained(MODEL_PATH, config=config)



def get_msa(chrom, start, end, strand):
    # getting each strand separately is not optimal, just temporary
    msa = [genome_msa.get_msa(chrom, start, end, strand=strand, tokenize=True) for genome_msa in genome_msa_list]
    msa = np.concatenate(msa, axis=-1)
    msa = torch.tensor(msa.astype(np.int64))
    return msa


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def embed(
        self,
        msa=None,
    ):
        B = len(msa)
        input_ids = msa[:, :, :1]
        source_ids = msa
        target_species = np.full((B, 1), 0, dtype=int)

        embed = self.model(
            input_ids=input_ids, source_ids=source_ids, target_species=target_species
        ).last_hidden_state[:, :, 0, :]
        center = WINDOW_SIZE // 2
        left = center - CENTER_WINDOW_SIZE // 2
        right = center + CENTER_WINDOW_SIZE // 2
        embed = embed[:, left:right]
        embed = embed.mean(dim=1)
        return embed

    def forward(self, msa_fwd=None, msa_rev=None):
        embed_fwd = self.embed(msa_fwd)
        embed_rev = self.embed(msa_rev)
        return (embed_fwd + embed_rev) / 2

model = ModelWrapper(_model)


df = pd.read_parquet(WINDOWS_PATH)
if "center" not in df.columns:
    df["center"] = (df["start"] + df["end"]) // 2
d = Dataset.from_pandas(df)

def transform(vs):
    chrom = np.array(vs["chrom"])
    n = len(chrom)
    center = np.array(vs["center"])
    start = center - WINDOW_SIZE // 2
    end = center + WINDOW_SIZE // 2
    msa_fwd = torch.stack([get_msa(chrom[i], start[i], end[i], "+") for i in range(n)])
    msa_rev = torch.stack([get_msa(chrom[i], start[i], end[i], "-") for i in range(n)])
    return dict(msa_fwd=msa_fwd, msa_rev=msa_rev)

d.set_transform(transform)

training_args = TrainingArguments(
    output_dir=tempfile.TemporaryDirectory().name,
    per_device_eval_batch_size=128,
    dataloader_num_workers=8,
    remove_unused_columns=False,
    torch_compile=True,  # can be faster to skip for small inputs
    bf16=True,
    bf16_full_eval=True,
    report_to="none",
)
trainer = Trainer(model=model, args=training_args)
embed = trainer.predict(test_dataset=d).predictions

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
pd.DataFrame(embed).to_parquet(OUTPUT_PATH, index=False)