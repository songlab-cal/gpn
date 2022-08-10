from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer


class GenomeSamplerDataset(IterableDataset):
    def __init__(
        self,
        fasta_path=None,
        tokenizer_path=None,
        window_size=None,
        max_length=None,
        random_seed=None,
        min_contig_size=None,
    ):
        super().__init__()
        self.fasta_path = fasta_path
        self.tokenizer_path = tokenizer_path
        self.window_size = window_size
        self.max_length = max_length
        self.random_seed = random_seed
        self.min_contig_size = min_contig_size

        print("Loading parquet.")
        self.contigs = pd.read_parquet(self.fasta_path)
        self.contigs["contig_len"] = self.contigs.seq.str.len()
        print(self.contigs.shape)
        self.contigs = self.contigs[self.contigs.contig_len >= self.min_contig_size]
        print(self.contigs.shape)
        if not "contig_weight" in self.contigs.columns:
            print("Setting contig weights according to lengths.")
            self.contigs["contig_weight"] = (1 + self.contigs.contig_len - self.window_size).clip(lower=1)
        else:
            print("Using predefined contig weights.")
        self.contigs["contig_prob"] = self.contigs.contig_weight / self.contigs.contig_weight.sum()
        print(self.contigs[["contig_len", "contig_weight", "contig_prob"]])
        print("Done.")

    def __iter__(self):
        print("Loading tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        print("Done.")

        seed = self.random_seed
        worker_info = get_worker_info()
        if worker_info is not None:
            seed = seed * (worker_info.id + 1)
        rs = np.random.RandomState(seed=seed)

        while True:
            contig_index = rs.choice(len(self.contigs), p=self.contigs.contig_prob.values)
            contig = self.contigs.iloc[contig_index]
            if contig.contig_len > self.window_size:
                start = rs.randint(contig.contig_len - self.window_size)
            else:
                start = 0
            end = start + self.window_size
            seq = contig.seq[start:end]
            strand = rs.choice(["+", "-"])
            if strand == "-":
                seq = str(Seq(seq).reverse_complement())

            x = tokenizer(
                seq,
                return_token_type_ids=False,
                return_attention_mask=False,
                return_tensors="pt",
            )
            x["input_ids"] = x["input_ids"].flatten()
            yield x
