from datasets import load_dataset
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from transformers import AutoTokenizer


#NUM_WORKERS = 0
NUM_WORKERS = 6
K = 6


def encode_dna_seq(seq):
    return np.array([encode_base(base) for base in seq], dtype=np.uint8)


def encode_base(base):
    if base == "N":
        base = np.random.choice(["A", "C", "G", "T"])
    encoding = {"A": 0, "C": 1, "G": 2, "T": 3}
    result = encoding[base]
    return result


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers


class DeepSEADataset(Dataset):
    def __init__(self, data_path):
        self.df = pd.read_parquet(data_path)
        self.features = [col for col in self.df.columns if col not in ["chromosome", "start", "end", "strand", "seq"]]

        #n = len(self.df)
        #idx = np.concatenate([np.arange(100), np.arange(100) + n//2])
        #self.df = self.df.iloc[idx]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row.seq
        x = encode_dna_seq(x).astype(int)
        y = row[self.features].values.astype(np.uint8)
        return x, y


class DeepSEADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        print("Loading train dataset")
        self.train_dataset = DeepSEADataset(os.path.join(self.data_dir, "train.parquet"))
        print("Loading val dataset")
        self.val_dataset = DeepSEADataset(os.path.join(self.data_dir, "val.parquet"))
        print("Loading test dataset")
        self.test_dataset = DeepSEADataset(os.path.join(self.data_dir, "test.parquet"))
        print(len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,#NUM_WORKERS,
            pin_memory=True,
        )


#class DNABERTDataset(Dataset):
#    def __init__(self, data_path, language_model_name):
#        self.df = pd.read_parquet(data_path)
#        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
#
#    def __len__(self):
#        return len(self.df)
#
#    def __getitem__(self, idx):
#        row = self.df.iloc[idx]
#        x = row.name
#        #x = x[256:256+512]
#        #x = x[400:600]
#        x = seq2kmer(x, K)
#        #x = self.tokenizer(x, padding="max_length", return_token_type_ids=False, return_tensors="pt")
#        x = self.tokenizer(x, padding="max_length", max_length=1000, return_token_type_ids=False, return_tensors="pt")
#        x["input_ids"] = x["input_ids"].flatten()
#        x["attention_mask"] = x["attention_mask"].flatten()
#        y = row.values
#        return x, y
#
#
#class DNABERTDataModule(pl.LightningDataModule):
#    def __init__(
#        self,
#        data_dir,
#        batch_size,
#        language_model_name,
#    ):
#        super().__init__()
#        self.data_dir = data_dir
#        self.batch_size = batch_size
#        self.language_model_name = language_model_name
#
#    def prepare_data(self):
#        print("Loading train dataset")
#        self.train_dataset = DeepSEADataset(os.path.join(self.data_dir, "train.parquet"), self.language_model_name)
#        print("Loading val dataset")
#        self.val_dataset = DeepSEADataset(os.path.join(self.data_dir, "val.parquet"), self.language_model_name)
#        print("Loading test dataset")
#        self.test_dataset = DeepSEADataset(os.path.join(self.data_dir, "test.parquet"), self.language_model_name)
#        print(len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))
#
#    def train_dataloader(self):
#        return DataLoader(
#            self.train_dataset,
#            batch_size=self.batch_size,
#            shuffle=True,
#            num_workers=NUM_WORKERS,
#            pin_memory=True,
#        )
#
#    def val_dataloader(self):
#        return DataLoader(
#            self.val_dataset,
#            batch_size=self.batch_size,
#            shuffle=False,
#            num_workers=NUM_WORKERS,
#            pin_memory=True,
#        )
#
#    def test_dataloader(self):
#        return DataLoader(
#            self.test_dataset,
#            batch_size=self.batch_size,
#            shuffle=False,
#            num_workers=4,#NUM_WORKERS,
#            pin_memory=True,
#        )
