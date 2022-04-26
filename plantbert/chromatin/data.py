from datasets import load_dataset
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from transformers import AutoTokenizer


K = 6


def encode_dna_seq(seq):
    return np.array([encode_base(base) for base in seq], dtype=np.uint8)


def encode_base(base):
    if base == "N":
        base = np.random.choice(["A", "C", "G", "T"])
    encoding = {"A": 0, "C": 1, "G": 2, "T": 3}
    result = encoding[base]
    return result


class DataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


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
        print(data_path, self.df)
        self.features = [col for col in self.df.columns if col not in ["chromosome", "start", "end", "strand", "seq"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row.seq
        d = {}
        d["input_ids"] = encode_dna_seq(x).astype(int)
        d["Y"] = row[self.features].values.astype(np.uint8)
        return d


class DeepSEADataModule(DataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        print("Loading train dataset")
        self.train_dataset = DeepSEADataset(os.path.join(self.data_dir, "train.parquet"))
        print("Loading val dataset")
        self.val_dataset = DeepSEADataset(os.path.join(self.data_dir, "val.parquet"))
        print("Loading test dataset")
        self.test_dataset = DeepSEADataset(os.path.join(self.data_dir, "test.parquet"))
        print(len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))


class DNABERTDataset(Dataset):
    def __init__(self, data_path, language_model_name):
        self.df = pd.read_parquet(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.features = [col for col in self.df.columns if col not in ["chromosome", "start", "end", "strand", "seq"]]

        #n = len(self.df)
        #idx = np.concatenate([np.arange(100), np.arange(100) + n//2])
        #self.df = self.df.iloc[idx]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row.seq
        x = seq2kmer(x, K)
        x = self.tokenizer(x, padding="max_length", max_length=1000, return_token_type_ids=False, return_tensors="pt")
        x["input_ids"] = x["input_ids"].flatten()
        x["attention_mask"] = x["attention_mask"].flatten()
        y = row[self.features].values.astype(np.uint8)
        return x, y


class DNABERTDataModule(DataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        language_model_name,
        num_workers,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.language_model_name = language_model_name
        self.num_workers = num_workers

    def prepare_data(self):
        print("Loading train dataset")
        self.train_dataset = DNABERTDataset(os.path.join(self.data_dir, "train.parquet"), self.language_model_name)
        print("Loading val dataset")
        self.val_dataset = DNABERTDataset(os.path.join(self.data_dir, "val.parquet"), self.language_model_name)
        print("Loading test dataset")
        self.test_dataset = DNABERTDataset(os.path.join(self.data_dir, "test.parquet"), self.language_model_name)
        print(len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))


class PlantBertDataset(Dataset):
    def __init__(self, data_path, language_model_path, max_length):
        self.df = pd.read_parquet(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_path)  # this should be loaded later to avoid memory leak with num_workers>0
        self.max_length = max_length
        self.features = [col for col in self.df.columns if col not in ["chromosome", "start", "end", "strand", "seq"]]

        #n = len(self.df)
        #idx = np.concatenate([np.arange(100), np.arange(100) + n//2])
        #self.df = self.df.iloc[idx]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row.seq
        #x = x[400:600]
        #x = x[300:700]
        x = self.tokenizer(x, padding="max_length", max_length=self.max_length, return_token_type_ids=False, return_tensors="pt", truncation=True)
        d = dict(
            input_ids=x["input_ids"].flatten(),
            attention_mask=x["attention_mask"].flatten(),
            Y=torch.tensor(row[self.features].values.astype(np.uint8)),
        )
        #x["global_attention_mask"] = torch.zeros_like(x["input_ids"])
        #x["global_attention_mask"][0] = 1
        return d


class PlantBertDataModule(DataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        language_model_path,
        num_workers,
        max_length,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.language_model_path = language_model_path
        self.num_workers = num_workers
        self.max_length = max_length

    def prepare_data(self):
        print("Loading train dataset")
        self.train_dataset = PlantBertDataset(os.path.join(self.data_dir, "train.parquet"), self.language_model_path, self.max_length)
        print("Loading val dataset")
        self.val_dataset = PlantBertDataset(os.path.join(self.data_dir, "val.parquet"), self.language_model_path, self.max_length)
        print("Loading test dataset")
        self.test_dataset = PlantBertDataset(os.path.join(self.data_dir, "test.parquet"), self.language_model_path, self.max_length)
        print(len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))
