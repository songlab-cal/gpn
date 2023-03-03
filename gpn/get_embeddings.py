import argparse
from Bio import SeqIO, bgzf
from Bio.Seq import Seq
from datasets import load_dataset
import gzip
import numpy as np
import os
import pandas as pd
import tempfile
import torch
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments

import gpn.model
from gpn.utils import Genome, load_dataset_from_file_or_dir


class ModelCenterEmbedding(torch.nn.Module):
    def __init__(self, model_path, center_window_size):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.center_window_size = center_window_size
        
    def get_center_embedding(self, input_ids):
        embedding = self.model.forward(input_ids=input_ids).last_hidden_state
        center = embedding.shape[1] // 2
        left = center - self.center_window_size // 2
        right = center + self.center_window_size // 2
        embedding = embedding[:, left:right]
        embedding = embedding.mean(axis=1)
        return embedding

    def forward(self, input_ids_fwd=None, input_ids_rev=None):
        embedding_fwd = self.get_center_embedding(input_ids_fwd)
        embedding_rev = self.get_center_embedding(input_ids_rev)
        embedding = (embedding_fwd+embedding_rev)/2
        return embedding


def get_embeddings(
    windows, genome, tokenizer, model, per_device_batch_size=8,
    dataloader_num_workers=0,
):
    def tokenize(seqs):
        return tokenizer(
            seqs,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
        )["input_ids"]

    def get_tokenized_seq(vs):
        chrom, start, end = vs["chrom"], vs["start"], vs["end"]
        n = len(chrom)
        seq_fwd, seq_rev = zip(*(
            genome.get_seq_fwd_rev(chrom[i], start[i], end[i]) for i in range(n)
        ))
        res = {}
        res["input_ids_fwd"] = tokenize(seq_fwd)
        res["input_ids_rev"] = tokenize(seq_rev)
        return res

    windows.set_transform(get_tokenized_seq)
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args)
    return trainer.predict(test_dataset=windows).predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get logits with AutoModelForMaskedLM"
    )
    parser.add_argument(
        "windows_path", type=str,
        help="windows path. Needs the following columns: chrom,start,end",
    )
    parser.add_argument(
        "genome_path", type=str, help="Genome path (fasta, potentially compressed)",
    )
    parser.add_argument(
        "center_window_size", type=int,
        help="Genomic window size to average at the center of the windows"
    )
    parser.add_argument(
        "model_path", help="Model path (local or on HF hub)", type=str
    )
    parser.add_argument("output_path", help="Output path (parquet)", type=str)
    parser.add_argument(
        "--per-device-batch-size",
        help="Per device batch size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--tokenizer-path", type=str,
        help="Tokenizer path (optional, else will use model_path)",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split",
    )
    parser.add_argument(
        "--dataloader-num-workers", type=int, default=0, help="Dataloader num workers"
    )
    parser.add_argument(
        "--is-file", action="store_true", help="windows_PATH is a file, not directory",
    )
    parser.add_argument(
        "--format", type=str, default="parquet",
        help="If is-file, specify format (parquet, csv, json)",
    )
    args = parser.parse_args()

    windows = load_dataset_from_file_or_dir(
        args.windows_path, split=args.split, is_file=args.is_file,
        format=args.format,
    )
    genome = Genome(args.genome_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path if args.tokenizer_path else args.model_path
    )
    model = ModelCenterEmbedding(args.model_path, args.center_window_size)
    pred = get_embeddings(
        windows, genome, tokenizer, model,
        per_device_batch_size=args.per_device_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    directory = os.path.dirname(args.output_path)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)
    columns = [f"embedding_{i}" for i in range(pred.shape[1])]
    pd.DataFrame(pred, columns=columns).to_parquet(args.output_path, index=False)
