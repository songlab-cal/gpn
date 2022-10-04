import argparse
from Bio import SeqIO
from Bio.Seq import Seq
import gzip
import numpy as np
import os
import pandas as pd
import tempfile
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForMaskedLM

import gpn.msa
from gpn.msa.data import GenomeMafIndex


class LogitsMSADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples_path=None,
        data_path=None,
        tokenizer_path=None,
        window_size=None,
        species_path=None,
    ):
        self.examples_path = examples_path
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.window_size = window_size
        self.species_path = species_path

        self.examples = pd.read_parquet(self.examples_path)

        df_pos = self.examples.copy()
        df_pos["start"] = df_pos.pos - self.window_size // 2
        df_pos["end"] = df_pos.start + self.window_size
        df_pos["strand"] = "+"
        df_neg = df_pos.copy()
        df_neg.strand = "-"

        self.df = pd.concat([df_pos, df_neg], ignore_index=True)

    def load_tokenizer(self):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        print("Done.")

    def load_maf(self):
        print("Loading MAF...")
        self.maf_index = GenomeMafIndex(path=self.data_path, chroms=self.examples.chrom.unique(), species_path=self.species_path)
        print("Done.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not hasattr(self, 'tokenizer'):
            self.load_tokenizer()
            self.load_maf()

        row = self.df.iloc[idx]
        seqs = self.maf_index.access(row.chrom, row.start, row.end, row.strand, max_length=self.window_size)
        seqs = [seq.upper() for seq in seqs]
        window_pos = self.window_size // 2
        seq = seqs[0]
        assert len(seq) == self.window_size

        if row.strand == "-":
            window_pos = self.window_size - window_pos - 1

        seq_list = list(seq)
        seq_list[window_pos] = "[MASK]"
        seq = "".join(seq_list)

        seqs[0] = seq

        x = self.tokenizer(
            seqs,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_tensors="pt",
        )
        mask_token_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        x["pos"] = torch.where(x["input_ids"] == mask_token_id)[1][0]
        return x


class MLMforLogitsMSAModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)

    def forward(self, pos=None, **kwargs):
        logits = self.model(**kwargs).logits
        logits = logits[torch.arange(len(pos)), torch.zeros_like(pos), pos]
        return logits


def main(args):
    d = LogitsMSADataset(
        examples_path=args.examples_path,
        data_path=args.data_path,
        tokenizer_path=args.model_path,
        window_size=args.window_size,
        species_path=args.species_path,
    )
    model = MLMforLogitsMSAModel(args.model_path)
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    trainer = Trainer(model=model, args=training_args)

    examples = d.examples
    n_examples = len(examples)
    pred = trainer.predict(test_dataset=d).predictions
    vocab = d.tokenizer.get_vocab()
    id_a = vocab["a"]
    id_c = vocab["c"]
    id_g = vocab["g"]
    id_t = vocab["t"]
    id_gap = vocab["-"]
    pred_pos = pred[:n_examples, [id_a, id_c, id_g, id_t, id_gap]]
    pred_neg = pred[n_examples:, [id_t, id_g, id_c, id_a, id_gap]]
    avg_pred = np.stack((pred_pos, pred_neg)).mean(axis=0)
    examples.loc[:, ["A", "C", "G", "T", "-"]] = avg_pred
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    examples.to_parquet(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run zero-shot variant effect prediction with AutoModelForMaskedLM."
    )
    parser.add_argument("--data-path", help="Data path (dir with maf files)", type=str)
    parser.add_argument("--species-path", help="Species path (target first)", type=str)
    parser.add_argument("--examples-path", help="Input parquet path (needs 'chrom', 'pos')", type=str)
    parser.add_argument(
        "--model-path", help="Model path (local or on HF hub)", type=str
    )
    parser.add_argument("--output-path", help="Output path (parquet)", type=str)
    parser.add_argument("--window-size", help="Window size (bp)", type=int, default=128)
    parser.add_argument(
        "--per-device-eval-batch-size",
        help="Per device eval batch size",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--dataloader-num-workers", help="Dataloader num workers", type=int, default=8
    )
    args = parser.parse_args()
    main(args)
