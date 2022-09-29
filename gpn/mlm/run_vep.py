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

import gpn.mlm


class VEPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        variants_path=None,
        genome_path=None,
        tokenizer_path=None,
        window_size=None,
    ):
        self.variants_path = variants_path
        self.genome_path = genome_path
        self.tokenizer_path = tokenizer_path
        self.window_size = window_size

        self.variants = pd.read_parquet(self.variants_path)

        df_pos = self.variants.copy()
        df_pos["start"] = df_pos.pos - self.window_size // 2
        df_pos["end"] = df_pos.start + self.window_size
        df_pos["strand"] = "+"
        df_neg = df_pos.copy()
        df_neg.strand = "-"

        self.df = pd.concat([df_pos, df_neg], ignore_index=True)

    def __len__(self):
        return len(self.df)

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def load_genome(self):
        with gzip.open(self.genome_path, "rt") if args.fasta_path.endswith(
            ".gz"
        ) else open(self.genome_path) as handle:
            self.genome = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))

    def __getitem__(self, idx):
        if not hasattr(self, "tokenizer"):
            self.load_tokenizer()
            self.load_genome()

        row = self.df.iloc[idx]
        seq = self.genome[row.chrom][row.start : row.end].seq
        window_pos = self.window_size // 2
        assert len(seq) == self.window_size
        ref_str = row.ref
        alt_str = row.alt

        if row.strand == "-":
            seq = seq.reverse_complement()
            window_pos = self.window_size - window_pos - 1
            ref_str = str(Seq(ref_str).reverse_complement())
            alt_str = str(Seq(alt_str).reverse_complement())
        seq = str(seq).upper()

        assert seq[window_pos] == ref_str

        seq_list = list(seq)
        seq_list[window_pos] = "[MASK]"
        seq = "".join(seq_list)

        x = self.tokenizer(
            seq,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        x["input_ids"] = x["input_ids"].flatten()
        mask_token_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        x["pos"] = torch.where(x["input_ids"] == mask_token_id)[0][0]
        x["ref"] = torch.tensor(
            self.tokenizer.encode(ref_str, add_special_tokens=False)[0],
            dtype=torch.int64,
        )
        x["alt"] = torch.tensor(
            self.tokenizer.encode(alt_str, add_special_tokens=False)[0],
            dtype=torch.int64,
        )
        return x


class MLMforVEPModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)

    def forward(self, pos=None, ref=None, alt=None, **kwargs):
        logits = self.model(**kwargs).logits
        logits = logits[torch.arange(len(pos)), pos]
        logits_ref = logits[torch.arange(len(ref)), ref]
        logits_alt = logits[torch.arange(len(alt)), alt]
        llr = logits_alt - logits_ref
        return llr


def main(args):
    d = VEPDataset(
        variants_path=args.variants_path,
        genome_path=args.fasta_path,
        tokenizer_path=args.model_path,
        window_size=args.window_size,
    )
    model = MLMforVEPModel(args.model_path)
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    trainer = Trainer(model=model, args=training_args)

    variants = d.variants
    n_variants = len(variants)
    pred = trainer.predict(test_dataset=d).predictions
    pred_pos = pred[:n_variants]
    pred_neg = pred[n_variants:]
    avg_pred = np.stack((pred_pos, pred_neg)).mean(axis=0)
    variants.loc[:, "model_score"] = avg_pred
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    variants.to_parquet(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run zero-shot variant effect prediction with AutoModelForMaskedLM."
    )
    parser.add_argument("--fasta-path", help="Genome fasta path", type=str)
    parser.add_argument("--variants-path", help="Variants parquet path", type=str)
    parser.add_argument(
        "--model-path", help="Model path (local or on HF hub)", type=str
    )
    parser.add_argument("--output-path", help="Output path (parquet)", type=str)
    parser.add_argument("--window-size", help="Window size (bp)", type=int, default=512)
    parser.add_argument(
        "--per-device-eval-batch-size",
        help="Per device eval batch size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--dataloader-num-workers", help="Dataloader num workers", type=int, default=8
    )
    args = parser.parse_args()
    main(args)
