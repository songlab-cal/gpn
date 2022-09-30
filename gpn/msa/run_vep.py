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


class VEPMSADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        variants_path=None,
        data_path=None,
        tokenizer_path=None,
        window_size=None,
        species_path=None,
    ):
        self.variants_path = variants_path
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.window_size = window_size
        self.species_path = species_path

        self.variants = pd.read_parquet(self.variants_path)

        df_pos = self.variants.copy()
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
        self.maf_index = GenomeMafIndex(path=self.data_path, chroms=self.variants.chrom.unique(), species_path=self.species_path)
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
        ref_str = row.ref
        alt_str = row.alt

        if row.strand == "-":
            window_pos = self.window_size - window_pos - 1
            ref_str = str(Seq(ref_str).reverse_complement())
            alt_str = str(Seq(alt_str).reverse_complement())

        assert seq[window_pos] == ref_str

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
        x["ref"] = torch.tensor(self.tokenizer.encode(ref_str, add_special_tokens=False)[0], dtype=torch.int64)
        x["alt"] = torch.tensor(self.tokenizer.encode(alt_str, add_special_tokens=False)[0], dtype=torch.int64)
        return x


class MLMforVEPMSAModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)

    def forward(self, pos=None, ref=None, alt=None, **kwargs):
        logits = self.model(**kwargs).logits
        logits = logits[torch.arange(len(pos)), torch.zeros_like(pos), pos]
        logits_ref = logits[torch.arange(len(ref)), ref]
        logits_alt = logits[torch.arange(len(alt)), alt]
        llr = logits_alt - logits_ref
        return llr


def main(args):
    d = VEPMSADataset(
        variants_path=args.variants_path,
        data_path=args.data_path,
        tokenizer_path=args.model_path,
        window_size=args.window_size,
        species_path=args.species_path,
    )
    model = MLMforVEPMSAModel(args.model_path)
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
    parser.add_argument("--data-path", help="Data path (dir with maf files)", type=str)
    parser.add_argument("--species-path", help="Species path (target first)", type=str)
    parser.add_argument("--variants-path", help="Variants parquet path", type=str)
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
