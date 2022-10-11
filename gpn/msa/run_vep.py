import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from datasets import load_dataset
import gzip
import numpy as np
import os
import pandas as pd
import tempfile
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForMaskedLM

import gpn.msa


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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    raw_dataset = load_dataset('parquet', data_files={'test': args.data_path})['test']
    print(raw_dataset)

    def tokenize_function(example):
        example["input_ids"] = tokenizer(
            example["seq"],
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
        )["input_ids"]
        example["ref"] = tokenizer.get_vocab()[example["ref"].lower()]
        example["alt"] = tokenizer.get_vocab()[example["alt"].lower()]
        return example

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=False,
        num_proc=args.dataloader_num_workers,
        remove_columns=["seq"],
        desc="Running tokenizer on dataset",
    )
    print(tokenized_dataset)

    model = MLMforVEPMSAModel(args.model_path)
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args)

    pred = trainer.predict(test_dataset=tokenized_dataset).predictions
    n_variants = len(pred)//2
    pred_pos = pred[:n_variants]
    pred_neg = pred[n_variants:]
    avg_pred = np.stack((pred_pos, pred_neg)).mean(axis=0)
    res = pd.Series(avg_pred, name="model_score").to_frame()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    res.to_parquet(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run zero-shot variant effect prediction with AutoModelForMaskedLM."
    )
    parser.add_argument("data_path", help="Variants parquet path", type=str)
    parser.add_argument(
        "model_path", help="Model path (local or on HF hub)", type=str
    )
    parser.add_argument("output_path", help="Output path (parquet)", type=str)
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
