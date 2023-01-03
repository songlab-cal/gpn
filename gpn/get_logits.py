import argparse
from datasets import load_dataset, Features, Sequence, Value
import numpy as np
import os
import pandas as pd
import tempfile
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForMaskedLM

import gpn.model


class MLMforLogitsModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)

    def forward(self, **kwargs):
        preds = self.model.get_logits(**kwargs)
        loss = torch.zeros_like(preds)  # not used
        return loss, preds


def main(args):
    if args.disable_caching:
        print("Disabling caching")
        datasets.disable_caching()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path if args.tokenizer_path else args.model_path
    )

    raw_features = Features({
        "seq": Value("string"),
        "pos": Value("int16"),
    })

    raw_dataset = load_dataset(
        'parquet', data_files={'test': args.data_path},
        features=raw_features,
    )['test']
    print(raw_dataset)

    tokenized_features = Features({
        "input_ids": Sequence(Value("int8")),
        "pos": Value("int16"),
    })

    def tokenize_function(example):
        example["input_ids"] = tokenizer(
            example["seq"],
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
        )["input_ids"]
        return example

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=False,
        num_proc=args.dataloader_num_workers,
        remove_columns=["seq"],
        features=tokenized_features,
        desc="Running tokenizer on dataset",
    )
    print(tokenized_dataset)

    model = MLMforLogitsModel(args.model_path)
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args)

    pred = trainer.predict(test_dataset=tokenized_dataset).predictions
    n_examples = len(pred)//2
    vocab = tokenizer.get_vocab()
    id_a = vocab["a"]
    id_c = vocab["c"]
    id_g = vocab["g"]
    id_t = vocab["t"]
    pred_pos = pred[:n_examples, [id_a, id_c, id_g, id_t]]
    pred_neg = pred[n_examples:, [id_t, id_g, id_c, id_a]]
    avg_pred = np.stack((pred_pos, pred_neg)).mean(axis=0)
    res = pd.DataFrame(avg_pred, columns=["A", "C", "G", "T"])
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    res.to_parquet(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get logits with AutoModelForMaskedLM."
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
    parser.add_argument("--disable-caching", help="Disable caching of tokenized dataset", action="store_true")
    parser.add_argument("--tokenizer-path", help="Tokenizer path (optional, else will use model_path)", type=str)
    args = parser.parse_args()
    main(args)
