import argparse
from datasets import load_dataset, Features, Sequence, Value
from einops import rearrange
import numpy as np
import os
import pandas as pd
import tempfile
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM

from gpn.utils import add_space_every_k


class CLMforVEP(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def forward(self, input_ids_ref=None, input_ids_alt=None):
        # not sure clone() incurs unnecessary overhead

        B, L = input_ids_ref.shape

        loss_fct = CrossEntropyLoss(reduction="none")

        def get_ll_per_seq(input_ids):
            logits = self.model(input_ids=input_ids).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.to(logits.dtype)
            loss = loss.reshape(B, L-1)
            loss = loss.mean(dim=1)  # TODO: make sure this doesn't count nan in the future, e.g. padding
            return - loss

        ll_ref = get_ll_per_seq(input_ids_ref)
        ll_alt = get_ll_per_seq(input_ids_alt)
        preds = ll_alt - ll_ref
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
        "seq_ref": Value("string"),
        "seq_alt": Value("string"),
    })

    raw_dataset = load_dataset(
        'parquet', data_files={'test': args.data_path},
        features=raw_features,
    )['test']
    print(raw_dataset)

    tokenized_features = Features({
        "input_ids_ref": Sequence(Value("int16")),
        "input_ids_alt": Sequence(Value("int16")),
    })

    def tokenize_function(example):
        for x in ["ref", "alt"]:
            example[f"input_ids_{x}"] = tokenizer(
                add_space_every_k(example[f"seq_{x}"], 4),
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
        remove_columns=["seq_ref", "seq_alt"],
        features=tokenized_features,
        desc="Running tokenizer on dataset",
    )
    print(tokenized_dataset)

    model = CLMforVEP(args.model_path)
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args)

    pred = trainer.predict(test_dataset=tokenized_dataset).predictions[1]
    print(pred.shape)
    n_variants = len(pred)//2
    pred_pos = pred[:n_variants]
    pred_neg = pred[n_variants:]
    avg_pred = np.stack((pred_pos, pred_neg)).mean(axis=0)
    res = pd.Series(avg_pred, name="model_score").to_frame()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    res.to_parquet(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run zero-shot variant effect prediction with AutoModelForCausalLM."
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
