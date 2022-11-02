import argparse
from datasets import load_dataset
import numpy as np
import os
import pandas as pd
import tempfile
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModel

import gpn.model


class DeltaEmbeddingModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)

    def forward(self, input_ids=None, pos=None, ref=None, alt=None):
        input_ids_ref = torch.clone(input_ids)
        input_ids_ref[torch.arange(len(pos)), pos] = ref
        embedding_ref = self.model(input_ids=input_ids_ref).last_hidden_state
        input_ids_alt = torch.clone(input_ids)
        input_ids_alt[torch.arange(len(pos)), pos] = alt
        embedding_alt = self.model(input_ids=input_ids_alt).last_hidden_state
        
        # approach 1
        #delta_embedding = embedding_alt - embedding_ref
        #delta_embedding = torch.linalg.norm(delta_embedding, ord=2, axis=1)

        # approach 2
        delta_embedding = (
            embedding_alt[torch.arange(len(pos)), pos] -
            embedding_ref[torch.arange(len(pos)), pos]
        )

        return delta_embedding


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    raw_dataset = load_dataset('parquet', data_files={'test': args.data_path})['test']
    raw_dataset = raw_dataset.select(range(len(raw_dataset)//2))  # ignoring the reverse complement for now
    #raw_dataset = raw_dataset.select(range(1000))
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
        example["ref"] = tokenizer.encode(example["ref"], add_special_tokens=False)[0]
        example["alt"] = tokenizer.encode(example["alt"], add_special_tokens=False)[0]
        return example

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=False,
        num_proc=args.dataloader_num_workers,
        remove_columns=["seq"],
        desc="Running tokenizer on dataset",
    )
    print(tokenized_dataset)

    model = DeltaEmbeddingModel(args.model_path)
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args)

    pred = trainer.predict(test_dataset=tokenized_dataset).predictions
    print(pred.shape)
    res = pd.DataFrame(pred, columns=[f"delta_embed_{i}" for i in range(pred.shape[1])])
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    res.to_parquet(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get delta embeddings with AutoModel."
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
