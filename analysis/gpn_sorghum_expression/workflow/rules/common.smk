from datasets import Dataset, load_dataset
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.preprocessing import StandardScaler
import tempfile
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


def dict_to_config_string(config_dict):
    # Convert dict to list of --key value strings and then join with space
    return ' '.join(f'--{k} {v}' for k, v in config_dict.items() if v is not None)


def get_batch_size(batch_size):
    per_device_batch_size = config["finetuning_per_device_batch_size"]
    n_gpu = config["finetuning_n_gpu"]
    if batch_size < per_device_batch_size:
        per_device_batch_size = batch_size
    assert batch_size % (per_device_batch_size * n_gpu) == 0
    gradient_accumulation_steps = batch_size // (per_device_batch_size * n_gpu)
    return (
        f"--per_device_train_batch_size {per_device_batch_size} "
        f"--per_device_eval_batch_size {per_device_batch_size} "
        f"--gradient_accumulation_steps {gradient_accumulation_steps}"
    )


def get_hparams(wildcards):
    hparams = config["finetuning"]["hparams"]["default"].copy()
    hparams.update(config["finetuning"]["hparams"][wildcards.hparams].copy())

    batch_size = hparams.pop("batch_size")
    batch_size_config = get_batch_size(batch_size)

    return dict_to_config_string(hparams) + " " + batch_size_config


def run_prediction_lfc(wildtype, mutants, model_path, track_index, **kwargs):
    d = Dataset.from_pandas(
        pd.DataFrame(dict(seq=[wildtype] + mutants)), preserve_index=False
    )
    preds = run_prediction(d, model_path, **kwargs)
    preds = preds[:, track_index]
    preds = np.expm1(preds)
    lfc = np.log(preds) - np.log(preds[0])
    lfc = lfc[1:]
    return lfc


def run_prediction(d, model_path, batch_size=256, threads=8):
    import gpn.model

    tokenizer_path = "songlab/tokenizer-dna-mlm"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if "lora" not in model_path:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        print("hardcoding lora model config")
        from peft import AutoPeftModelForSequenceClassification
        model = AutoPeftModelForSequenceClassification.from_pretrained(model_path, num_labels=26, regression_softplus=True)

    def tokenize(seq):
        return tokenizer(
            seq, return_tensors="pt", return_attention_mask=False,
            return_token_type_ids=False
        )

    d = d.map(
        lambda examples: tokenize(examples["seq"]),
        batched=True,
        num_proc=threads,
        remove_columns=d.column_names,
    )
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=threads,
        torch_compile=False,
        bf16=True,
        bf16_full_eval=True,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args)
    return trainer.predict(test_dataset=d).predictions


rule random_init_gpn_brassicales:
    output:
        directory("results/checkpoints_pretraining/random_gpn_brassicales"),
    run:
        model_path = "songlab/gpn-brassicales"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(output[0])

        set_seed(42)
        model = AutoModel.from_config(AutoConfig.from_pretrained(model_path))
        model.save_pretrained(output[0])