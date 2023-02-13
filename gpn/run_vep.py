import argparse
from Bio import SeqIO, bgzf
from Bio.Seq import Seq
from datasets import load_dataset
import gzip
import os
import pandas as pd
import tempfile
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments

import gpn.model
from gpn.utils import Genome


class MLMforVEPModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_path, trust_remote_code=True,
        )
        
    def get_llr(self, input_ids, pos, ref, alt):
        logits = self.model.forward(input_ids=input_ids).logits
        logits = logits[torch.arange(len(pos)), pos]
        logits_ref = logits[torch.arange(len(ref)), ref]
        logits_alt = logits[torch.arange(len(alt)), alt]
        llr = logits_alt - logits_ref
        return llr

    def forward(
        self,
        input_ids_fwd=None,
        pos_fwd=None,
        ref_fwd=None,
        alt_fwd=None,
        input_ids_rev=None,
        pos_rev=None,
        ref_rev=None,
        alt_rev=None,
    ):
        llr_fwd = self.get_llr(input_ids_fwd, pos_fwd, ref_fwd, alt_fwd)
        llr_rev = self.get_llr(input_ids_rev, pos_rev, ref_rev, alt_rev)
        llr = (llr_fwd+llr_rev)/2
        return llr


def run_vep(
    variants, genome, window_size, tokenizer, model,
    n_prefix=0, per_device_batch_size=8,
):
    n_variants = len(list(
        variants.map(lambda examples: {"chrom": examples["chrom"]}, batched=True)
    ))
    original_cols = list(variants.take(1))[0].keys()

    def tokenize(seq):
        return tokenizer(
            seq,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
        )["input_ids"]

    def token_input_id(token):
        return tokenizer(token)["input_ids"][n_prefix]

    def get_tokenized_seq(v):
        # we convert from 1-based coordinate (standard in VCF) to 
        # 0-based, to use with Genome
        pos = v["pos"] - 1
        start = pos - window_size//2
        end = pos + window_size//2
        
        seq_fwd, seq_rev = genome.get_seq_fwd_rev(v["chrom"], start, end)
        seq_fwd = list(seq_fwd.upper())
        seq_rev = list(seq_rev.upper())
        assert len(seq_fwd) == window_size
        assert len(seq_rev) == window_size
        
        ref_fwd = v["ref"]
        alt_fwd = v["alt"]
        ref_rev = str(Seq(ref_fwd).reverse_complement())
        alt_rev = str(Seq(alt_fwd).reverse_complement())

        pos_fwd = window_size // 2
        pos_rev = pos_fwd - 1 if window_size % 2 == 0 else pos_fwd
        assert seq_fwd[pos_fwd] == ref_fwd, f"{seq_fwd[pos_fwd]}, {ref_fwd}"
        seq_fwd[pos_fwd] = tokenizer.mask_token
        assert seq_rev[pos_rev] == ref_rev, f"{seq_rev[pos_rev]}, {ref_rev}"
        seq_rev[pos_rev] = tokenizer.mask_token

        v["input_ids_fwd"] = tokenize("".join(seq_fwd))
        v["pos_fwd"] = pos_fwd + n_prefix
        v["ref_fwd"] = token_input_id(ref_fwd)
        v["alt_fwd"] = token_input_id(alt_fwd)
        
        v["input_ids_rev"] = tokenize("".join(seq_rev))
        v["pos_rev"] = pos_rev + n_prefix
        v["ref_rev"] = token_input_id(ref_rev)
        v["alt_rev"] = token_input_id(alt_rev)
        
        return v

    # TODO: batched mapping might improve performance
    variants = variants.map(get_tokenized_seq, remove_columns=original_cols)
    # Ugly hack to be able to display a progress bar
    # Warning: this will override len() for all instances of datasets.IterableDataset
    # Didn't find a way to just override for this instance
    variants.__class__.__len__ = lambda self: n_variants
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args)
    return trainer.predict(test_dataset=variants).predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run zero-shot variant effect prediction with AutoModelForMaskedLM"
    )
    parser.add_argument(
        "variants_path", type=str,
        help="Variants path. Needs the following columns: chrom,pos,ref,alt",
    )
    parser.add_argument(
        "genome_path", type=str, help="Genome path (fasta, potentially compressed)",
    )
    parser.add_argument("window_size", type=int, help="Genomic window size")
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
        "--n-prefix", type=int, default=0, help="Number of prefix tokens (e.g. CLS)."
    )
    args = parser.parse_args()

    # TODO: there should be more flexibility here, including loading
    # from a local file, and having different split names
    variants = load_dataset(args.variants_path, streaming=True, split="test")

    variants = variants.take(1000)

    genome = Genome(args.genome_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path if args.tokenizer_path else args.model_path
    )
    model = MLMforVEPModel(args.model_path)
    pred = run_vep(
        variants, genome, args.window_size, tokenizer, model,
        per_device_batch_size=args.per_device_batch_size, n_prefix=args.n_prefix,
    )
    directory = os.path.dirname(args.output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # TODO: could add chrom,pos,ref,alt besides the score, as in CADD output
    # or make it an option
    # could also compress the output with bgzip, as tsv, so it can be indexed with tabix
    pd.DataFrame(pred, columns=["score"]).to_parquet(args.output_path, index=False)
