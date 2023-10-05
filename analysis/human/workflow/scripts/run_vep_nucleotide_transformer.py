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
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments

import gpn.model
from gpn.data import Genome, load_dataset_from_file_or_dir, token_input_id

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


window_size = 5994
n_prefix = 1  # CLS
k = 6
nucleotides = list("ACGT")


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
    variants, genome, tokenizer, model,
    per_device_batch_size=8, dataloader_num_workers=0,
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
        # we convert from 1-based coordinate (standard in VCF) to 
        # 0-based, to use with Genome
        chrom = np.array(vs["chrom"])
        n = len(chrom)
        pos = np.array(vs["pos"]) - 1
        start = pos - window_size//2
        end = pos + window_size//2
        seq_fwd, seq_rev = zip(*(
            genome.get_seq_fwd_rev(chrom[i], start[i], end[i]) for i in range(n)
        ))
        seq_fwd = np.array([list(seq.upper()) for seq in seq_fwd], dtype="object")
        seq_rev = np.array([list(seq.upper()) for seq in seq_rev], dtype="object")
        assert seq_fwd.shape[1] == window_size
        assert np.isin(seq_fwd, nucleotides).all()
        n_kmers = window_size // k
        pos = n_kmers // 2  # pos of the central kmer in the kmers

        def get_kmer(seqs):
            return np.array([seq[pos*k:(pos+1)*k] for seq in seqs])

        pos_in_kmer_fwd = k // 2
        pos_in_kmer_rev = k // 2 - 1

        ref_fwd = np.array(vs["ref"])
        alt_fwd = np.array(vs["alt"])
        ref_rev = np.array([str(Seq(x).reverse_complement()) for x in ref_fwd])
        alt_rev = np.array([str(Seq(x).reverse_complement()) for x in alt_fwd])

        mask_id = tokenizer.mask_token_id

        def prepare_output(seq, pos_in_kmer, ref, alt):
            ref_kmer = get_kmer(seq)
            assert (ref_kmer[:, pos_in_kmer] == ref).all(), f"{ref_kmer[:, pos_in_kmer]}, {ref}"
            alt_kmer = ref_kmer.copy()
            alt_kmer[:, pos_in_kmer] = alt
            input_ids = np.array(tokenize(["".join(x) for x in seq]))
            input_ids[:, n_prefix+pos] = mask_id

            return (
                input_ids,
                [pos + n_prefix for _ in range(n)],
                [tokenizer.token_to_id("".join(x)) for x in ref_kmer],
                [tokenizer.token_to_id("".join(x)) for x in alt_kmer],
            )

        res = {}
        res["input_ids_fwd"], res["pos_fwd"], res["ref_fwd"], res["alt_fwd"] = prepare_output(
            seq_fwd, pos_in_kmer_fwd, ref_fwd, alt_fwd
        )
        res["input_ids_rev"], res["pos_rev"], res["ref_rev"], res["alt_rev"] = prepare_output(
            seq_rev, pos_in_kmer_rev, ref_rev, alt_rev
        )
        return res

    variants.set_transform(get_tokenized_seq)
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_num_workers=dataloader_num_workers,
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
        help="Variants path. Needs the following columns: chrom,pos,ref,alt. pos should be 1-based",
    )
    parser.add_argument(
        "genome_path", type=str, help="Genome path (fasta, potentially compressed)",
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
        "--dataloader-num-workers", type=int, default=0, help="Dataloader num workers"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split",
    )
    parser.add_argument(
        "--is-file", action="store_true", help="VARIANTS_PATH is a file, not directory",
    )
    parser.add_argument(
        "--format", type=str, default="parquet",
        help="If is-file, specify format (parquet, csv, json)",
    )
    args = parser.parse_args()

    variants = load_dataset_from_file_or_dir(
        args.variants_path, split=args.split, is_file=args.is_file,
        format=args.format,
    )
    subset_chroms = np.unique(variants["chrom"])
    genome = Genome(args.genome_path, subset_chroms=subset_chroms)

    df = variants.to_pandas()

    def check_valid(v):
        pos = v.pos - 1
        start = pos - window_size//2
        end = pos + window_size//2
        seq = genome.get_seq(v.chrom, start, end).upper()
        no_undefined = np.isin(list(seq), nucleotides).all()
        clinvar_subset = (
            v["source"]=="ClinVar" or
            (v["label"]=="Common" and "missense" in v["consequence"])
        )
        return no_undefined and clinvar_subset

    df["is_valid"] = df.parallel_apply(check_valid, axis=1)
    print(df.is_valid.value_counts())
    variants = variants.select(np.where(df.is_valid)[0])

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path if args.tokenizer_path else args.model_path
    )
    model = MLMforVEPModel(args.model_path)
    pred = run_vep(
        variants, genome, tokenizer, model,
        per_device_batch_size=args.per_device_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    directory = os.path.dirname(args.output_path)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)
    df.loc[df.is_valid, "score"] = pred
    print(df["score"].describe())
    df[["score"]].to_parquet(args.output_path, index=False)
