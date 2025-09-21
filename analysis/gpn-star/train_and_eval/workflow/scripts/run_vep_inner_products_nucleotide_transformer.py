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
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments

import gpn.model
from gpn.data import Genome, load_dataset_from_file_or_dir, token_input_id


window_size = 5994 #11994 #5994
k = 6
nucleotides = list("ACGT")


class VEPEmbedding(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
        )

    def get_embedding(self, input_ids):
        return self.model(input_ids=input_ids).last_hidden_state

    def get_scores(self, input_ids_ref, input_ids_alt):
        embedding_ref = self.get_embedding(input_ids_ref)
        embedding_alt = self.get_embedding(input_ids_alt)
        return (embedding_ref * embedding_alt).sum(dim=1)

    def forward(
        self,
        input_ids_ref_fwd=None,
        input_ids_alt_fwd=None,
        input_ids_ref_rev=None,
        input_ids_alt_rev=None,
    ):
        fwd = self.get_scores(input_ids_ref_fwd, input_ids_alt_fwd)
        rev = self.get_scores(input_ids_ref_rev, input_ids_alt_rev)
        return (fwd + rev) / 2


def run_vep(
    variants,
    genome,
    tokenizer,
    model,
    per_device_batch_size=8,
    dataloader_num_workers=0,
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
        start = pos - window_size // 2
        end = pos + window_size // 2
        seq_fwd, seq_rev = zip(
            *(genome.get_seq_fwd_rev(chrom[i], start[i], end[i]) for i in range(n))
        )
        seq_fwd = np.array([list(seq.upper()) for seq in seq_fwd], dtype="object")
        seq_rev = np.array([list(seq.upper()) for seq in seq_rev], dtype="object")
        n_kmers = window_size // k
        assert seq_fwd.shape[1] == window_size
        assert seq_rev.shape[1] == window_size
        ref_fwd = np.array(vs["ref"])
        alt_fwd = np.array(vs["alt"])
        ref_rev = np.array([str(Seq(x).reverse_complement()) for x in ref_fwd])
        alt_rev = np.array([str(Seq(x).reverse_complement()) for x in alt_fwd])
        pos_fwd = window_size // 2
        pos_rev = pos_fwd - 1 if window_size % 2 == 0 else pos_fwd

        def handle_undefined(seq):
            # the goal is to have a fixed length sequence of tokens
            # the issue is that some sequences have Ns which distort the length
            # so we replace kmers containing N with a single N, so it maps to a single
            # unk token
            res = []
            for x in seq:
                kmers = ["".join(x[i*k:(i+1)*k]) for i in range(n_kmers)]
                kmers = [kmer if set(kmer) <= set(nucleotides) else "N" for kmer in kmers]
                res.append("".join(kmers))
            return res

        def prepare_output(seq, pos, ref, alt):
            assert (seq[:, pos] == ref).all(), f"{seq[:, pos]}, {ref}"
            seq_ref = seq
            seq_alt = seq.copy()
            seq_alt[:, pos] = alt
            return (
                np.array(tokenize(handle_undefined(seq_ref))),
                np.array(tokenize(handle_undefined(seq_alt))),
            )

        res = {}
        res["input_ids_ref_fwd"], res["input_ids_alt_fwd"] = prepare_output(seq_fwd, pos_fwd, ref_fwd, alt_fwd)
        res["input_ids_ref_rev"], res["input_ids_alt_rev"] = prepare_output(seq_rev, pos_rev, ref_rev, alt_rev)
        return res

    variants.set_transform(get_tokenized_seq)
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        torch_compile=True,
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
        "--per_device_batch_size",
        help="Per device batch size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--tokenizer_path", type=str,
        help="Tokenizer path (optional, else will use model_path)",
    )
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=0, help="Dataloader num workers"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split",
    )
    parser.add_argument(
        "--is_file", action="store_true", help="VARIANTS_PATH is a file, not directory",
    )
    args = parser.parse_args()

    variants = load_dataset_from_file_or_dir(
        args.variants_path, split=args.split, is_file=args.is_file,
    )
    subset_chroms = np.unique(variants["chrom"])
    genome = Genome(args.genome_path, subset_chroms=subset_chroms)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path if args.tokenizer_path else args.model_path
    )
    model = VEPEmbedding(args.model_path)
    pred = run_vep(
        variants, genome, tokenizer, model,
        per_device_batch_size=args.per_device_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    directory = os.path.dirname(args.output_path)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)
    pd.DataFrame(
        pred, columns=[f"embed_{i}" for i in range(pred.shape[1])]
    ).to_parquet(args.output_path, index=False)
