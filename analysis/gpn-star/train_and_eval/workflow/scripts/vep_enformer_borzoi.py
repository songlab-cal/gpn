import argparse
from Bio.Seq import Seq
from datasets import load_dataset
from gpn.data import Genome, load_dataset_from_file_or_dir
import grelu.resources
from grelu.sequence.format import strings_to_one_hot
import numpy as np
import os
import pandas as pd
import tempfile
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments


class VEPModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_scores(self, x_ref, x_alt):
        y_ref = self.model(x_ref)
        y_alt = self.model(x_alt)
        lfc = torch.log2(1 + y_alt) - torch.log2(1 + y_ref)
        l2 = torch.linalg.norm(lfc, dim=2)
        return l2

    def forward(
        self,
        x_ref_fwd=None,
        x_alt_fwd=None,
        x_ref_rev=None,
        x_alt_rev=None,
    ):
        fwd = self.get_scores(x_ref_fwd, x_alt_fwd)
        rev = self.get_scores(x_ref_rev, x_alt_rev)
        return (fwd + rev) / 2


def run_vep(
    variants,
    genome,
    window_size,
    model,
    per_device_batch_size=8,
    dataloader_num_workers=0,
):
    def transform(V):
        # we convert from 1-based coordinate (standard in VCF) to
        # 0-based, to use with Genome
        chrom = np.array(V["chrom"])
        n = len(chrom)
        pos = np.array(V["pos"]) - 1
        start = pos - window_size // 2
        end = pos + window_size // 2
        seq_fwd, seq_rev = zip(
            *(genome.get_seq_fwd_rev(chrom[i], start[i], end[i]) for i in range(n))
        )
        seq_fwd = np.array([list(seq.upper()) for seq in seq_fwd], dtype="object")
        seq_rev = np.array([list(seq.upper()) for seq in seq_rev], dtype="object")
        assert seq_fwd.shape[1] == window_size
        assert seq_rev.shape[1] == window_size
        ref_fwd = np.array(V["ref"])
        alt_fwd = np.array(V["alt"])
        ref_rev = np.array([str(Seq(x).reverse_complement()) for x in ref_fwd])
        alt_rev = np.array([str(Seq(x).reverse_complement()) for x in alt_fwd])
        pos_fwd = window_size // 2
        pos_rev = pos_fwd - 1 if window_size % 2 == 0 else pos_fwd

        def prepare_output(seq, pos, ref, alt):
            assert (seq[:, pos] == ref).all(), f"{seq[:, pos]}, {ref}"
            seq_ref = seq
            seq_alt = seq.copy()
            seq_alt[:, pos] = alt
            return (
                strings_to_one_hot(["".join(x) for x in seq_ref]),
                strings_to_one_hot(["".join(x) for x in seq_alt]),
            )

        res = {}
        res["x_ref_fwd"], res["x_alt_fwd"] = prepare_output(
            seq_fwd, pos_fwd, ref_fwd, alt_fwd
        )
        res["x_ref_rev"], res["x_alt_rev"] = prepare_output(
            seq_rev, pos_rev, ref_rev, alt_rev
        )
        return res

    variants.set_transform(transform)
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args)
    return trainer.predict(test_dataset=variants).predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variant effect prediction")
    parser.add_argument(
        "variants_path",
        type=str,
        help="Variants path. Needs the following columns: chrom,pos,ref,alt. pos should be 1-based",
    )
    parser.add_argument(
        "genome_path",
        type=str,
        help="Genome path (fasta, potentially compressed)",
    )
    parser.add_argument("project", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("output_path", help="Output path (parquet)", type=str)
    parser.add_argument(
        "--per_device_batch_size",
        help="Per device batch size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=0, help="Dataloader num workers"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--is_file",
        action="store_true",
        help="VARIANTS_PATH is a file, not directory",
    )
    args = parser.parse_args()

    variants = load_dataset_from_file_or_dir(
        args.variants_path,
        split=args.split,
        is_file=args.is_file,
    )
    genome = Genome(args.genome_path)

    model = grelu.resources.load_model(project=args.project, model_name=args.model_name)
    columns = model.data_params["tasks"]["name"]
    window_size = model.data_params["train_seq_len"]
    model = VEPModel(model.model)

    pred = run_vep(
        variants,
        genome,
        window_size,
        model,
        per_device_batch_size=args.per_device_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    directory = os.path.dirname(args.output_path)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)
    pd.DataFrame(pred, columns=columns).to_parquet(args.output_path, index=False)
