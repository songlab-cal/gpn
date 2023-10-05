import argparse
from datasets import load_dataset, disable_caching
import os
import tempfile
from transformers import Trainer, TrainingArguments

from gpn.data import load_dataset_from_file_or_dir
from gpn.data import GenomeMSA
from gpn.msa.vep import VEPInference
from gpn.msa.logits import LogitsInference
from gpn.msa.embedding import EmbeddingInference


disable_caching()


class_mapping = {
    "vep": VEPInference,
    "logits": LogitsInference,
    "embedding": EmbeddingInference,
}


def run_inference(
    dataset,
    inference,
    per_device_batch_size=8,
    dataloader_num_workers=0,
):
    dataset.set_transform(inference.tokenize_function)
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=inference.model, args=training_args)
    pred = trainer.predict(test_dataset=dataset).predictions
    return inference.postprocess(pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with AutoModelForMaskedLM",
    )
    parser.add_argument(
        "command",
        type=str,
        help="""Command to run:
        - vep: zero-shot variant effect prediction (LLR)
        - logits: masked language model logits
        - embedding: averaged embedding from last layer
        """,
        choices=["vep", "logits", "embedding"],
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="""Input path, either HF dataset, parquet, csv/tsv, vcf, with columns:
        - vep: chrom, pos, ref, alt
        - logits: chrom, pos
        - embedding: chrom, start, end
        """,
    )
    parser.add_argument(
        "msa_path",
        type=str,
        help="Genome MSA path (zarr)",
    )
    parser.add_argument("window_size", type=int, help="Genomic window size")
    parser.add_argument("model_path", help="Model path (local or on HF hub)", type=str)
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
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        help="If is-file, specify format (parquet, csv, json)",
    )
    parser.add_argument(
        "--disable_aux_features",
        action="store_true",
    )
    parser.add_argument(
        "--center_window_size",
        type=int,
        help="[embedding] Genomic window size to average at the center of the windows",
    )
    args = parser.parse_args()
    print(args)

    dataset = load_dataset_from_file_or_dir(
        args.input_path,
        split=args.split,
        is_file=args.is_file,
        format=args.format,
    )
    genome_msa = GenomeMSA(
        args.msa_path, subset_chroms=dataset.unique("chrom"), in_memory=False
    )
    # sorry this is hacky, should use subparsers
    kwargs = (
        dict(center_window_size=args.center_window_size)
        if args.command == "embedding"
        else {}
    )
    inference = class_mapping[args.command](
        args.model_path,
        genome_msa,
        args.window_size,
        disable_aux_features=args.disable_aux_features,
        **kwargs,
    )
    pred = run_inference(
        dataset,
        inference,
        per_device_batch_size=args.per_device_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    directory = os.path.dirname(args.output_path)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)
    pred.to_parquet(args.output_path, index=False)
