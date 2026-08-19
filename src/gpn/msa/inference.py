import argparse
import importlib
import tempfile

import torch
from datasets import disable_caching
from transformers import Trainer, TrainingArguments

from gpn.data import GenomeMSA, load_dataset_from_file_or_dir
from gpn.star.checkpoint import write_dataframe_atomic

class_mapping = {
    "vep": "gpn.msa.vep:VEPInference",
    "logits": "gpn.msa.logits:LogitsInference",
    "embedding": "gpn.msa.embedding:EmbeddingInference",
    "vep_embedding": "gpn.msa.vep_embedding:VEPEmbeddingInference",
    "vep_influence": "gpn.msa.vep_influence:VEPInfluenceInference",
    "vep_ref_embed": "gpn.msa.vep_ref_embed:VEPRefEmbedInference",
    "vep_delta_embed": "gpn.msa.vep_delta_embed:VEPDeltaEmbedInference",
    "vep_euclidean_dist": "gpn.msa.vep_euclidean_dist:VEPEuclideanDistInference",
    "vep_embeddings": "gpn.msa.vep_embeddings:VEPEmbeddingsInference",
}


def _load_inference_class(command):
    module_name, class_name = class_mapping[command].split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), class_name)


def run_inference(
    dataset,
    inference,
    per_device_batch_size=8,
    dataloader_num_workers=0,
    fp16=None,
    torch_compile=None,
):
    dataset.set_transform(inference.tokenize_function)
    temporary_output_dir = tempfile.TemporaryDirectory(prefix="gpn-msa-inference-")
    use_cuda = torch.cuda.is_available()
    compile_enabled = use_cuda if torch_compile is None else torch_compile
    previous_suppress_errors = torch._dynamo.config.suppress_errors
    if compile_enabled:
        # Deprecated GPN-MSA models historically fell back to eager execution
        # when a custom operation could not be compiled.
        torch._dynamo.config.suppress_errors = True
    try:
        training_args = TrainingArguments(
            output_dir=temporary_output_dir.name,
            per_device_eval_batch_size=per_device_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            remove_unused_columns=False,
            torch_compile=compile_enabled,
            fp16=use_cuda if fp16 is None else fp16,
            report_to="none",
        )
        trainer = Trainer(model=inference.model, args=training_args)
        trainer._gpn_temporary_output_dir = temporary_output_dir
        pred = trainer.predict(test_dataset=dataset).predictions
    finally:
        if compile_enabled:
            torch._dynamo.config.suppress_errors = previous_suppress_errors
    if not trainer.accelerator.is_main_process:
        return None
    return inference.postprocess(pred)


def _build_parser(command=None):
    parser = argparse.ArgumentParser(
        description="Run inference with the deprecated GPN-MSA model family",
    )
    if command is None:
        parser.add_argument(
            "command",
            type=str,
            help="""Command to run:
            - vep: zero-shot variant effect prediction (LLR)
            - logits: masked language model logits
            - embedding: averaged embedding from last layer
            """,
            choices=class_mapping.keys(),
        )
    else:
        if command not in class_mapping:
            raise ValueError(f"Unknown GPN-MSA inference command: {command}")
        parser.set_defaults(command=command)
    parser.add_argument(
        "input_path",
        type=str,
        help="""Input path, either HF dataset, parquet, csv/tsv, vcf, with columns:
        - vep: chrom, one-based pos, canonical ref, canonical alt
        - logits: chrom, one-based pos
        - embedding: chrom, zero-based half-open start, end
        """,
    )
    parser.add_argument(
        "msa_path",
        type=str,
        help="Local GPN-MSA Zarr path compatible with the selected model",
    )
    parser.add_argument("window_size", type=int, help="Genomic window size")
    parser.add_argument("model_path", help="Model path (local or on HF hub)", type=str)
    parser.add_argument("output_path", help="Output path (parquet)", type=str)
    parser.add_argument(
        "--per-device-batch-size",
        "--per_device_batch_size",
        dest="per_device_batch_size",
        help="Per device batch size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--dataloader-num-workers",
        "--dataloader_num_workers",
        dest="dataloader_num_workers",
        type=int,
        default=0,
        help="Dataloader num workers",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--is-file",
        "--is_file",
        dest="is_file",
        action="store_true",
        help="VARIANTS_PATH is a file, not directory",
    )
    parser.add_argument(
        "--disable-aux-features",
        "--disable_aux_features",
        dest="disable_aux_features",
        action="store_true",
    )
    parser.add_argument(
        "--center-window-size",
        "--center_window_size",
        dest="center_window_size",
        type=int,
        help="[embedding] Genomic window size to average at the center of the windows",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use FP16 inference (default: enabled when CUDA is available)",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use torch.compile (default: enabled when CUDA is available)",
    )
    return parser


def _validate_cli_args(parser, args):
    if args.window_size <= 0:
        parser.error("window_size must be positive")
    if args.per_device_batch_size <= 0:
        parser.error("--per-device-batch-size must be positive")
    if args.dataloader_num_workers < 0:
        parser.error("--dataloader-num-workers must be non-negative")
    if args.center_window_size is not None and args.center_window_size <= 0:
        parser.error("--center-window-size must be positive")
    if args.command != "embedding" and args.window_size % 2:
        parser.error("window_size must be even for centered MSA inference")


def main(argv=None, *, command=None):
    parser = _build_parser(command=command)
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)
    disable_caching()

    dataset = load_dataset_from_file_or_dir(
        args.input_path,
        split=args.split,
        is_file=args.is_file,
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
    inference_class = _load_inference_class(args.command)
    inference = inference_class(
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
        fp16=args.fp16,
        torch_compile=args.torch_compile,
    )
    if pred is not None:
        write_dataframe_atomic(pred, args.output_path)


if __name__ == "__main__":
    main()
