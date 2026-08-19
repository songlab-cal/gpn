import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import torch
from accelerate.utils import broadcast_object_list
from datasets import Dataset, disable_caching
from transformers import Trainer, TrainingArguments

from gpn.star.checkpoint import (
    CheckpointError,
    CheckpointManifest,
    CheckpointStore,
    write_dataframe_atomic,
)
from gpn.star.data import GenomeMSA, load_dataset_from_file_or_dir
from gpn.star.utils import find_directory_sum_paths

class_mapping = {
    "vep": "gpn.star.vep:VEPInference",
    "logits": "gpn.star.logits:LogitsInference",
    "embedding": "gpn.star.embedding:EmbeddingInference",
    "vep_embedding": "gpn.star.vep_embedding:VEPEmbeddingInference",
}


def _resolve_acceleration_option(value):
    return torch.cuda.is_available() if value is None else value


def _load_inference_class(command):
    module_name, class_name = class_mapping[command].split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), class_name)


def _validate_runtime_options(
    dataset,
    per_device_batch_size,
    dataloader_num_workers,
    checkpoint_batch_size=None,
):
    if len(dataset) == 0:
        raise ValueError("Inference dataset must contain at least one row")
    if per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be positive")
    if dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers must be non-negative")
    if checkpoint_batch_size is not None and checkpoint_batch_size <= 0:
        raise ValueError("checkpoint_batch_size must be positive")


def _create_trainer(
    inference,
    per_device_batch_size,
    dataloader_num_workers,
    fp16=None,
    torch_compile=None,
):
    temporary_output_dir = tempfile.TemporaryDirectory(prefix="gpn-star-inference-")
    training_args = TrainingArguments(
        output_dir=temporary_output_dir.name,
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        torch_compile=_resolve_acceleration_option(torch_compile),
        fp16=_resolve_acceleration_option(fp16),
        report_to="none",
    )
    trainer = Trainer(model=inference.model, args=training_args)
    # Keep the directory alive for as long as Trainer may use it.
    trainer._gpn_temporary_output_dir = temporary_output_dir
    return trainer


def _is_main_process(trainer):
    return trainer.accelerator.is_main_process


def _predict_with_trainer(dataset, inference, trainer):
    if _is_main_process(trainer):
        print(dataset)
    predictions = trainer.predict(test_dataset=dataset).predictions
    if not _is_main_process(trainer):
        return None
    return inference.postprocess(predictions)


def run_inference(
    dataset,
    inference,
    per_device_batch_size=8,
    dataloader_num_workers=0,
    fp16=None,
    torch_compile=None,
):
    """Run one inference pass and return predictions on the main process."""

    _validate_runtime_options(
        dataset,
        per_device_batch_size,
        dataloader_num_workers,
    )
    dataset.set_transform(inference.tokenize_function)
    trainer = _create_trainer(
        inference,
        per_device_batch_size,
        dataloader_num_workers,
        fp16=fp16,
        torch_compile=torch_compile,
    )
    return _predict_with_trainer(dataset, inference, trainer)


def _call_on_main_process(trainer, operation, description):
    """Run a filesystem operation on rank zero and share its result or error."""

    is_main_process = _is_main_process(trainer)
    num_processes = trainer.accelerator.num_processes
    if num_processes == 1:
        if not is_main_process:
            raise RuntimeError("Single-process Trainer did not select a main process")
        return operation()

    original_error = None
    payload = None
    if is_main_process:
        try:
            payload = {
                "ok": True,
                "value": operation(),
            }
        except BaseException as error:
            original_error = error
            payload = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }

    shared_payload = [payload]
    broadcast_object_list(shared_payload, from_process=0)
    payload = shared_payload[0]
    if not payload["ok"]:
        if original_error is not None:
            raise original_error
        raise CheckpointError(
            f"{description} failed on the main process: "
            f"{payload['error_type']}: {payload['error']}"
        )
    return payload["value"]


def run_inference_with_checkpoints(
    dataset,
    inference,
    output_path,
    checkpoint_dir,
    checkpoint_batch_size,
    run_signature,
    per_device_batch_size=8,
    dataloader_num_workers=0,
    cleanup_checkpoints=False,
    fp16=None,
    torch_compile=None,
):
    """Run resumable inference and atomically assemble the final Parquet file.

    Checkpoint row ranges do not depend on process count, so an interrupted run
    can resume with a different number of GPUs. Only the Trainer main process
    reads or writes checkpoint files. Inputs are assumed to be immutable while
    a checkpoint directory is in use. The automatic Zarr identity validates
    metadata and array directories, but not in-place chunk rewrites; after such
    a rewrite, use a new ``checkpoint_revision`` or checkpoint directory.
    """

    _validate_runtime_options(
        dataset,
        per_device_batch_size,
        dataloader_num_workers,
        checkpoint_batch_size=checkpoint_batch_size,
    )
    manifest = CheckpointManifest(
        run_signature=run_signature,
        total_rows=len(dataset),
        batch_size=checkpoint_batch_size,
    )
    store = CheckpointStore(checkpoint_dir, manifest)

    dataset.set_transform(inference.tokenize_function)
    # Trainer/Accelerate must initialize distributed state before rank checks.
    trainer = _create_trainer(
        inference,
        per_device_batch_size,
        dataloader_num_workers,
        fp16=fp16,
        torch_compile=torch_compile,
    )

    def prepare_checkpoints():
        store.initialize()
        return store.completed_batch_indices()

    completed = set(
        _call_on_main_process(
            trainer,
            prepare_checkpoints,
            "Checkpoint initialization",
        )
    )
    if completed and _is_main_process(trainer):
        print(
            f"Resuming from {len(completed)} of {len(store.batches)} "
            f"completed checkpoint batches in {checkpoint_dir}"
        )

    for batch in store.batches:
        if batch.index in completed:
            if _is_main_process(trainer):
                print(f"Skipping completed checkpoint batch {batch.index}")
            continue

        if _is_main_process(trainer):
            print(
                f"Processing checkpoint batch {batch.index}: "
                f"rows {batch.start}:{batch.stop}"
            )
        batch_dataset = dataset.select(range(batch.start, batch.stop))
        predictions = trainer.predict(test_dataset=batch_dataset).predictions

        def commit_batch():
            prediction_frame = inference.postprocess(predictions)
            store.write_batch(batch, prediction_frame)

        _call_on_main_process(
            trainer,
            commit_batch,
            f"Writing checkpoint batch {batch.index}",
        )

    def finalize_output():
        store.combine_to(output_path)
        removed_checkpoint_dir = None
        if cleanup_checkpoints:
            removed_checkpoint_dir = store.cleanup(output_path)
        return removed_checkpoint_dir

    removed_checkpoint_dir = _call_on_main_process(
        trainer,
        finalize_output,
        "Final output assembly",
    )
    if _is_main_process(trainer):
        print(f"Wrote predictions to {output_path}")
        if cleanup_checkpoints:
            if removed_checkpoint_dir:
                print(f"Cleaned up checkpoint directory: {checkpoint_dir}")
            else:
                print(
                    "Removed managed checkpoint files but preserved the "
                    f"non-empty checkpoint directory: {checkpoint_dir}"
                )
        return Path(output_path)
    return None


def _resource_identity(value):
    """Return a JSON identity for a local resource or remote identifier."""

    value = str(value)
    path = Path(value)
    try:
        exists = path.exists()
    except OSError:
        exists = False
    if not exists:
        return {"identifier": value}

    resolved = path.resolve()
    stat = resolved.stat()
    identity = {
        "path": str(resolved),
        "kind": "directory" if resolved.is_dir() else "file",
        "ctime_ns": stat.st_ctime_ns,
        "mtime_ns": stat.st_mtime_ns,
    }
    if resolved.is_file():
        identity["size"] = stat.st_size
        if stat.st_size <= 16 * 1024 * 1024:
            identity["sha256"] = _file_digest(resolved)
    else:
        identity["tree"] = _directory_tree_identity(resolved)
    return identity


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _should_hash_directory_file(path):
    return path.name in {
        ".zarray",
        ".zattrs",
        ".zgroup",
        ".zmetadata",
    } or path.suffix.lower() in {".json", ".yaml", ".yml"}


def _directory_tree_identity(root):
    """Fingerprint file metadata and small control-file contents below root.

    Reading every byte of a model checkpoint or Zarr MSA would make startup
    impractical. Size, mtime, and ctime cover ordinary data replacement, while
    model/Zarr metadata files are content-hashed as well.
    """

    if (root / ".zgroup").is_file():
        return _zarr_tree_identity(root)

    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for directory, directory_names, file_names in os.walk(root):
        directory_names.sort()
        file_names.sort()
        directory_path = Path(directory)
        for file_name in file_names:
            path = directory_path / file_name
            relative_path = path.relative_to(root).as_posix()
            stat = path.lstat()
            entry = {
                "path": relative_path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
            if path.is_symlink():
                entry["symlink"] = os.readlink(path)
            elif path.is_file() and _should_hash_directory_file(path):
                entry["sha256"] = _file_digest(path)
            digest.update(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
            digest.update(b"\n")
            file_count += 1
            total_size += stat.st_size
    return {
        "strategy": "recursive-file-metadata-v1",
        "file_count": file_count,
        "total_size": total_size,
        "metadata_sha256": digest.hexdigest(),
    }


def _zarr_tree_identity(root):
    """Fingerprint Zarr metadata without traversing millions of chunk files."""

    digest = hashlib.sha256()
    array_count = 0
    metadata_file_count = 0
    metadata_size = 0
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        stat = path.lstat()
        entry = {
            "path": path.name,
            "kind": "directory" if path.is_dir() else "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
        if path.is_symlink():
            entry["symlink"] = os.readlink(path)
        elif path.is_file() and _should_hash_directory_file(path):
            entry["sha256"] = _file_digest(path)
            metadata_file_count += 1
            metadata_size += stat.st_size
        elif path.is_dir():
            array_count += 1
            for metadata_name in [".zarray", ".zattrs", ".zgroup"]:
                metadata_path = path / metadata_name
                if not metadata_path.is_file():
                    continue
                metadata_stat = metadata_path.stat()
                entry[metadata_name] = {
                    "size": metadata_stat.st_size,
                    "mtime_ns": metadata_stat.st_mtime_ns,
                    "ctime_ns": metadata_stat.st_ctime_ns,
                    "sha256": _file_digest(metadata_path),
                }
                metadata_file_count += 1
                metadata_size += metadata_stat.st_size
        digest.update(
            json.dumps(
                entry,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "strategy": "zarr-metadata-and-array-directories-v1",
        "array_count": array_count,
        "metadata_file_count": metadata_file_count,
        "metadata_size": metadata_size,
        "metadata_sha256": digest.hexdigest(),
    }


def _source_revision():
    """Hash the checked-out GPN Python sources that can affect inference."""

    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    source_files = sorted(package_root.rglob("*.py"))
    for path in source_files:
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "file_count": len(source_files),
        "sha256": digest.hexdigest(),
    }


def _dependency_versions():
    versions = {}
    for distribution in [
        "accelerate",
        "datasets",
        "numpy",
        "pandas",
        "pyarrow",
        "torch",
        "transformers",
    ]:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _model_config_signature(inference):
    wrapped_model = getattr(inference.model, "model", None)
    config = getattr(wrapped_model, "config", None)
    if config is None:
        return None

    config_payload = config.to_dict()
    serialized = json.dumps(
        config_payload,
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "commit_hash": getattr(config, "_commit_hash", None),
    }


def build_run_signature(args, dataset, msa_paths, inference):
    """Describe every semantic input that can affect checkpoint predictions."""

    wrapped_model = getattr(inference.model, "model", None)
    config = getattr(wrapped_model, "config", None)
    resolved_phylo_dist_path = getattr(config, "phylo_dist_path", None)
    return {
        "checkpoint_revision": getattr(args, "checkpoint_revision", None),
        "command": args.command,
        "dataset": {
            "columns": list(dataset.column_names),
            "fingerprint": getattr(dataset, "_fingerprint", None),
            "input": _resource_identity(args.input_path),
            "is_file": args.is_file,
            "split": args.split,
        },
        "inference": {
            "center_window_size": args.center_window_size,
            "disable_aux_features": args.disable_aux_features,
            "implementation": (
                f"{type(inference).__module__}.{type(inference).__qualname__}"
            ),
            "fp16": _resolve_acceleration_option(getattr(args, "fp16", None)),
            "torch_compile": _resolve_acceleration_option(
                getattr(args, "torch_compile", None)
            ),
            "window_size": args.window_size,
        },
        "model": {
            "config": _model_config_signature(inference),
            "resource": _resource_identity(args.model_path),
        },
        "msa": [
            {
                "order": order,
                "n_species": str(n_species),
                "resource": _resource_identity(path),
            }
            for order, (n_species, path) in enumerate(msa_paths.items())
        ],
        "phylo_dist": (
            _resource_identity(resolved_phylo_dist_path)
            if resolved_phylo_dist_path is not None
            else None
        ),
        "software": {
            "dependencies": _dependency_versions(),
            "gpn_source": _source_revision(),
        },
    }


def _write_parquet_atomic(frame, output_path):
    return write_dataframe_atomic(frame, output_path)


def _build_parser(command=None):
    parser = argparse.ArgumentParser(
        description="Run inference with AutoModelForMaskedLM",
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
            raise ValueError(f"Unknown GPN-Star inference command: {command}")
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
        help=(
            "Local MSA parent: a numeric species-count directory containing "
            "all.zarr, or a root containing such numeric directories"
        ),
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
        help="INPUT_PATH is a file, not a directory or Hub dataset",
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
        "--checkpoint-batch-size",
        "--checkpoint_batch_size",
        dest="checkpoint_batch_size",
        type=int,
        default=None,
        help=(
            "Rows per durable checkpoint batch. Enables resumable inference when set."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        dest="checkpoint_dir",
        type=str,
        default=None,
        help=("Checkpoint directory. Defaults to OUTPUT_PATH + '_checkpoints'."),
    )
    parser.add_argument(
        "--checkpoint-revision",
        "--checkpoint_revision",
        dest="checkpoint_revision",
        type=str,
        default=None,
        help=(
            "Optional immutable data/model revision identifier included in "
            "resume checks. Zarr chunk contents are assumed immutable; use a "
            "new value after any in-place MSA chunk rewrite, which automatic "
            "metadata checks cannot detect."
        ),
    )
    parser.add_argument(
        "--cleanup-checkpoints",
        "--cleanup_checkpoints",
        dest="cleanup_checkpoints",
        action="store_true",
        help="Remove managed checkpoints after the final output is committed",
    )
    parser.add_argument(
        "--phylo-dist-path",
        "--phylo_dist_path",
        dest="phylo_dist_path",
        type=str,
        default=None,
        help=(
            "[logits] Override the phylogenetic-distance directory stored in "
            "the model config"
        ),
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
    if args.checkpoint_batch_size is not None and args.checkpoint_batch_size <= 0:
        parser.error("--checkpoint-batch-size must be positive")
    if args.checkpoint_dir is not None and args.checkpoint_batch_size is None:
        parser.error("--checkpoint-dir requires --checkpoint-batch-size")
    if args.checkpoint_revision is not None and args.checkpoint_batch_size is None:
        parser.error("--checkpoint-revision requires --checkpoint-batch-size")
    if args.cleanup_checkpoints and args.checkpoint_batch_size is None:
        parser.error("--cleanup-checkpoints requires --checkpoint-batch-size")
    if args.phylo_dist_path is not None and args.command != "logits":
        parser.error("--phylo-dist-path is only supported by the logits command")
    if args.center_window_size is not None and args.center_window_size <= 0:
        parser.error("--center-window-size must be positive")
    if args.command != "embedding" and args.window_size % 2:
        parser.error("window_size must be even for centered MSA inference")


def main(argv=None, *, command=None):
    parser = _build_parser(command=command)
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)
    disable_caching()

    try:
        dataset = load_dataset_from_file_or_dir(
            args.input_path,
            split=args.split,
            is_file=args.is_file,
        )
    except Exception:
        dataset = Dataset.from_pandas(
            pd.read_parquet(os.path.join(args.input_path, "test.parquet"))
        )

    _validate_runtime_options(
        dataset,
        args.per_device_batch_size,
        args.dataloader_num_workers,
        checkpoint_batch_size=args.checkpoint_batch_size,
    )

    msa_paths = find_directory_sum_paths(args.msa_path)
    genome_msa_list = [
        GenomeMSA(
            path,
            n_species=n_species,
            subset_chroms=dataset.unique("chrom"),
            in_memory=False,
        )
        for n_species, path in msa_paths.items()
    ]

    # This predates subparsers; keep command-specific kwargs explicit.
    kwargs = {}
    if args.command == "embedding" and args.center_window_size is not None:
        kwargs["center_window_size"] = args.center_window_size
    if args.command == "logits" and args.phylo_dist_path is not None:
        kwargs["phylo_dist_path"] = args.phylo_dist_path

    inference_class = _load_inference_class(args.command)
    inference = inference_class(
        args.model_path,
        genome_msa_list,
        args.window_size,
        disable_aux_features=args.disable_aux_features,
        **kwargs,
    )

    if args.checkpoint_batch_size is not None:
        checkpoint_dir = args.checkpoint_dir or (args.output_path + "_checkpoints")
        run_signature = build_run_signature(
            args,
            dataset,
            msa_paths,
            inference,
        )
        run_inference_with_checkpoints(
            dataset,
            inference,
            output_path=args.output_path,
            checkpoint_dir=checkpoint_dir,
            checkpoint_batch_size=args.checkpoint_batch_size,
            run_signature=run_signature,
            per_device_batch_size=args.per_device_batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
            cleanup_checkpoints=args.cleanup_checkpoints,
            fp16=args.fp16,
            torch_compile=args.torch_compile,
        )
    else:
        predictions = run_inference(
            dataset,
            inference,
            per_device_batch_size=args.per_device_batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
            fp16=args.fp16,
            torch_compile=args.torch_compile,
        )
        if predictions is not None:
            _write_parquet_atomic(predictions, args.output_path)
            print(f"Wrote predictions to {args.output_path}")


if __name__ == "__main__":
    main()
