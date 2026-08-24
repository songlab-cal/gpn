"""Shared execution and resumability for maintained inference families."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from accelerate.utils import broadcast_object_list
from datasets import Dataset
from torch import nn
from transformers import Trainer, TrainingArguments

from gpn.arguments import CheckpointArguments
from gpn.checkpoint import (
    CheckpointError,
    CheckpointManifest,
    CheckpointStore,
    write_dataframe_atomic,
)


class InferenceAdapter(Protocol):
    """The small interface supplied by each model-family operation."""

    model: nn.Module

    def tokenize_function(self, batch: dict[str, list[Any]]) -> dict[str, Any]: ...

    def postprocess(self, predictions: Any) -> pd.DataFrame: ...


@dataclass
class InferenceRunner:
    """A configured Transformers prediction runner."""

    trainer: Trainer
    temporary_output_dir: tempfile.TemporaryDirectory[str] | None = None

    @property
    def is_main_process(self) -> bool:
        return bool(self.trainer.accelerator.is_main_process)

    def predict(self, dataset: Dataset) -> Any | None:
        predictions = self.trainer.predict(test_dataset=dataset).predictions
        return predictions if self.is_main_process else None


def inference_training_arguments(
    arguments: TrainingArguments,
    *,
    temporary_output_dir: str | None = None,
) -> TrainingArguments:
    """Apply GPN prediction invariants without hiding Transformers options."""

    if arguments.push_to_hub:
        raise ValueError("push_to_hub is not supported by inference commands")
    output_dir = arguments.output_dir or temporary_output_dir
    if output_dir is None:
        raise ValueError("An inference output directory could not be determined")
    return replace(
        arguments,
        output_dir=output_dir,
        do_train=False,
        do_eval=False,
        do_predict=True,
        dataloader_drop_last=False,
        dataloader_in_order=True,
        remove_unused_columns=False,
        prediction_loss_only=False,
    )


def prediction_arguments_signature(arguments: TrainingArguments) -> dict[str, Any]:
    """Return the effective, secret-free Trainer prediction configuration.

    The signature is intentionally conservative: every serialized Transformers
    argument is tracked except ephemeral filesystem/process identity. This keeps
    future Trainer flags resume-safe without maintaining an allowlist.
    """

    values = arguments.to_dict()
    values.update(
        {
            "do_train": False,
            "do_eval": False,
            "do_predict": True,
            "dataloader_drop_last": False,
            "dataloader_in_order": True,
            "remove_unused_columns": False,
            "prediction_loss_only": False,
        }
    )
    for field in ("output_dir", "logging_dir", "run_name", "local_rank", "hub_token"):
        values.pop(field, None)
    return values


def create_inference_runner(
    model: nn.Module,
    arguments: TrainingArguments,
    *,
    output_prefix: str,
) -> InferenceRunner:
    """Create a Trainer, using a temporary output directory when none was given."""

    temporary_output_dir = None
    output_dir = arguments.output_dir
    if not output_dir:
        temporary_output_dir = tempfile.TemporaryDirectory(prefix=output_prefix)
        output_dir = temporary_output_dir.name
    prediction_arguments = inference_training_arguments(
        arguments,
        temporary_output_dir=output_dir,
    )
    return InferenceRunner(
        trainer=Trainer(model=model, args=prediction_arguments),
        temporary_output_dir=temporary_output_dir,
    )


def _validate_dataset(dataset: Dataset) -> None:
    if len(dataset) == 0:
        raise ValueError("Inference dataset must contain at least one row")


def run_inference(
    dataset: Dataset,
    inference: InferenceAdapter,
    training_arguments: TrainingArguments,
    *,
    output_prefix: str,
) -> pd.DataFrame | None:
    """Run one prediction pass and return a frame on the main process."""

    _validate_dataset(dataset)
    dataset.set_transform(inference.tokenize_function)
    runner = create_inference_runner(
        inference.model,
        training_arguments,
        output_prefix=output_prefix,
    )
    predictions = runner.predict(dataset)
    if predictions is None:
        return None
    result = inference.postprocess(predictions)
    if len(result) != len(dataset):
        raise ValueError(
            "Inference produced a different number of predictions than input rows: "
            f"{len(result)} predictions for {len(dataset)} rows"
        )
    return result


def _call_on_main_process(
    runner: InferenceRunner,
    operation: Any,
    description: str,
) -> Any:
    """Run a filesystem operation on rank zero and share its result or error."""

    accelerator = runner.trainer.accelerator
    if accelerator.num_processes == 1:
        if not runner.is_main_process:
            raise RuntimeError("Single-process Trainer did not select a main process")
        return operation()

    original_error = None
    payload = None
    if runner.is_main_process:
        try:
            payload = {"ok": True, "value": operation()}
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
    if payload is None:
        raise CheckpointError(f"{description} returned no distributed status")
    if not payload["ok"]:
        if original_error is not None:
            raise original_error
        raise CheckpointError(
            f"{description} failed on the main process: "
            f"{payload['error_type']}: {payload['error']}"
        )
    return payload["value"]


def run_inference_with_checkpoints(
    dataset: Dataset,
    inference: InferenceAdapter,
    training_arguments: TrainingArguments,
    checkpoint_arguments: CheckpointArguments,
    *,
    output_path: Path,
    run_signature: dict[str, Any],
    output_prefix: str,
) -> Path | None:
    """Run resumable inference and atomically assemble the final Parquet file."""

    _validate_dataset(dataset)
    batch_size = checkpoint_arguments.checkpoint_batch_size
    if batch_size is None:
        raise ValueError("checkpoint_batch_size is required for checkpointed inference")
    checkpoint_dir = checkpoint_arguments.checkpoint_dir or Path(
        f"{output_path}_checkpoints"
    )
    manifest = CheckpointManifest(
        run_signature=run_signature,
        total_rows=len(dataset),
        batch_size=batch_size,
    )
    store = CheckpointStore(checkpoint_dir, manifest)

    dataset.set_transform(inference.tokenize_function)
    runner = create_inference_runner(
        inference.model,
        training_arguments,
        output_prefix=output_prefix,
    )

    def prepare_checkpoints() -> tuple[int, ...]:
        store.initialize()
        return store.completed_batch_indices()

    completed = set(
        _call_on_main_process(
            runner,
            prepare_checkpoints,
            "Checkpoint initialization",
        )
    )
    if completed and runner.is_main_process:
        print(
            f"Resuming from {len(completed)} of {len(store.batches)} "
            f"completed checkpoint batches in {checkpoint_dir}"
        )

    for batch in store.batches:
        if batch.index in completed:
            if runner.is_main_process:
                print(f"Skipping completed checkpoint batch {batch.index}")
            continue
        if runner.is_main_process:
            print(
                f"Processing checkpoint batch {batch.index}: "
                f"rows {batch.start}:{batch.stop}"
            )
        batch_dataset = dataset.select(range(batch.start, batch.stop))
        predictions = runner.trainer.predict(test_dataset=batch_dataset).predictions

        def commit_batch() -> None:
            store.write_batch(batch, inference.postprocess(predictions))

        _call_on_main_process(
            runner,
            commit_batch,
            f"Writing checkpoint batch {batch.index}",
        )

    def finalize_output() -> bool | None:
        store.combine_to(output_path)
        if checkpoint_arguments.cleanup_checkpoints:
            return store.cleanup(output_path)
        return None

    removed_checkpoint_dir = _call_on_main_process(
        runner,
        finalize_output,
        "Final output assembly",
    )
    if not runner.is_main_process:
        return None
    print(f"Wrote predictions to {output_path}")
    if checkpoint_arguments.cleanup_checkpoints:
        if removed_checkpoint_dir:
            print(f"Cleaned up checkpoint directory: {checkpoint_dir}")
        else:
            print(
                "Removed managed checkpoint files but preserved the non-empty "
                f"checkpoint directory: {checkpoint_dir}"
            )
    return output_path


def execute_inference(
    dataset: Dataset,
    inference: InferenceAdapter,
    training_arguments: TrainingArguments,
    checkpoint_arguments: CheckpointArguments,
    *,
    output_path: Path,
    run_signature_factory: Callable[[], dict[str, Any]],
    output_prefix: str,
) -> Path | None:
    """Select direct or resumable execution and atomically write the result."""

    if checkpoint_arguments.checkpoint_batch_size is not None:
        return run_inference_with_checkpoints(
            dataset,
            inference,
            training_arguments,
            checkpoint_arguments,
            output_path=output_path,
            run_signature=run_signature_factory(),
            output_prefix=output_prefix,
        )
    predictions = run_inference(
        dataset,
        inference,
        training_arguments,
        output_prefix=output_prefix,
    )
    if predictions is None:
        return None
    write_dataframe_atomic(predictions, output_path)
    print(f"Wrote predictions to {output_path}")
    return output_path


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _should_hash_directory_file(path: Path) -> bool:
    return path.name in {
        ".zarray",
        ".zattrs",
        ".zgroup",
        ".zmetadata",
    } or path.suffix.lower() in {".json", ".yaml", ".yml"}


def _zarr_tree_identity(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    array_count = 0
    metadata_file_count = 0
    metadata_size = 0
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        stat = path.lstat()
        entry: dict[str, Any] = {
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


def _directory_tree_identity(root: Path) -> dict[str, Any]:
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
            stat = path.lstat()
            entry: dict[str, Any] = {
                "path": path.relative_to(root).as_posix(),
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


def resource_identity(value: str | Path) -> dict[str, Any]:
    """Return a bounded JSON identity for a local resource or Hub identifier."""

    text = str(value)
    path = Path(text)
    try:
        exists = path.exists()
    except OSError:
        exists = False
    if not exists:
        return {"identifier": text}

    resolved = path.resolve()
    stat = resolved.stat()
    identity: dict[str, Any] = {
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


def tokenizer_identity(tokenizer: Any, source: str | Path) -> dict[str, Any]:
    """Identify a tokenizer source and its effective serialized state."""

    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    with tempfile.TemporaryDirectory(prefix="gpn-tokenizer-signature-") as directory:
        root = Path(directory)
        tokenizer.save_pretrained(root)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file():
                continue
            relative_path = path.relative_to(root).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    total_size += len(chunk)
            digest.update(b"\0")
            file_count += 1
    return {
        "resource": resource_identity(source),
        "effective": {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "file_count": file_count,
            "size": total_size,
            "sha256": digest.hexdigest(),
        },
    }


def _source_revision() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    source_files = sorted(package_root.rglob("*.py"))
    for path in source_files:
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return {"file_count": len(source_files), "sha256": digest.hexdigest()}


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
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


def _model_config_signature(inference: InferenceAdapter) -> dict[str, Any] | None:
    wrapped_model = getattr(inference.model, "model", None)
    config = getattr(wrapped_model, "config", None)
    if config is None:
        return None
    serialized = json.dumps(
        config.to_dict(),
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "commit_hash": getattr(config, "_commit_hash", None),
    }


def build_run_signature(
    *,
    family: str,
    operation: str,
    dataset: Dataset,
    input_path: str | Path,
    split: str,
    is_file: bool,
    model_path: str | Path,
    inference: InferenceAdapter,
    training_arguments: TrainingArguments,
    checkpoint_revision: str | None,
    resources: dict[str, Any],
    operation_arguments: dict[str, Any],
) -> dict[str, Any]:
    """Describe semantic inputs that can affect checkpoint predictions."""

    return {
        "checkpoint_revision": checkpoint_revision,
        "command": {"family": family, "operation": operation},
        "dataset": {
            "columns": list(dataset.column_names),
            "fingerprint": getattr(dataset, "_fingerprint", None),
            "input": resource_identity(input_path),
            "is_file": is_file,
            "split": split,
        },
        "inference": {
            "arguments": operation_arguments,
            "implementation": (
                f"{type(inference).__module__}.{type(inference).__qualname__}"
            ),
            "runtime": prediction_arguments_signature(training_arguments),
        },
        "model": {
            "config": _model_config_signature(inference),
            "resource": resource_identity(model_path),
        },
        "resources": resources,
        "software": {
            "dependencies": _dependency_versions(),
            "gpn_source": _source_revision(),
        },
    }
