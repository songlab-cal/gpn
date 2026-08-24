"""Durable, resumable storage shared by maintained inference families.

This module deliberately does not know about models, datasets, or distributed
execution. The inference runner is responsible for making the same batch
decision on every process and for allowing only its main process to call the
filesystem-mutating methods on :class:`CheckpointStore`.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CHECKPOINT_FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_METADATA_KEY = b"gpn.checkpoint"
FINAL_METADATA_KEY = b"gpn.inference"


def _default_file_mode() -> int:
    current_umask = os.umask(0)
    os.umask(current_umask)
    return 0o666 & ~current_umask


DEFAULT_FILE_MODE = _default_file_mode()


class CheckpointError(RuntimeError):
    """Base class for checkpoint persistence errors."""


class IncompatibleCheckpointError(CheckpointError):
    """Raised when a checkpoint directory belongs to a different run."""


class InvalidCheckpointError(CheckpointError):
    """Raised when checkpoint contents are missing, corrupt, or inconsistent."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Checkpoint run_signature must contain only JSON-compatible values"
        ) from error


def _normalized_json_value(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory sync after an atomic replacement."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some filesystems do not support syncing directories.
        pass
    finally:
        os.close(descriptor)


def _temporary_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def _replacement_file_mode(target: Path) -> int:
    try:
        return stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        return DEFAULT_FILE_MODE


def _atomic_write_json(value: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        os.chmod(temporary, _replacement_file_mode(target))
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_table(table: pa.Table, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        os.chmod(temporary, _replacement_file_mode(target))
        pq.write_table(table, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_dataframe_atomic(
    frame: pd.DataFrame,
    output_path: os.PathLike[str] | str,
) -> Path:
    """Write a DataFrame atomically with normal creation permissions."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Output must be a pandas DataFrame")
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
    except (pa.ArrowException, TypeError, ValueError) as error:
        raise InvalidCheckpointError(
            f"Could not convert output to Arrow: {error}"
        ) from error
    output = Path(output_path)
    _atomic_write_table(table, output)
    return output


def _schema_without_metadata(schema: pa.Schema) -> pa.Schema:
    return schema.remove_metadata()


def _schemas_equal(left: pa.Schema, right: pa.Schema) -> bool:
    return _schema_without_metadata(left).equals(
        _schema_without_metadata(right),
        check_metadata=False,
    )


def _dict_differences(left: Any, right: Any, prefix: str = "") -> Sequence[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left:
                differences.append(f"{path}: missing from existing manifest")
            elif key not in right:
                differences.append(f"{path}: unexpected in existing manifest")
            else:
                differences.extend(_dict_differences(left[key], right[key], path))
        return differences
    if left != right:
        return [f"{prefix}: existing={left!r}, requested={right!r}"]
    return []


@dataclass(frozen=True)
class BatchRange:
    """A half-open row range assigned to one checkpoint file."""

    index: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not _is_int(self.index) or self.index < 0:
            raise ValueError("Batch index must be a non-negative integer")
        if not _is_int(self.start) or self.start < 0:
            raise ValueError("Batch start must be a non-negative integer")
        if not _is_int(self.stop) or self.stop <= self.start:
            raise ValueError("Batch stop must be greater than batch start")

    @property
    def num_rows(self) -> int:
        return self.stop - self.start

    @property
    def filename(self) -> str:
        return f"batch_{self.index:08d}.parquet"


def expected_batch_ranges(
    total_rows: int,
    batch_size: int,
) -> tuple[BatchRange, ...]:
    """Return stable, ordered ranges without depending on process count."""

    if not _is_int(total_rows) or total_rows <= 0:
        raise ValueError("total_rows must be a positive integer")
    if not _is_int(batch_size) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    return tuple(
        BatchRange(
            index=index,
            start=start,
            stop=min(start + batch_size, total_rows),
        )
        for index, start in enumerate(range(0, total_rows, batch_size))
    )


@dataclass(frozen=True)
class CheckpointManifest:
    """Semantic identity and row partitioning for a checkpoint directory."""

    run_signature: Mapping[str, Any]
    total_rows: int
    batch_size: int
    format_version: int = CHECKPOINT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.run_signature, Mapping):
            raise TypeError("run_signature must be a mapping")
        if not _is_int(self.total_rows) or self.total_rows <= 0:
            raise ValueError("total_rows must be a positive integer")
        if not _is_int(self.batch_size) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not _is_int(self.format_version) or self.format_version <= 0:
            raise ValueError("format_version must be a positive integer")
        if self.format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"format_version must be {CHECKPOINT_FORMAT_VERSION}")

        normalized_signature = _normalized_json_value(dict(self.run_signature))
        if not isinstance(normalized_signature, dict):
            raise TypeError("run_signature must serialize to a JSON object")
        object.__setattr__(self, "run_signature", normalized_signature)

    @cached_property
    def batches(self) -> tuple[BatchRange, ...]:
        return expected_batch_ranges(self.total_rows, self.batch_size)

    @cached_property
    def payload(self) -> Mapping[str, Any]:
        return {
            "batch_size": self.batch_size,
            "format_version": self.format_version,
            "num_batches": len(self.batches),
            "run_signature": self.run_signature,
            "total_rows": self.total_rows,
        }

    @cached_property
    def digest(self) -> str:
        return _sha256_json(self.payload)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            **self.payload,
            "manifest_digest": self.digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointManifest":
        if not isinstance(value, Mapping):
            raise InvalidCheckpointError("Checkpoint manifest must be a JSON object")

        required_keys = {
            "batch_size",
            "format_version",
            "manifest_digest",
            "num_batches",
            "run_signature",
            "total_rows",
        }
        if set(value) != required_keys:
            missing = sorted(required_keys - set(value))
            extra = sorted(set(value) - required_keys)
            raise InvalidCheckpointError(
                "Checkpoint manifest has invalid fields; "
                f"missing={missing}, extra={extra}"
            )

        if value["format_version"] != CHECKPOINT_FORMAT_VERSION:
            raise IncompatibleCheckpointError(
                "Unsupported checkpoint format version "
                f"{value['format_version']!r}; expected {CHECKPOINT_FORMAT_VERSION}"
            )

        try:
            manifest = cls(
                run_signature=value["run_signature"],
                total_rows=value["total_rows"],
                batch_size=value["batch_size"],
                format_version=value["format_version"],
            )
        except (TypeError, ValueError) as error:
            raise InvalidCheckpointError(
                f"Checkpoint manifest contains invalid values: {error}"
            ) from error

        if value["num_batches"] != len(manifest.batches):
            raise InvalidCheckpointError(
                "Checkpoint manifest num_batches does not match its row partitioning"
            )
        if value["manifest_digest"] != manifest.digest:
            raise InvalidCheckpointError("Checkpoint manifest digest is invalid")
        return manifest

    @classmethod
    def read(
        cls,
        path: os.PathLike[str] | str,
    ) -> CheckpointManifest:
        try:
            with Path(path).open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise InvalidCheckpointError(
                f"Could not read checkpoint manifest {path}: {error}"
            ) from error
        return cls.from_dict(value)

    def assert_compatible(self, requested: "CheckpointManifest") -> None:
        differences = _dict_differences(self.payload, requested.payload)
        if differences:
            detail = "\n".join(f"- {difference}" for difference in differences[:20])
            if len(differences) > 20:
                detail += f"\n- ... and {len(differences) - 20} more differences"
            raise IncompatibleCheckpointError(
                "Checkpoint manifest is incompatible with this inference run:\n"
                f"{detail}"
            )


class CheckpointStore:
    """Manage one manifest and its atomically committed Parquet batches."""

    def __init__(
        self,
        directory: os.PathLike[str] | str,
        manifest: CheckpointManifest,
    ) -> None:
        self.directory = Path(directory)
        self.manifest = manifest
        self._known_schema: pa.Schema | None = None

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_FILENAME

    @property
    def batches(self) -> tuple[BatchRange, ...]:
        return self.manifest.batches

    def batch_path(self, batch: BatchRange | int) -> Path:
        return self.directory / self._expected_batch(batch).filename

    def initialize(self) -> None:
        """Create a new manifest or validate an existing one exactly."""

        if self.directory.exists() and not self.directory.is_dir():
            raise InvalidCheckpointError(
                f"Checkpoint path is not a directory: {self.directory}"
            )
        self.directory.mkdir(parents=True, exist_ok=True)

        if self.manifest_path.exists():
            existing = CheckpointManifest.read(self.manifest_path)
            existing.assert_compatible(self.manifest)
            return

        contents = list(self.directory.iterdir())
        if contents:
            raise IncompatibleCheckpointError(
                "Refusing to reuse a non-empty checkpoint directory without "
                f"{MANIFEST_FILENAME}: {self.directory}"
            )
        _atomic_write_json(self.manifest.to_dict(), self.manifest_path)

    def completed_batch_indices(self) -> tuple[int, ...]:
        """Validate present batches and return their indices in row order."""

        self._require_compatible_manifest()
        expected_names = {batch.filename for batch in self.batches}
        unexpected = sorted(
            path.name
            for path in self.directory.glob("batch_*.parquet")
            if path.name not in expected_names
        )
        if unexpected:
            raise InvalidCheckpointError(
                f"Unexpected checkpoint batch files: {unexpected}"
            )

        completed = []
        expected_schema = None
        for batch in self.batches:
            schema = self.validate_batch(batch)
            if schema is None:
                continue
            if expected_schema is not None and not _schemas_equal(
                schema, expected_schema
            ):
                raise InvalidCheckpointError(
                    f"Checkpoint batch {batch.index} has a different schema"
                )
            expected_schema = schema
            completed.append(batch.index)
        self._known_schema = expected_schema
        return tuple(completed)

    def validate_batch(
        self,
        batch: BatchRange | int,
        expected_schema: pa.Schema | None = None,
    ) -> pa.Schema | None:
        """Return a committed batch's data schema, or ``None`` if absent."""

        expected_batch = self._expected_batch(batch)
        path = self.directory / expected_batch.filename
        if not path.exists():
            return None
        if not path.is_file():
            raise InvalidCheckpointError(f"Checkpoint batch is not a file: {path}")

        try:
            parquet_file = pq.ParquetFile(path)
            schema = parquet_file.schema_arrow
            row_count = parquet_file.metadata.num_rows
        except (OSError, pa.ArrowException) as error:
            raise InvalidCheckpointError(
                f"Could not read checkpoint batch {path}: {error}"
            ) from error

        if row_count != expected_batch.num_rows:
            raise InvalidCheckpointError(
                f"Checkpoint batch {expected_batch.index} has {row_count} rows; "
                f"expected {expected_batch.num_rows}"
            )

        metadata = schema.metadata or {}
        raw_metadata = metadata.get(CHECKPOINT_METADATA_KEY)
        if raw_metadata is None:
            raise InvalidCheckpointError(
                f"Checkpoint batch {expected_batch.index} lacks GPN metadata"
            )
        try:
            checkpoint_metadata = json.loads(raw_metadata.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InvalidCheckpointError(
                f"Checkpoint batch {expected_batch.index} has invalid GPN metadata"
            ) from error

        expected_metadata = self._batch_metadata(expected_batch)
        if checkpoint_metadata != expected_metadata:
            differences = _dict_differences(checkpoint_metadata, expected_metadata)
            detail = "; ".join(differences)
            raise InvalidCheckpointError(
                f"Checkpoint batch {expected_batch.index} metadata is incompatible: "
                f"{detail}"
            )

        data_schema = _schema_without_metadata(schema)
        if expected_schema is not None and not _schemas_equal(
            data_schema, expected_schema
        ):
            raise InvalidCheckpointError(
                f"Checkpoint batch {expected_batch.index} has a different schema"
            )
        return data_schema

    def write_batch(
        self,
        batch: BatchRange | int,
        frame: pd.DataFrame,
    ) -> Path:
        """Atomically commit one prediction frame for its expected row range."""

        self._require_compatible_manifest()
        expected_batch = self._expected_batch(batch)
        path = self.directory / expected_batch.filename
        if path.exists():
            self.validate_batch(expected_batch)
            raise FileExistsError(f"Checkpoint batch is already committed: {path}")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Checkpoint predictions must be a pandas DataFrame")
        if len(frame) != expected_batch.num_rows:
            raise ValueError(
                f"Prediction frame has {len(frame)} rows; "
                f"batch {expected_batch.index} expects {expected_batch.num_rows}"
            )

        try:
            table = pa.Table.from_pandas(frame, preserve_index=False)
        except (pa.ArrowException, TypeError, ValueError) as error:
            raise InvalidCheckpointError(
                f"Could not convert batch {expected_batch.index} to Arrow: {error}"
            ) from error

        completed_schema = self._known_or_first_committed_schema(
            excluding=expected_batch
        )
        if completed_schema is not None and not _schemas_equal(
            table.schema, completed_schema
        ):
            raise InvalidCheckpointError(
                f"Prediction frame for batch {expected_batch.index} has a "
                "different schema from existing checkpoints"
            )

        metadata = dict(table.schema.metadata or {})
        metadata[CHECKPOINT_METADATA_KEY] = _canonical_json(
            self._batch_metadata(expected_batch)
        ).encode("utf-8")
        table = table.replace_schema_metadata(metadata)
        _atomic_write_table(table, path)
        committed_schema = self.validate_batch(expected_batch)
        if self._known_schema is None:
            self._known_schema = committed_schema
        return path

    def combine_to(self, output_path: os.PathLike[str] | str) -> Path:
        """Stream all batches in numeric order into one atomic final output."""

        self._require_compatible_manifest()
        output = Path(output_path)
        self._validate_output_location(output)

        schema = self._validate_complete_set()
        final_metadata = {
            FINAL_METADATA_KEY: _canonical_json(
                {
                    "manifest_digest": self.manifest.digest,
                    "total_rows": self.manifest.total_rows,
                }
            ).encode("utf-8")
        }
        final_schema = schema.with_metadata(final_metadata)

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_path(output)
        try:
            os.chmod(temporary, _replacement_file_mode(output))
            with pq.ParquetWriter(temporary, final_schema) as writer:
                for batch in self.batches:
                    try:
                        table = pq.read_table(self.batch_path(batch))
                    except (OSError, pa.ArrowException) as error:
                        raise InvalidCheckpointError(
                            f"Could not read checkpoint batch {batch.index}: {error}"
                        ) from error
                    if not _schemas_equal(table.schema, schema):
                        raise InvalidCheckpointError(
                            f"Checkpoint batch {batch.index} has a different schema"
                        )
                    writer.write_table(table.replace_schema_metadata(final_metadata))

            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            self._validate_final_file(temporary, schema)
            os.replace(temporary, output)
            _fsync_directory(output.parent)
        finally:
            temporary.unlink(missing_ok=True)

        self.validate_final(output)
        return output

    def validate_final(
        self,
        output_path: os.PathLike[str] | str,
    ) -> pa.Schema:
        """Validate that a final file was assembled from this exact manifest."""

        self._require_compatible_manifest()
        schema = self._validate_complete_set()
        output = Path(output_path)
        if not output.is_file():
            raise InvalidCheckpointError(f"Final output does not exist: {output}")
        self._validate_final_file(output, schema)
        return schema

    def cleanup(
        self,
        final_output_path: os.PathLike[str] | str,
    ) -> bool:
        """Remove only managed files after validating the committed final output.

        Returns ``True`` when the checkpoint directory was removed. Unknown
        files are preserved; in that case the compatible manifest is also kept
        and this method returns ``False``.
        """

        self.validate_final(final_output_path)

        known_batch_paths = {self.batch_path(batch) for batch in self.batches}
        known_paths = known_batch_paths | {self.manifest_path}
        unknown_paths = [
            path for path in self.directory.iterdir() if path not in known_paths
        ]

        for path in known_batch_paths:
            path.unlink(missing_ok=True)

        if unknown_paths:
            return False

        self.manifest_path.unlink(missing_ok=True)
        try:
            self.directory.rmdir()
        except OSError:
            return False
        return True

    def _expected_batch(self, batch: BatchRange | int) -> BatchRange:
        if isinstance(batch, bool):
            raise TypeError("Batch must be a BatchRange or integer index")
        if isinstance(batch, int):
            if batch < 0:
                raise IndexError(f"Batch index out of range: {batch}")
            try:
                return self.batches[batch]
            except IndexError as error:
                raise IndexError(f"Batch index out of range: {batch}") from error
        if not isinstance(batch, BatchRange):
            raise TypeError("Batch must be a BatchRange or integer index")
        try:
            expected = self.batches[batch.index]
        except IndexError as error:
            raise IndexError(f"Batch index out of range: {batch.index}") from error
        if batch != expected:
            raise ValueError(
                f"Batch range {batch} does not match expected range {expected}"
            )
        return expected

    def _batch_metadata(self, batch: BatchRange) -> Mapping[str, Any]:
        return {
            "batch_index": batch.index,
            "manifest_digest": self.manifest.digest,
            "start": batch.start,
            "stop": batch.stop,
        }

    def _known_or_first_committed_schema(
        self,
        excluding: BatchRange,
    ) -> pa.Schema | None:
        if self._known_schema is not None:
            return self._known_schema
        for batch in self.batches:
            if batch == excluding or not self.batch_path(batch).exists():
                continue
            self._known_schema = self.validate_batch(batch)
            return self._known_schema
        return None

    def _require_compatible_manifest(self) -> None:
        if not self.manifest_path.is_file():
            raise InvalidCheckpointError(
                f"Checkpoint manifest does not exist: {self.manifest_path}"
            )
        existing = CheckpointManifest.read(self.manifest_path)
        existing.assert_compatible(self.manifest)

    def _validate_complete_set(self) -> pa.Schema:
        completed = self.completed_batch_indices()
        expected = tuple(batch.index for batch in self.batches)
        if completed != expected:
            missing = sorted(set(expected) - set(completed))
            raise InvalidCheckpointError(
                f"Cannot assemble final output; missing checkpoint batches: {missing}"
            )

        schema = self.validate_batch(self.batches[0])
        if schema is None:
            raise InvalidCheckpointError("No checkpoint schema is available")
        for batch in self.batches[1:]:
            self.validate_batch(batch, expected_schema=schema)
        return schema

    def _validate_final_file(self, path: Path, expected_schema: pa.Schema) -> None:
        try:
            parquet_file = pq.ParquetFile(path)
            schema = parquet_file.schema_arrow
            row_count = parquet_file.metadata.num_rows
        except (OSError, pa.ArrowException) as error:
            raise InvalidCheckpointError(
                f"Could not read final output {path}: {error}"
            ) from error

        if row_count != self.manifest.total_rows:
            raise InvalidCheckpointError(
                f"Final output has {row_count} rows; "
                f"expected {self.manifest.total_rows}"
            )
        if not _schemas_equal(schema, expected_schema):
            raise InvalidCheckpointError("Final output has an incompatible schema")

        metadata = schema.metadata or {}
        raw_metadata = metadata.get(FINAL_METADATA_KEY)
        if raw_metadata is None:
            raise InvalidCheckpointError("Final output lacks GPN provenance metadata")
        try:
            final_metadata = json.loads(raw_metadata.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InvalidCheckpointError(
                "Final output has invalid GPN provenance metadata"
            ) from error
        expected_metadata = {
            "manifest_digest": self.manifest.digest,
            "total_rows": self.manifest.total_rows,
        }
        if final_metadata != expected_metadata:
            raise InvalidCheckpointError(
                "Final output belongs to a different checkpoint manifest"
            )

    def _validate_output_location(self, output: Path) -> None:
        checkpoint_directory = self.directory.resolve()
        resolved_output = output.resolve()
        if resolved_output == checkpoint_directory or checkpoint_directory in (
            resolved_output,
            *resolved_output.parents,
        ):
            raise ValueError("Final output must be outside the checkpoint directory")
