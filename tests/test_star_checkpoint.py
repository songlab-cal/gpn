import json
from pathlib import Path
import stat

import pandas as pd
import pyarrow.parquet as pq
import pytest

import gpn.star.checkpoint as checkpoint
from gpn.star.checkpoint import (
    BatchRange,
    CheckpointManifest,
    CheckpointStore,
    IncompatibleCheckpointError,
    InvalidCheckpointError,
    expected_batch_ranges,
)


def make_manifest(**overrides):
    values = {
        "run_signature": {
            "command": "logits",
            "dataset": {
                "columns": ["chrom", "pos"],
                "fingerprint": "dataset-v1",
            },
            "model": "model-v1",
            "window_size": 128,
        },
        "total_rows": 7,
        "batch_size": 3,
    }
    values.update(overrides)
    return CheckpointManifest(**values)


def make_store(tmp_path, **manifest_overrides):
    manifest = make_manifest(**manifest_overrides)
    store = CheckpointStore(tmp_path / "checkpoints", manifest)
    store.initialize()
    return store


def frame_for(batch, value_column="prediction"):
    return pd.DataFrame(
        {
            "row_id": list(range(batch.start, batch.stop)),
            value_column: [
                float(index) + 0.5 for index in range(batch.start, batch.stop)
            ],
        },
        index=range(100 + batch.start, 100 + batch.stop),
    )


def populate(store, order=None):
    order = order if order is not None else range(len(store.batches))
    for index in order:
        batch = store.batches[index]
        store.write_batch(batch, frame_for(batch))


@pytest.mark.parametrize(
    ("total_rows", "batch_size"),
    [
        (0, 1),
        (-1, 1),
        (True, 1),
        (1, 0),
        (1, -1),
        (1, True),
    ],
)
def test_expected_batch_ranges_reject_invalid_sizes(total_rows, batch_size):
    with pytest.raises(ValueError):
        expected_batch_ranges(total_rows, batch_size)


def test_expected_batch_ranges_are_stable_and_ordered():
    assert expected_batch_ranges(7, 3) == (
        BatchRange(index=0, start=0, stop=3),
        BatchRange(index=1, start=3, stop=6),
        BatchRange(index=2, start=6, stop=7),
    )


def test_manifest_normalizes_json_and_has_stable_digest():
    first = CheckpointManifest(
        run_signature={"options": ("a", "b"), "command": "logits"},
        total_rows=4,
        batch_size=2,
    )
    second = CheckpointManifest(
        run_signature={"command": "logits", "options": ["a", "b"]},
        total_rows=4,
        batch_size=2,
    )

    assert first.run_signature["options"] == ["a", "b"]
    assert first.digest == second.digest
    assert CheckpointManifest.from_dict(first.to_dict()) == first


def test_manifest_rejects_non_json_signature():
    with pytest.raises(TypeError, match="JSON-compatible"):
        CheckpointManifest(
            run_signature={"not_json": object()},
            total_rows=4,
            batch_size=2,
        )


def test_initialize_is_idempotent_for_compatible_manifest(tmp_path):
    manifest = make_manifest()
    first = CheckpointStore(tmp_path / "checkpoints", manifest)
    second = CheckpointStore(tmp_path / "checkpoints", manifest)

    first.initialize()
    second.initialize()

    assert CheckpointManifest.read(first.manifest_path) == manifest


@pytest.mark.parametrize(
    "overrides",
    [
        {"run_signature": {"command": "vep"}},
        {"total_rows": 8},
        {"batch_size": 2},
    ],
)
def test_initialize_rejects_incompatible_manifest(tmp_path, overrides):
    original = make_store(tmp_path)
    requested = make_manifest(**overrides)

    with pytest.raises(IncompatibleCheckpointError, match="incompatible"):
        CheckpointStore(original.directory, requested).initialize()


def test_initialize_rejects_old_nonempty_directory_without_manifest(tmp_path):
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    pd.DataFrame({"x": [1]}).to_parquet(directory / "batch_00000000.parquet")

    with pytest.raises(IncompatibleCheckpointError, match="without manifest.json"):
        CheckpointStore(directory, make_manifest()).initialize()


def test_tampered_manifest_is_rejected(tmp_path):
    store = make_store(tmp_path)
    value = json.loads(store.manifest_path.read_text())
    value["total_rows"] = 8
    store.manifest_path.write_text(json.dumps(value))

    with pytest.raises(InvalidCheckpointError, match="digest"):
        store.completed_batch_indices()


def test_manifest_write_is_atomic(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path / "checkpoints", make_manifest())
    real_replace = checkpoint.os.replace

    def fail_manifest_replace(source, destination):
        if Path(destination) == store.manifest_path:
            raise OSError("simulated replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpoint.os, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="simulated"):
        store.initialize()

    assert not store.manifest_path.exists()
    assert list(store.directory.iterdir()) == []


def test_write_and_resume_accept_only_committed_batches(tmp_path):
    store = make_store(tmp_path)
    store.write_batch(0, frame_for(store.batches[0]))
    store.write_batch(2, frame_for(store.batches[2]))

    resumed = CheckpointStore(store.directory, store.manifest)
    resumed.initialize()

    assert resumed.completed_batch_indices() == (0, 2)
    assert resumed.validate_batch(1) is None


def test_write_batch_does_not_rescan_prior_batches(tmp_path, monkeypatch):
    store = make_store(tmp_path, total_rows=20, batch_size=1)
    assert store.completed_batch_indices() == ()
    calls = []
    original_validate_batch = store.validate_batch

    def record_validation(batch, expected_schema=None):
        calls.append(batch.index if isinstance(batch, BatchRange) else batch)
        return original_validate_batch(batch, expected_schema)

    monkeypatch.setattr(store, "validate_batch", record_validation)

    for batch in store.batches:
        store.write_batch(batch, frame_for(batch))

    assert calls == list(range(len(store.batches)))


def test_write_batch_is_atomic(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    batch = store.batches[0]
    target = store.batch_path(batch)
    real_replace = checkpoint.os.replace

    def fail_batch_replace(source, destination):
        if Path(destination) == target:
            raise OSError("simulated replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpoint.os, "replace", fail_batch_replace)

    with pytest.raises(OSError, match="simulated"):
        store.write_batch(batch, frame_for(batch))

    assert not target.exists()
    assert not list(store.directory.glob(f".{target.name}.*.tmp"))
    assert store.completed_batch_indices() == ()


def test_atomic_parquet_permissions_follow_umask_and_preserve_existing_mode(
    tmp_path,
):
    store = make_store(tmp_path)
    batch = store.batches[0]
    batch_path = store.write_batch(batch, frame_for(batch))

    assert stat.S_IMODE(batch_path.stat().st_mode) == checkpoint.DEFAULT_FILE_MODE

    populate(store, order=[1, 2])
    output = tmp_path / "output.parquet"
    output.write_bytes(b"old output")
    output.chmod(0o640)
    store.combine_to(output)

    assert stat.S_IMODE(output.stat().st_mode) == 0o640


def test_write_batch_validates_type_and_row_count(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(TypeError, match="pandas DataFrame"):
        store.write_batch(0, [{"prediction": 1.0}])
    with pytest.raises(ValueError, match="expects 3"):
        store.write_batch(0, pd.DataFrame({"prediction": [1.0]}))


def test_write_batch_refuses_to_overwrite_committed_file(tmp_path):
    store = make_store(tmp_path)
    batch = store.batches[0]
    store.write_batch(batch, frame_for(batch))

    with pytest.raises(FileExistsError, match="already committed"):
        store.write_batch(batch, frame_for(batch))


def test_wrong_batch_range_is_rejected(tmp_path):
    store = make_store(tmp_path)
    wrong = BatchRange(index=0, start=0, stop=2)

    with pytest.raises(ValueError, match="does not match"):
        store.write_batch(wrong, pd.DataFrame({"prediction": [1.0, 2.0]}))


def test_negative_batch_index_is_rejected(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(IndexError, match="out of range"):
        store.batch_path(-1)


def test_batch_without_checkpoint_metadata_is_rejected(tmp_path):
    store = make_store(tmp_path)
    batch = store.batches[0]
    pd.DataFrame({"prediction": [1.0, 2.0, 3.0]}).to_parquet(
        store.batch_path(batch),
        index=False,
    )

    with pytest.raises(InvalidCheckpointError, match="lacks GPN metadata"):
        store.validate_batch(batch)


def test_corrupt_batch_is_rejected(tmp_path):
    store = make_store(tmp_path)
    store.batch_path(0).write_bytes(b"not a parquet file")

    with pytest.raises(InvalidCheckpointError, match="Could not read"):
        store.completed_batch_indices()


def test_wrong_batch_row_count_is_rejected(tmp_path):
    store = make_store(tmp_path)
    batch = store.batches[0]
    store.write_batch(batch, frame_for(batch))
    table = pq.read_table(store.batch_path(batch)).slice(0, 2)
    pq.write_table(table, store.batch_path(batch))

    with pytest.raises(InvalidCheckpointError, match="has 2 rows; expected 3"):
        store.validate_batch(batch)


def test_cross_batch_schema_mismatch_is_rejected_before_commit(tmp_path):
    store = make_store(tmp_path)
    store.write_batch(0, frame_for(store.batches[0]))
    second = store.batches[1]
    incompatible = pd.DataFrame(
        {
            "row_id": [str(index) for index in range(second.start, second.stop)],
            "prediction": ["x"] * second.num_rows,
        }
    )

    with pytest.raises(InvalidCheckpointError, match="different schema"):
        store.write_batch(second, incompatible)

    assert not store.batch_path(second).exists()


def test_unexpected_batch_file_is_rejected(tmp_path):
    store = make_store(tmp_path)
    pd.DataFrame({"x": [1]}).to_parquet(
        store.directory / "batch_99999999.parquet",
        index=False,
    )

    with pytest.raises(InvalidCheckpointError, match="Unexpected"):
        store.completed_batch_indices()


def test_combine_requires_every_batch(tmp_path):
    store = make_store(tmp_path)
    store.write_batch(0, frame_for(store.batches[0]))

    with pytest.raises(InvalidCheckpointError, match="missing.*1, 2"):
        store.combine_to(tmp_path / "output.parquet")


def test_combine_uses_numeric_batch_order_and_drops_dataframe_index(tmp_path):
    store = make_store(tmp_path)
    populate(store, order=[2, 0, 1])

    output = store.combine_to(tmp_path / "output.parquet")
    result = pd.read_parquet(output)

    assert result["row_id"].tolist() == list(range(7))
    assert result.columns.tolist() == ["row_id", "prediction"]
    assert store.validate_final(output).names == ["row_id", "prediction"]


def test_final_write_is_atomic_and_preserves_existing_output_on_failure(
    tmp_path,
    monkeypatch,
):
    store = make_store(tmp_path)
    populate(store)
    output = tmp_path / "output.parquet"
    original = b"previous complete output"
    output.write_bytes(original)
    real_replace = checkpoint.os.replace

    def fail_final_replace(source, destination):
        if Path(destination) == output:
            raise OSError("simulated final replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpoint.os, "replace", fail_final_replace)

    with pytest.raises(OSError, match="simulated"):
        store.combine_to(output)

    assert output.read_bytes() == original
    assert store.completed_batch_indices() == (0, 1, 2)
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_final_output_must_be_outside_checkpoint_directory(tmp_path):
    store = make_store(tmp_path)
    populate(store)

    with pytest.raises(ValueError, match="outside"):
        store.combine_to(store.directory / "output.parquet")


def test_validate_final_rejects_unrelated_parquet(tmp_path):
    store = make_store(tmp_path)
    populate(store)
    unrelated = tmp_path / "unrelated.parquet"
    pd.concat([frame_for(batch) for batch in store.batches]).to_parquet(
        unrelated,
        index=False,
    )

    with pytest.raises(InvalidCheckpointError, match="lacks GPN provenance"):
        store.validate_final(unrelated)


def test_cleanup_requires_matching_committed_final_output(tmp_path):
    store = make_store(tmp_path)
    populate(store)
    unrelated = tmp_path / "unrelated.parquet"
    pd.concat([frame_for(batch) for batch in store.batches]).to_parquet(
        unrelated,
        index=False,
    )

    with pytest.raises(InvalidCheckpointError, match="provenance"):
        store.cleanup(unrelated)

    assert store.manifest_path.exists()
    assert store.completed_batch_indices() == (0, 1, 2)


def test_cleanup_runs_after_final_commit_and_removes_managed_directory(tmp_path):
    store = make_store(tmp_path)
    populate(store)
    output = store.combine_to(tmp_path / "output.parquet")

    assert store.cleanup(output) is True
    assert output.is_file()
    assert not store.directory.exists()


def test_cleanup_preserves_unknown_files_and_manifest(tmp_path):
    store = make_store(tmp_path)
    populate(store)
    output = store.combine_to(tmp_path / "output.parquet")
    sentinel = store.directory / "keep-me.txt"
    sentinel.write_text("user data")

    assert store.cleanup(output) is False
    assert sentinel.read_text() == "user data"
    assert store.manifest_path.is_file()
    assert all(not store.batch_path(batch).exists() for batch in store.batches)
