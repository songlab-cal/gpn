from pathlib import Path
import stat
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gpn.star.inference as inference_module
from gpn.star.checkpoint import DEFAULT_FILE_MODE, IncompatibleCheckpointError


class FakeDataset:
    column_names = ["value"]
    _fingerprint = "fake-dataset-v1"

    def __init__(self, values):
        self.values = list(values)
        self.transform = None

    def __len__(self):
        return len(self.values)

    def set_transform(self, transform):
        self.transform = transform

    def select(self, indices):
        selected = FakeDataset([self.values[index] for index in indices])
        selected.transform = self.transform
        return selected


class FakeInference:
    model = object()

    @staticmethod
    def tokenize_function(batch):
        return batch

    @staticmethod
    def postprocess(predictions):
        return pd.DataFrame({"score": predictions[:, 0]})


class FakeTrainer:
    def __init__(self, fail_on_call=None, is_main_process=True, num_processes=1):
        self.accelerator = SimpleNamespace(
            is_main_process=is_main_process,
            num_processes=num_processes,
        )
        self.fail_on_call = fail_on_call
        self.calls = []

    def predict(self, test_dataset):
        self.calls.append(list(test_dataset.values))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("simulated interruption")
        predictions = np.asarray(test_dataset.values, dtype=float)[:, None]
        return SimpleNamespace(predictions=predictions)


def run_checkpointed(
    monkeypatch,
    tmp_path,
    trainer,
    *,
    dataset=None,
    signature=None,
    cleanup=False,
):
    dataset = dataset if dataset is not None else FakeDataset(range(7))
    signature = signature or {
        "command": "logits",
        "dataset": {"fingerprint": dataset._fingerprint},
        "model": {"config": "v1"},
    }
    trainer_factory_calls = []

    def fake_create_trainer(*args, **kwargs):
        trainer_factory_calls.append((args, kwargs))
        return trainer

    monkeypatch.setattr(
        inference_module,
        "_create_trainer",
        fake_create_trainer,
    )
    output = tmp_path / "predictions.parquet"
    checkpoints = tmp_path / "checkpoints"
    result = inference_module.run_inference_with_checkpoints(
        dataset,
        FakeInference(),
        output_path=output,
        checkpoint_dir=checkpoints,
        checkpoint_batch_size=3,
        run_signature=signature,
        per_device_batch_size=2,
        dataloader_num_workers=0,
        cleanup_checkpoints=cleanup,
    )
    return result, output, checkpoints, trainer_factory_calls


def test_interrupted_run_resumes_only_missing_batches_in_row_order(
    monkeypatch,
    tmp_path,
):
    interrupted_trainer = FakeTrainer(fail_on_call=2)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_checkpointed(monkeypatch, tmp_path, interrupted_trainer)

    checkpoints = tmp_path / "checkpoints"
    assert interrupted_trainer.calls == [[0, 1, 2], [3, 4, 5]]
    assert sorted(path.name for path in checkpoints.glob("batch_*.parquet")) == [
        "batch_00000000.parquet"
    ]
    assert not (tmp_path / "predictions.parquet").exists()

    resumed_trainer = FakeTrainer()
    result, output, _, factory_calls = run_checkpointed(
        monkeypatch,
        tmp_path,
        resumed_trainer,
    )

    assert result == output
    assert len(factory_calls) == 1
    assert resumed_trainer.calls == [[3, 4, 5], [6]]
    assert pd.read_parquet(output)["score"].tolist() == list(np.arange(7, dtype=float))


def test_incompatible_resume_is_rejected_before_prediction(
    monkeypatch,
    tmp_path,
):
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_checkpointed(
            monkeypatch,
            tmp_path,
            FakeTrainer(fail_on_call=2),
        )

    trainer = FakeTrainer()
    with pytest.raises(IncompatibleCheckpointError, match="incompatible"):
        run_checkpointed(
            monkeypatch,
            tmp_path,
            trainer,
            signature={
                "command": "logits",
                "dataset": {"fingerprint": "different-dataset"},
                "model": {"config": "v1"},
            },
        )

    assert trainer.calls == []
    assert not (tmp_path / "predictions.parquet").exists()


def test_cleanup_occurs_only_after_final_output_is_committed(
    monkeypatch,
    tmp_path,
):
    result, output, checkpoints, _ = run_checkpointed(
        monkeypatch,
        tmp_path,
        FakeTrainer(),
        cleanup=True,
    )

    assert result == output
    assert output.is_file()
    assert pd.read_parquet(output)["score"].tolist() == list(np.arange(7, dtype=float))
    assert not checkpoints.exists()


@pytest.mark.parametrize(
    ("dataset", "checkpoint_batch_size", "message"),
    [
        (FakeDataset([]), 3, "at least one row"),
        (FakeDataset([1]), 0, "checkpoint_batch_size must be positive"),
    ],
)
def test_invalid_inputs_fail_before_filesystem_mutation(
    monkeypatch,
    tmp_path,
    dataset,
    checkpoint_batch_size,
    message,
):
    monkeypatch.setattr(
        inference_module,
        "_create_trainer",
        lambda *args, **kwargs: pytest.fail("Trainer should not be created"),
    )
    checkpoints = tmp_path / "checkpoints"

    with pytest.raises(ValueError, match=message):
        inference_module.run_inference_with_checkpoints(
            dataset,
            FakeInference(),
            output_path=tmp_path / "predictions.parquet",
            checkpoint_dir=checkpoints,
            checkpoint_batch_size=checkpoint_batch_size,
            run_signature={"command": "logits"},
        )

    assert not checkpoints.exists()


def test_non_main_process_does_not_execute_filesystem_operation(monkeypatch):
    trainer = FakeTrainer(is_main_process=False, num_processes=2)
    called = False

    def operation():
        nonlocal called
        called = True

    def fake_broadcast(payload, from_process):
        assert from_process == 0
        assert payload == [None]
        payload[0] = {"ok": True, "value": "rank-zero-result"}

    monkeypatch.setattr(
        inference_module,
        "broadcast_object_list",
        fake_broadcast,
    )

    result = inference_module._call_on_main_process(
        trainer,
        operation,
        "test operation",
    )

    assert result == "rank-zero-result"
    assert called is False


def test_worker_receives_rank_zero_checkpoint_error(monkeypatch):
    trainer = FakeTrainer(is_main_process=False, num_processes=2)

    def fake_broadcast(payload, from_process):
        payload[0] = {
            "ok": False,
            "error_type": "InvalidCheckpointError",
            "error": "corrupt batch",
        }

    monkeypatch.setattr(
        inference_module,
        "broadcast_object_list",
        fake_broadcast,
    )

    with pytest.raises(
        RuntimeError,
        match="InvalidCheckpointError: corrupt batch",
    ):
        inference_module._call_on_main_process(
            trainer,
            lambda: None,
            "Resume audit",
        )


def test_run_signature_tracks_source_revision_resource_metadata_and_msa_order(
    tmp_path,
):
    input_path = tmp_path / "positions.parquet"
    input_path.write_bytes(b"positions")
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "GPNStar"}')
    first_msa = tmp_path / "first.zarr"
    second_msa = tmp_path / "second.zarr"
    for msa_path in [first_msa, second_msa]:
        msa_path.mkdir()
        (msa_path / ".zgroup").write_text('{"zarr_format": 2}')
        chromosome = msa_path / "chr1"
        chromosome.mkdir()
        (chromosome / ".zarray").write_text('{"shape": [10, 2]}')

    config = SimpleNamespace(
        phylo_dist_path=None,
        _commit_hash="model-commit",
        to_dict=lambda: {"model_type": "GPNStar"},
    )
    model = SimpleNamespace(model=SimpleNamespace(config=config))
    fake_inference = SimpleNamespace(model=model)
    args = SimpleNamespace(
        checkpoint_revision="msa-release-2",
        command="logits",
        input_path=str(input_path),
        is_file=True,
        split="test",
        center_window_size=None,
        disable_aux_features=False,
        window_size=128,
        model_path=str(model_path),
    )
    msa_paths = {
        100: str(first_msa),
        36: str(second_msa),
    }

    signature = inference_module.build_run_signature(
        args,
        FakeDataset([1]),
        msa_paths,
        fake_inference,
    )

    assert signature["checkpoint_revision"] == "msa-release-2"
    assert signature["software"]["gpn_source"]["sha256"]
    assert signature["model"]["config"]["commit_hash"] == "model-commit"
    assert [(item["order"], item["n_species"]) for item in signature["msa"]] == [
        (0, "100"),
        (1, "36"),
    ]
    assert all(
        item["resource"]["tree"]["strategy"] == "zarr-metadata-and-array-directories-v1"
        for item in signature["msa"]
    )


def test_direct_inference_reuses_main_process_semantics(monkeypatch):
    trainer = FakeTrainer()
    monkeypatch.setattr(
        inference_module,
        "_create_trainer",
        lambda *args, **kwargs: trainer,
    )

    result = inference_module.run_inference(
        FakeDataset([2, 4]),
        FakeInference(),
    )

    assert trainer.calls == [[2, 4]]
    assert result["score"].tolist() == [2.0, 4.0]


def test_atomic_direct_write_preserves_existing_output_on_failure(
    monkeypatch,
    tmp_path,
):
    output = tmp_path / "predictions.parquet"
    original = b"previous output"
    output.write_bytes(original)

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(inference_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        inference_module._write_parquet_atomic(
            pd.DataFrame({"score": [1.0]}),
            output,
        )

    assert output.read_bytes() == original
    assert list(tmp_path.glob(".predictions.parquet.*.tmp")) == []


def test_atomic_direct_write_uses_normal_permissions_and_preserves_mode(tmp_path):
    output = tmp_path / "predictions.parquet"

    inference_module._write_parquet_atomic(
        pd.DataFrame({"score": [1.0]}),
        output,
    )
    assert stat.S_IMODE(output.stat().st_mode) == DEFAULT_FILE_MODE

    output.chmod(0o640)
    inference_module._write_parquet_atomic(
        pd.DataFrame({"score": [2.0]}),
        output,
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o640
