import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from transformers import TrainingArguments

import gpn.checkpoint as checkpoint_module
import gpn.inference as inference_module
import gpn.ss.inference as ss_inference
import gpn.star.inference as star_inference
from gpn.arguments import CheckpointArguments
from gpn.checkpoint import (
    DEFAULT_FILE_MODE,
    CheckpointManifest,
    CheckpointStore,
    IncompatibleCheckpointError,
)
from gpn.inference import InferenceRunner
from gpn.star.utils import find_directory_sum_paths


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


class FakeTokenizer:
    def __init__(self, state):
        self.state = state

    def save_pretrained(self, directory):
        path = Path(directory) / "tokenizer.json"
        path.write_text(self.state)
        return (str(path),)


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
    runner_factory_calls = []

    def fake_create_runner(*args, **kwargs):
        runner_factory_calls.append((args, kwargs))
        return InferenceRunner(trainer)

    monkeypatch.setattr(
        inference_module,
        "create_inference_runner",
        fake_create_runner,
    )
    output = tmp_path / "predictions.parquet"
    checkpoints = tmp_path / "checkpoints"
    result = inference_module.run_inference_with_checkpoints(
        dataset,
        FakeInference(),
        TrainingArguments(use_cpu=True),
        CheckpointArguments(
            checkpoint_batch_size=3,
            checkpoint_dir=checkpoints,
            cleanup_checkpoints=cleanup,
        ),
        output_path=output,
        run_signature=signature,
        output_prefix="test-checkpoint-",
    )
    return result, output, checkpoints, runner_factory_calls


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


def test_empty_dataset_fails_before_filesystem_mutation(tmp_path):
    checkpoints = tmp_path / "checkpoints"

    with pytest.raises(ValueError, match="at least one row"):
        inference_module.run_inference_with_checkpoints(
            FakeDataset([]),
            FakeInference(),
            TrainingArguments(use_cpu=True),
            CheckpointArguments(
                checkpoint_batch_size=3,
                checkpoint_dir=checkpoints,
            ),
            output_path=tmp_path / "predictions.parquet",
            run_signature={"command": "logits"},
            output_prefix="test-checkpoint-",
        )

    assert not checkpoints.exists()


def test_invalid_checkpoint_options_fail_at_typed_boundary():
    with pytest.raises(ValueError, match="checkpoint_batch_size must be positive"):
        CheckpointArguments(checkpoint_batch_size=0)
    with pytest.raises(ValueError, match="require checkpoint_batch_size"):
        CheckpointArguments(checkpoint_dir=Path("checkpoints"))


def test_non_main_process_does_not_execute_filesystem_operation(monkeypatch):
    runner = InferenceRunner(FakeTrainer(is_main_process=False, num_processes=2))
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
        runner,
        operation,
        "test operation",
    )

    assert result == "rank-zero-result"
    assert called is False


def test_worker_receives_rank_zero_checkpoint_error(monkeypatch):
    runner = InferenceRunner(FakeTrainer(is_main_process=False, num_processes=2))

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
            runner,
            lambda: None,
            "Resume audit",
        )


def test_star_signature_tracks_source_resources_and_msa_order(
    monkeypatch,
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
    captured = {}

    def fake_execute(*args, **kwargs):
        captured.update(kwargs["run_signature_factory"]())

    monkeypatch.setattr(star_inference, "execute_inference", fake_execute)
    msa_paths = {100: str(first_msa), 36: str(second_msa)}

    star_inference._execute(
        operation="logits",
        dataset=FakeDataset([1]),
        input_path=str(input_path),
        msa_paths=msa_paths,
        model_path=str(model_path),
        output_path=tmp_path / "output.parquet",
        split="test",
        is_file=True,
        inference=fake_inference,
        training_arguments=TrainingArguments(use_cpu=True),
        checkpoint_arguments=CheckpointArguments(
            checkpoint_batch_size=3, checkpoint_revision="msa-release-2"
        ),
        operation_arguments={"window_size": 128},
    )

    assert captured["checkpoint_revision"] == "msa-release-2"
    assert captured["software"]["gpn_source"]["sha256"]
    assert captured["model"]["config"]["commit_hash"] == "model-commit"
    msa = captured["resources"]["msa"]
    assert [(item["order"], item["n_species"]) for item in msa] == [
        (0, 100),
        (1, 36),
    ]
    assert all(
        item["resource"]["tree"]["strategy"] == "zarr-metadata-and-array-directories-v1"
        for item in msa
    )


def test_star_msa_discovery_orders_species_counts_numerically(tmp_path):
    for species_count in (9, 36, 100):
        (tmp_path / str(species_count) / "all.zarr").mkdir(parents=True)

    result = find_directory_sum_paths(str(tmp_path))

    assert list(result) == [100, 36, 9]
    assert [Path(path).parent.name for path in result.values()] == ["100", "36", "9"]


def test_star_msa_discovery_accepts_trailing_slash_on_species_directory(tmp_path):
    species_directory = tmp_path / "100"
    (species_directory / "all.zarr").mkdir(parents=True)

    result = find_directory_sum_paths(f"{species_directory}/")

    assert result == {100: str(species_directory / "all.zarr")}


def test_ss_checkpoint_resume_rejects_changed_tokenizer(
    monkeypatch,
    tmp_path,
):
    input_path = tmp_path / "positions.parquet"
    input_path.write_bytes(b"positions")
    genome_path = tmp_path / "genome.fa"
    genome_path.write_text(">chr1\nACGT\n")
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "GPN"}')
    tokenizer_paths = []
    for name, vocab in (("tokenizer-a", "ACGT"), ("tokenizer-b", "TGCA")):
        path = tmp_path / name
        path.mkdir()
        (path / "tokenizer.json").write_text(vocab)
        tokenizer_paths.append(path)

    captured_signatures = []

    def fake_execute(*args, **kwargs):
        captured_signatures.append(kwargs["run_signature_factory"]())

    monkeypatch.setattr(ss_inference, "execute_inference", fake_execute)
    for tokenizer_path, state in zip(tokenizer_paths, ("ACGT", "TGCA"), strict=True):
        fake_inference = SimpleNamespace(
            model=object(),
            tokenizer=FakeTokenizer(state),
        )
        ss_inference._execute(
            operation="logits",
            dataset=FakeDataset([1]),
            input_path=str(input_path),
            genome_path=str(genome_path),
            model_path=str(model_path),
            tokenizer_path=str(tokenizer_path),
            output_path=tmp_path / "output.parquet",
            split="test",
            is_file=True,
            inference=fake_inference,
            training_arguments=TrainingArguments(use_cpu=True),
            checkpoint_arguments=CheckpointArguments(checkpoint_batch_size=1),
            operation_arguments={"window_size": 128},
        )

    assert (
        captured_signatures[0]["resources"]["tokenizer"]
        != captured_signatures[1]["resources"]["tokenizer"]
    )
    directory = tmp_path / "checkpoints"
    first = CheckpointManifest(captured_signatures[0], total_rows=1, batch_size=1)
    second = CheckpointManifest(captured_signatures[1], total_rows=1, batch_size=1)
    CheckpointStore(directory, first).initialize()
    with pytest.raises(IncompatibleCheckpointError, match="tokenizer"):
        CheckpointStore(directory, second).initialize()


def test_ss_checkpoint_resume_rejects_mutated_remote_tokenizer(
    monkeypatch,
    tmp_path,
):
    signatures = []

    def fake_execute(*args, **kwargs):
        signatures.append(kwargs["run_signature_factory"]())

    monkeypatch.setattr(ss_inference, "execute_inference", fake_execute)
    for state in ("first-vocabulary", "mutated-vocabulary"):
        ss_inference._execute(
            operation="logits",
            dataset=FakeDataset([1]),
            input_path="songlab/positions",
            genome_path="songlab/genome",
            model_path="songlab/gpn",
            tokenizer_path="songlab/mutable-tokenizer",
            output_path=tmp_path / "output.parquet",
            split="test",
            is_file=False,
            inference=SimpleNamespace(
                model=object(),
                tokenizer=FakeTokenizer(state),
            ),
            training_arguments=TrainingArguments(use_cpu=True),
            checkpoint_arguments=CheckpointArguments(checkpoint_batch_size=1),
            operation_arguments={"window_size": 128},
        )

    tokenizer_identities = [
        signature["resources"]["tokenizer"] for signature in signatures
    ]
    assert tokenizer_identities[0]["resource"] == tokenizer_identities[1]["resource"]
    assert tokenizer_identities[0]["effective"] != tokenizer_identities[1]["effective"]

    directory = tmp_path / "remote-tokenizer-checkpoints"
    first = CheckpointManifest(signatures[0], total_rows=1, batch_size=1)
    second = CheckpointManifest(signatures[1], total_rows=1, batch_size=1)
    CheckpointStore(directory, first).initialize()
    with pytest.raises(IncompatibleCheckpointError, match="tokenizer"):
        CheckpointStore(directory, second).initialize()


def test_checkpoint_signature_tracks_prediction_affecting_trainer_options(
    monkeypatch,
    tmp_path,
):
    signatures = []

    def fake_execute(*args, **kwargs):
        signatures.append(kwargs["run_signature_factory"]())

    monkeypatch.setattr(star_inference, "execute_inference", fake_execute)
    fake_inference = SimpleNamespace(model=object())
    common = {
        "operation": "logits",
        "dataset": FakeDataset([1]),
        "input_path": "songlab/positions",
        "msa_paths": {100: "songlab/msa-100"},
        "model_path": "songlab/gpn-star",
        "output_path": tmp_path / "output.parquet",
        "split": "test",
        "is_file": False,
        "inference": fake_inference,
        "checkpoint_arguments": CheckpointArguments(checkpoint_batch_size=1),
        "operation_arguments": {"window_size": 128},
    }
    star_inference._execute(
        **common,
        training_arguments=TrainingArguments(
            use_cpu=True,
            per_device_eval_batch_size=8,
        ),
    )
    star_inference._execute(
        **common,
        training_arguments=TrainingArguments(
            use_cpu=True,
            per_device_eval_batch_size=16,
        ),
    )

    assert (
        signatures[0]["inference"]["runtime"] != signatures[1]["inference"]["runtime"]
    )
    directory = tmp_path / "trainer-option-checkpoints"
    first = CheckpointManifest(signatures[0], total_rows=1, batch_size=1)
    second = CheckpointManifest(signatures[1], total_rows=1, batch_size=1)
    CheckpointStore(directory, first).initialize()
    with pytest.raises(
        IncompatibleCheckpointError,
        match="per_device_eval_batch_size",
    ):
        CheckpointStore(directory, second).initialize()


def test_direct_inference_does_not_build_checkpoint_signature(
    monkeypatch,
    tmp_path,
):
    signature_built = False

    def fail_if_called():
        nonlocal signature_built
        signature_built = True
        raise AssertionError("checkpoint signature should be lazy")

    monkeypatch.setattr(
        inference_module,
        "run_inference",
        lambda *args, **kwargs: pd.DataFrame({"score": [1.0]}),
    )
    monkeypatch.setattr(
        inference_module,
        "write_dataframe_atomic",
        lambda frame, output_path: Path(output_path),
    )

    result = inference_module.execute_inference(
        FakeDataset([1]),
        FakeInference(),
        TrainingArguments(use_cpu=True),
        CheckpointArguments(),
        output_path=tmp_path / "output.parquet",
        run_signature_factory=fail_if_called,
        output_prefix="test-direct-",
    )

    assert result == tmp_path / "output.parquet"
    assert signature_built is False


def test_direct_inference_reuses_main_process_semantics(monkeypatch):
    trainer = FakeTrainer()
    monkeypatch.setattr(
        inference_module,
        "create_inference_runner",
        lambda *args, **kwargs: InferenceRunner(trainer),
    )

    result = inference_module.run_inference(
        FakeDataset([2, 4]),
        FakeInference(),
        TrainingArguments(use_cpu=True),
        output_prefix="test-direct-",
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

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        checkpoint_module.write_dataframe_atomic(
            pd.DataFrame({"score": [1.0]}),
            output,
        )

    assert output.read_bytes() == original
    assert list(tmp_path.glob(".predictions.parquet.*.tmp")) == []


def test_atomic_direct_write_uses_normal_permissions_and_preserves_mode(tmp_path):
    output = tmp_path / "predictions.parquet"

    checkpoint_module.write_dataframe_atomic(
        pd.DataFrame({"score": [1.0]}),
        output,
    )
    assert stat.S_IMODE(output.stat().st_mode) == DEFAULT_FILE_MODE

    output.chmod(0o640)
    checkpoint_module.write_dataframe_atomic(
        pd.DataFrame({"score": [2.0]}),
        output,
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o640
