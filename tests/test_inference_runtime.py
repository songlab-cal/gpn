import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import TrainingArguments

import gpn.inference as shared_inference
import gpn.msa.data as msa_data
import gpn.msa.inference as msa_inference
import gpn.ss.inference as ss_inference
import gpn.star.inference as star_inference
from gpn.data import ReverseComplementer, Tokenizer

BASELINE = json.loads(
    (Path(__file__).parent / "fixtures" / "published_model_baseline.json").read_text()
)


class FakeDataset:
    def __init__(self, n_rows: int = 1) -> None:
        self.transform = None
        self.n_rows = n_rows

    def __len__(self) -> int:
        return self.n_rows

    def set_transform(self, transform) -> None:
        self.transform = transform


class FakeInference:
    model = object()

    @staticmethod
    def tokenize_function(batch):
        return batch

    @staticmethod
    def postprocess(predictions):
        return pd.DataFrame({"score": predictions[:, 0]})


class FakeTrainer:
    def __init__(self, model, args, *, is_main_process=True) -> None:
        self.model = model
        self.args = args
        self.accelerator = SimpleNamespace(
            is_main_process=is_main_process,
            num_processes=1,
        )

    @staticmethod
    def predict(test_dataset):
        return SimpleNamespace(predictions=np.array([[1.25]]))


def test_inference_defaults_are_explicit_fp32_without_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainers = []

    def fake_trainer(model, args):
        trainer = FakeTrainer(model, args)
        trainers.append(trainer)
        return trainer

    monkeypatch.setattr(shared_inference, "Trainer", fake_trainer)

    result = shared_inference.run_inference(
        FakeDataset(),
        FakeInference(),
        TrainingArguments(use_cpu=True),
        output_prefix="test-gpn-inference-",
    )

    assert result["score"].tolist() == [1.25]
    arguments = trainers[0].args
    assert arguments.fp16_full_eval is False
    assert arguments.bf16_full_eval is False
    assert arguments.torch_compile is False
    assert arguments.remove_unused_columns is False
    assert arguments.prediction_loss_only is False
    assert arguments.dataloader_drop_last is False
    assert arguments.dataloader_in_order is True
    assert arguments.do_train is False
    assert arguments.do_predict is True
    assert trainers[0].args.output_dir


def test_inference_preserves_explicit_transformers_prediction_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trainers = []
    monkeypatch.setattr(
        shared_inference,
        "Trainer",
        lambda model, args: trainers.append(FakeTrainer(model, args)) or trainers[-1],
    )
    arguments = TrainingArguments(
        output_dir=str(tmp_path / "trainer"),
        use_cpu=True,
        per_device_eval_batch_size=17,
        dataloader_num_workers=2,
        full_determinism=True,
        do_train=True,
        prediction_loss_only=True,
        remove_unused_columns=True,
        dataloader_drop_last=True,
        dataloader_in_order=False,
    )

    shared_inference.run_inference(
        FakeDataset(),
        FakeInference(),
        arguments,
        output_prefix="unused-",
    )

    actual = trainers[0].args
    assert actual.output_dir == str(tmp_path / "trainer")
    assert actual.per_device_eval_batch_size == 17
    assert actual.dataloader_num_workers == 2
    assert actual.full_determinism is True
    assert actual.do_train is False
    assert actual.prediction_loss_only is False
    assert actual.remove_unused_columns is False
    assert actual.dataloader_drop_last is False
    assert actual.dataloader_in_order is True


def test_direct_inference_rejects_missing_prediction_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shared_inference, "Trainer", FakeTrainer)

    with pytest.raises(
        ValueError,
        match="1 predictions for 2 rows",
    ):
        shared_inference.run_inference(
            FakeDataset(n_rows=2),
            FakeInference(),
            TrainingArguments(use_cpu=True),
            output_prefix="test-gpn-inference-",
        )


def test_inference_rejects_hub_publication() -> None:
    with pytest.raises(ValueError, match="push_to_hub is not supported"):
        shared_inference.inference_training_arguments(
            TrainingArguments(output_dir="output", push_to_hub=True)
        )


def test_compile_errors_are_not_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTrainer(FakeTrainer):
        @staticmethod
        def predict(test_dataset):
            raise RuntimeError("compile failed")

    monkeypatch.setattr(shared_inference, "Trainer", FailingTrainer)
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)

    with pytest.raises(RuntimeError, match="compile failed"):
        shared_inference.run_inference(
            FakeDataset(),
            FakeInference(),
            TrainingArguments(use_cpu=True, torch_compile=True),
            output_prefix="test-gpn-compile-",
        )

    assert torch._dynamo.config.suppress_errors is False


def test_gpn_vep_loads_installed_model_without_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registrations = []
    loads = []

    class FakeModel:
        def eval(self):
            return self

    def fake_from_pretrained(*args, **kwargs):
        loads.append((args, kwargs))
        return FakeModel()

    monkeypatch.setattr(ss_inference, "register_auto_classes", registrations.append)
    monkeypatch.setattr(
        ss_inference.AutoModelForMaskedLM,
        "from_pretrained",
        fake_from_pretrained,
    )

    wrapped = ss_inference.MLMforVEPModel("songlab/gpn-brassicales")

    assert isinstance(wrapped.model, FakeModel)
    assert registrations == ["ss"]
    assert loads == [(("songlab/gpn-brassicales",), {})]


class FakeGenome:
    def __init__(self, sequence="GGGACG") -> None:
        self.sequence = sequence
        self.calls = []

    def get_seq_fwd_rev(self, chrom, start, end):
        self.calls.append((chrom, start, end))
        complement = str.maketrans("ACGT", "TGCA")
        return self.sequence, self.sequence.translate(complement)[::-1]


class FakeSequenceTokenizer:
    mask_token = "?"
    vocab = "ACGT?"

    def __call__(self, sequences, **kwargs):
        def encode(sequence):
            return [self.vocab.index(base) for base in sequence]

        if isinstance(sequences, str):
            encoded = encode(sequences)
        else:
            encoded = [encode(sequence) for sequence in sequences]
        return {"input_ids": encoded}


class FakeAlignment:
    def __init__(self, reference="GGGACG") -> None:
        tokenizer = Tokenizer()
        forward = tokenizer(np.array(list(reference), dtype="S1"))
        reverse_sequence = str.maketrans("ACGT", "TGCA")
        reverse = tokenizer(
            np.array(list(reference.translate(reverse_sequence)[::-1]), dtype="S1")
        )
        self.forward = np.stack((forward, forward), axis=-1)[None, ...]
        self.reverse = np.stack((reverse, reverse), axis=-1)[None, ...]
        self.calls = []

    def get_msa_batch_fwd_rev(self, chrom, start, end, tokenize=True):
        self.calls.append((chrom.tolist(), start.tolist(), end.tolist(), tokenize))
        return self.forward.copy(), self.reverse.copy()


def test_gpn_vep_tokenization_preserves_vcf_coordinate_and_orientation_contract():
    genome = FakeGenome()

    actual = ss_inference._tokenize_variant_batch(
        {"chrom": ["chr1"], "pos": [100], "ref": ["A"], "alt": ["C"]},
        genome,
        window_size=6,
        tokenizer=FakeSequenceTokenizer(),
    )

    assert genome.calls == [("chr1", 96, 102)]
    assert actual["pos_fwd"] == [3]
    assert actual["pos_rev"] == [2]
    assert actual["ref_fwd"] == [0]
    assert actual["alt_fwd"] == [1]
    assert actual["ref_rev"] == [3]
    assert actual["alt_rev"] == [2]


@pytest.mark.parametrize("module", (msa_inference, star_inference))
def test_msa_vep_tokenization_preserves_coordinate_and_orientation_contract(module):
    alignment = FakeAlignment()
    inference = object.__new__(module.VEPInference)
    inference.window_size = 6
    inference.reverse_complementer = ReverseComplementer()
    inference.tokenizer = Tokenizer()
    if module is msa_inference:
        inference.genome_msa = alignment
    else:
        inference.genome_msa_list = [alignment]

    actual = inference.tokenize_function(
        {"chrom": ["chr1"], "pos": [100], "ref": ["A"], "alt": ["C"]}
    )

    assert alignment.calls == [(["chr1"], [96], [102], True)]
    assert actual["pos_fwd"].tolist() == [3]
    assert actual["pos_rev"].tolist() == [2]
    assert actual["ref_fwd"].tolist() == [1]
    assert actual["alt_fwd"].tolist() == [2]
    assert actual["ref_rev"].tolist() == [4]
    assert actual["alt_rev"].tolist() == [3]


@pytest.mark.parametrize("module", (msa_inference, star_inference))
def test_msa_vep_reference_mismatch_fails_with_variant_coordinate(module):
    alignment = FakeAlignment(reference="GGGCCG")
    inference = object.__new__(module.VEPInference)
    inference.window_size = 6
    inference.reverse_complementer = ReverseComplementer()
    inference.tokenizer = Tokenizer()
    if module is msa_inference:
        inference.genome_msa = alignment
    else:
        inference.genome_msa_list = [alignment]

    with pytest.raises(ValueError, match=r"chr1:100.*genome has 'C'.*input has 'A'"):
        inference.tokenize_function(
            {"chrom": ["chr1"], "pos": [100], "ref": ["A"], "alt": ["C"]}
        )


@pytest.mark.parametrize(
    "inference_class",
    (
        ss_inference.LogitsInference,
        msa_inference.LogitsInference,
        star_inference.LogitsInference,
    ),
)
@pytest.mark.parametrize("position", (0, -1, 1.5, True))
def test_logits_reject_nonpositive_or_noninteger_one_based_positions(
    inference_class,
    position,
):
    inference = object.__new__(inference_class)

    with pytest.raises(ValueError, match="positive"):
        inference.tokenize_function({"chrom": ["chr1"], "pos": [position]})


def test_msa_boundary_windows_are_gap_padded_on_both_strands():
    genome_msa = object.__new__(msa_data.GenomeMSA)
    genome_msa.data = {
        "chr1": np.array(
            [list("AA"), list("CC"), list("GG"), list("TT")],
            dtype="S1",
        )
    }
    genome_msa.reverse_complementer = ReverseComplementer()
    genome_msa.tokenizer = Tokenizer()

    forward, reverse = genome_msa.get_msa_batch_fwd_rev(
        np.array(["chr1", "chr1"]),
        np.array([-2, 2]),
        np.array([4, 8]),
    )

    assert forward[:, :, 0].astype(str).tolist() == [
        ["-", "-", "A", "C", "G", "T"],
        ["G", "T", "-", "-", "-", "-"],
    ]
    assert reverse[:, :, 0].astype(str).tolist() == [
        ["A", "C", "G", "T", "-", "-"],
        ["-", "-", "-", "-", "A", "C"],
    ]


class FixedEmbeddingModel:
    def __call__(self, **kwargs):
        return SimpleNamespace(
            last_hidden_state=torch.tensor([[[0.0], [1.0], [4.0], [9.0], [16.0]]])
        )


@pytest.mark.parametrize(
    ("wrapper", "extra_arguments"),
    (
        (ss_inference.ModelCenterEmbedding, ()),
        (msa_inference.ModelCenterEmbedding, (torch.zeros((1, 5, 1)),)),
        (
            star_inference.ModelCenterEmbedding,
            (torch.zeros((1, 5, 1)), torch.zeros((1, 1))),
        ),
    ),
)
@pytest.mark.parametrize(
    ("center_window_size", "expected"),
    ((2, 2.5), (3, 14.0 / 3.0)),
)
def test_embedding_center_window_uses_exact_odd_and_even_size(
    wrapper,
    extra_arguments,
    center_window_size,
    expected,
):
    instance = SimpleNamespace(
        model=FixedEmbeddingModel(),
        center_window_size=center_window_size,
    )

    actual = wrapper.get_center_embedding(
        instance,
        torch.zeros((1, 5), dtype=torch.long),
        *extra_arguments,
    )

    torch.testing.assert_close(actual, torch.tensor([[expected]]))


@pytest.mark.parametrize(
    "inference_class",
    (
        msa_inference.VEPInference,
        msa_inference.LogitsInference,
        star_inference.VEPInference,
        star_inference.LogitsInference,
    ),
)
def test_centered_msa_inference_rejects_odd_windows_before_loading_model(
    inference_class,
):
    with pytest.raises(ValueError, match="must be even"):
        inference_class("model", object(), 5)


class FixedLogitModel:
    def __init__(self, logits) -> None:
        self.logits = logits

    def __call__(self, **kwargs):
        return SimpleNamespace(logits=self.logits)


@pytest.mark.parametrize(
    ("family", "wrapper", "logit_shape", "reference_id", "alternate_id", "kwargs"),
    (
        (
            "gpn",
            ss_inference.MLMforVEPModel,
            (1, 1, 4),
            0,
            1,
            {},
        ),
        (
            "gpn_msa",
            msa_inference.MLMforVEPModel,
            (1, 1, 6),
            1,
            2,
            {"aux_features": np.zeros((1, 1, 1))},
        ),
        (
            "gpn_star",
            star_inference.MLMforVEPModel,
            (1, 1, 1, 6),
            1,
            2,
            {
                "source_ids": np.zeros((1, 1, 1)),
                "target_species": np.zeros((1, 1)),
            },
        ),
    ),
)
def test_vep_wrappers_compute_published_alt_minus_ref_likelihood_direction(
    family,
    wrapper,
    logit_shape,
    reference_id,
    alternate_id,
    kwargs,
):
    expected = BASELINE["models"][family]["expected"]
    logits = torch.zeros(logit_shape)
    logits[..., reference_id] = expected["logits"][0]
    logits[..., alternate_id] = expected["logits"][1]
    instance = SimpleNamespace(model=FixedLogitModel(logits))

    actual = wrapper.get_llr(
        instance,
        input_ids=torch.zeros((1, 1), dtype=torch.long),
        pos=torch.tensor([0]),
        ref=torch.tensor([reference_id]),
        alt=torch.tensor([alternate_id]),
        **kwargs,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([expected["llr_alt_minus_ref"]["C"]]),
        rtol=BASELINE["models"][family]["relative_tolerance"],
        atol=BASELINE["models"][family]["absolute_tolerance"],
    )


@pytest.mark.parametrize(
    "wrapper",
    (
        ss_inference.MLMforVEPModel,
        msa_inference.MLMforVEPModel,
        star_inference.MLMforVEPModel,
    ),
)
def test_vep_wrappers_average_forward_and_reverse_scores(wrapper):
    instance = SimpleNamespace(
        get_llr=Mock(side_effect=(torch.tensor([-2.0]), torch.tensor([-4.0])))
    )

    actual = wrapper.forward(instance)

    torch.testing.assert_close(actual, torch.tensor([-3.0]))
    assert instance.get_llr.call_count == 2
