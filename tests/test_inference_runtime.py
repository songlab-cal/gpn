import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

import gpn.msa.inference as msa_inference
import gpn.msa.logits as msa_logits
import gpn.msa.vep as msa_vep
import gpn.ss.run_vep as gpn_vep
import gpn.star.inference as star_inference
import gpn.star.logits as star_logits
import gpn.star.vep as star_vep
from gpn.data import ReverseComplementer, Tokenizer

BASELINE = json.loads(
    (Path(__file__).parent / "fixtures" / "published_model_baseline.json").read_text()
)


class FakeDataset:
    def __init__(self) -> None:
        self.transform = None

    def __len__(self) -> int:
        return 1

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
        self.accelerator = SimpleNamespace(is_main_process=is_main_process)

    @staticmethod
    def predict(test_dataset):
        return SimpleNamespace(predictions=np.array([[1.25]]))


@pytest.mark.parametrize("module", (gpn_vep, msa_inference, star_inference))
def test_inference_cpu_defaults_disable_gpu_only_acceleration(
    monkeypatch: pytest.MonkeyPatch,
    module,
) -> None:
    captured = {}

    def fake_training_arguments(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    trainer = None

    def fake_trainer(model, args):
        nonlocal trainer
        trainer = FakeTrainer(model, args)
        return trainer

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module, "TrainingArguments", fake_training_arguments)
    monkeypatch.setattr(module, "Trainer", fake_trainer)

    dataset = FakeDataset()
    if module is gpn_vep:
        result = module.run_vep(
            dataset,
            genome=object(),
            window_size=16,
            tokenizer=object(),
            model=object(),
        )
        assert result.tolist() == [[1.25]]
    else:
        result = module.run_inference(dataset, FakeInference())
        assert result["score"].tolist() == [1.25]

    assert captured["fp16"] is False
    assert captured["torch_compile"] is False
    assert captured["report_to"] == "none"
    assert trainer is not None
    temporary_directory = trainer._gpn_temporary_output_dir
    assert Path(temporary_directory.name).is_dir()


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

    monkeypatch.setattr(gpn_vep, "register_auto_classes", registrations.append)
    monkeypatch.setattr(
        gpn_vep.AutoModelForMaskedLM,
        "from_pretrained",
        fake_from_pretrained,
    )

    wrapped = gpn_vep.MLMforVEPModel("songlab/gpn-brassicales")

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

    actual = gpn_vep._tokenize_variant_batch(
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


@pytest.mark.parametrize("module", (msa_vep, star_vep))
def test_msa_vep_tokenization_preserves_coordinate_and_orientation_contract(module):
    alignment = FakeAlignment()
    inference = object.__new__(module.VEPInference)
    inference.window_size = 6
    inference.disable_aux_features = False
    inference.reverse_complementer = ReverseComplementer()
    inference.tokenizer = Tokenizer()
    if module is msa_vep:
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


@pytest.mark.parametrize("module", (msa_vep, star_vep))
def test_msa_vep_reference_mismatch_fails_with_variant_coordinate(module):
    alignment = FakeAlignment(reference="GGGCCG")
    inference = object.__new__(module.VEPInference)
    inference.window_size = 6
    inference.disable_aux_features = False
    inference.reverse_complementer = ReverseComplementer()
    inference.tokenizer = Tokenizer()
    if module is msa_vep:
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
        msa_vep.VEPInference,
        msa_logits.LogitsInference,
        star_vep.VEPInference,
        star_logits.LogitsInference,
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

    def forward(self, **kwargs):
        return SimpleNamespace(logits=self.logits)


@pytest.mark.parametrize(
    ("family", "wrapper", "logit_shape", "reference_id", "alternate_id", "kwargs"),
    (
        (
            "gpn",
            gpn_vep.MLMforVEPModel,
            (1, 1, 4),
            0,
            1,
            {},
        ),
        (
            "gpn_msa",
            msa_vep.MLMforVEPModel,
            (1, 1, 6),
            1,
            2,
            {"aux_features": np.zeros((1, 1, 1))},
        ),
        (
            "gpn_star",
            star_vep.MLMforVEPModel,
            (1, 1, 1, 6),
            1,
            2,
            {"source_ids": np.zeros((1, 1, 1)), "target_species": np.zeros((1, 1))},
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
    (gpn_vep.MLMforVEPModel, msa_vep.MLMforVEPModel, star_vep.MLMforVEPModel),
)
def test_vep_wrappers_average_forward_and_reverse_scores(wrapper):
    instance = SimpleNamespace(
        get_llr=Mock(side_effect=(torch.tensor([-2.0]), torch.tensor([-4.0])))
    )

    actual = wrapper.forward(instance)

    torch.testing.assert_close(actual, torch.tensor([-3.0]))
    assert instance.get_llr.call_count == 2


def test_msa_compile_errors_are_suppressed_only_during_compiled_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def fake_training_arguments(**kwargs):
        return SimpleNamespace(**kwargs)

    class CompileAwareTrainer(FakeTrainer):
        @staticmethod
        def predict(test_dataset):
            observed.append(msa_inference.torch._dynamo.config.suppress_errors)
            return SimpleNamespace(predictions=np.array([[1.25]]))

    monkeypatch.setattr(msa_inference.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(msa_inference, "TrainingArguments", fake_training_arguments)
    monkeypatch.setattr(msa_inference, "Trainer", CompileAwareTrainer)
    monkeypatch.setattr(msa_inference.torch._dynamo.config, "suppress_errors", False)

    result = msa_inference.run_inference(FakeDataset(), FakeInference())

    assert result["score"].tolist() == [1.25]
    assert observed == [True]
    assert msa_inference.torch._dynamo.config.suppress_errors is False
