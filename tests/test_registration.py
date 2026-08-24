import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from gpn import register_auto_classes


def test_importing_gpn_is_lightweight_and_does_not_register_models():
    code = """
import sys

import gpn

assert "torch" not in sys.modules
assert "transformers" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_importing_command_modules_does_not_register_models():
    code = """
import importlib
from gpn import _registration

for module in (
    "gpn.ss.train",
    "gpn.ss.inference",
    "gpn.msa.inference",
    "gpn.star.train",
    "gpn.star.inference",
):
    importlib.import_module(module)

assert _registration._registered_families == set()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_installed_package_never_enables_hugging_face_remote_code():
    package_root = Path(__file__).parents[1] / "src" / "gpn"
    for source_path in package_root.rglob("*.py"):
        source = source_path.read_text()
        assert "trust_remote_code=True" not in source, source_path
        assert "trust_remote_code = True" not in source, source_path


def test_ss_registration_is_explicit_and_idempotent():
    from gpn.ss.model import (
        GPNConfig,
        GPNForMaskedLM,
        GPNForSequenceClassification,
        GPNForTokenClassification,
        GPNModel,
    )

    register_auto_classes("ss")
    register_auto_classes("ss")

    assert type(AutoConfig.for_model("GPN")) is GPNConfig
    assert AutoModel._model_mapping[GPNConfig] is GPNModel
    assert AutoModelForMaskedLM._model_mapping[GPNConfig] is GPNForMaskedLM
    assert (
        AutoModelForSequenceClassification._model_mapping[GPNConfig]
        is GPNForSequenceClassification
    )
    assert (
        AutoModelForTokenClassification._model_mapping[GPNConfig]
        is GPNForTokenClassification
    )


def test_star_registration_is_explicit_and_idempotent():
    from gpn.star.model import GPNStarConfig, GPNStarForMaskedLM, GPNStarModel

    register_auto_classes("star")
    register_auto_classes("star")

    assert type(AutoConfig.for_model("GPNStar")) is GPNStarConfig
    assert AutoModel._model_mapping[GPNStarConfig] is GPNStarModel
    assert AutoModelForMaskedLM._model_mapping[GPNStarConfig] is GPNStarForMaskedLM


def test_phylo_registration_is_explicit_and_idempotent(tmp_path: Path):
    from gpn.phylo.model import PhyloGPNConfig, PhyloGPNModel, PhyloGPNTokenizer

    register_auto_classes("phylo")
    register_auto_classes("phylo")

    config = AutoConfig.for_model(
        "phylogpn",
        outer_dim=8,
        inner_dim=4,
        kernel_size=3,
        stack_size=1,
        num_stacks=2,
    )
    assert type(config) is PhyloGPNConfig
    assert AutoModel._model_mapping[PhyloGPNConfig] is PhyloGPNModel

    config.save_pretrained(tmp_path)
    PhyloGPNTokenizer().save_pretrained(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    assert type(tokenizer) is PhyloGPNTokenizer

    model = AutoModel.from_config(config).eval()
    input_ids = tokenizer("ACGTACGT", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    assert set(outputs) == set("ACGT")
    assert all(output.shape == (1, 4) for output in outputs.values())

    model.save_pretrained(tmp_path)
    restored = AutoModel.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        restored_outputs = restored(input_ids=input_ids)
    for nucleotide in "ACGT":
        torch.testing.assert_close(restored_outputs[nucleotide], outputs[nucleotide])


def test_tiny_gpn_model_round_trip(tmp_path: Path):
    register_auto_classes("ss")

    config = AutoConfig.for_model(
        "GPN",
        vocab_size=7,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        first_kernel_size=3,
        rest_kernel_size=3,
        mlm_head_transform=False,
    )
    model = AutoModelForMaskedLM.from_config(config).eval()
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0]])

    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    model.save_pretrained(tmp_path)
    restored = AutoModelForMaskedLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = restored(input_ids=input_ids).logits

    assert actual.shape == (1, 8, 7)
    torch.testing.assert_close(actual, expected)


def test_published_ss_and_msa_architectures_construct_with_auto_classes():
    register_auto_classes("ss")
    register_auto_classes("msa")

    convnet_config = AutoConfig.for_model(
        "ConvNet",
        vocab_size=7,
        hidden_size=8,
        n_layers=1,
        kernel_size=3,
    )
    convnet = AutoModelForMaskedLM.from_config(convnet_config).eval()
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0]])
    with torch.no_grad():
        convnet_logits = convnet(input_ids=input_ids).logits
    assert convnet_logits.shape == (1, 8, 7)

    roformer_config = AutoConfig.for_model(
        "GPNRoFormer",
        vocab_size=6,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=16,
    )
    roformer = AutoModelForMaskedLM.from_config(roformer_config).eval()
    with torch.no_grad():
        roformer_logits = roformer(input_ids=input_ids[:, :6]).logits
    assert roformer_logits.shape == (1, 6, 6)


def test_registering_msa_does_not_register_ss():
    """Family selectors should not import or register unrelated implementations."""

    code = """
import sys

from transformers import AutoConfig
from gpn import register_auto_classes

register_auto_classes("msa")
assert type(AutoConfig.for_model("GPNRoFormer")).__name__ == "GPNMSAConfig"
assert "gpn.ss.model" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_unknown_registration_family_is_rejected():
    try:
        register_auto_classes("unknown")  # type: ignore[arg-type]
    except ValueError as error:
        assert "Unknown GPN model family" in str(error)
    else:
        raise AssertionError("Unknown model family was accepted")
