import sys
from importlib import metadata, util
from pathlib import Path
from types import SimpleNamespace

import pytest

import gpn.cli as cli


def _help(capsys: pytest.CaptureFixture[str], *arguments: str) -> str:
    assert cli.main((*arguments, "--help")) == 0
    return capsys.readouterr().out


def test_console_script_is_packaged() -> None:
    scripts = {
        entry_point.name: entry_point.value
        for entry_point in metadata.entry_points(group="console_scripts")
    }

    assert scripts["gpn"] == "gpn.cli:main"


def test_top_level_help_describes_only_maintained_groups(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _help(capsys)

    for group in ("ss", "msa", "star"):
        assert group in output
    assert "GPN-MSA" in output
    assert "phylo" not in output.lower()


@pytest.mark.parametrize(
    ("group", "commands"),
    (
        ("ss", ("train", "vep", "logits", "embedding")),
        ("msa", ("vep", "logits", "embedding")),
        ("star", ("train", "vep", "logits", "embedding")),
    ),
)
def test_group_help_exposes_the_maintained_surface(
    group: str,
    commands: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _help(capsys, group)

    for command in commands:
        assert command in output


def test_msa_training_is_not_a_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(("msa", "train"))

    assert raised.value.code != 0
    assert "Unknown command" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "expected_help"),
    (
        (("ss", "train", "--help"), "PROFILE"),
        (("ss", "vep", "--help"), "--input-path"),
        (("ss", "logits", "--help"), "--genome-path"),
        (("ss", "embedding", "--help"), "--center-window-size"),
        (("msa", "vep", "--help"), "--msa-path"),
        (("star", "train", "--help"), "PROFILE"),
        (("star", "vep", "--help"), "--checkpoint-batch-size"),
    ),
)
def test_leaf_help_is_generated_from_typed_signatures(
    arguments: tuple[str, ...],
    expected_help: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _help(capsys, *arguments[:-1])

    assert expected_help in output
    if "train" not in arguments:
        assert "--bf16-full-eval" in output
        assert "--torch-compile" in output
        assert "--trainer" not in output
        assert "--is-file" not in output


def test_inference_dispatch_builds_flat_transformers_and_checkpoint_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_vep(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "gpn.ss.inference",
        SimpleNamespace(vep=fake_vep),
    )
    input_path = tmp_path / "variants.parquet"
    input_path.touch()
    output_path = tmp_path / "scores.parquet"

    assert (
        cli.main(
            (
                "ss",
                "vep",
                "--input-path",
                str(input_path),
                "--genome-path",
                "genome.fa",
                "--window-size",
                "512",
                "--model-path",
                "songlab/model",
                "--output-path",
                str(output_path),
                "--per-device-eval-batch-size",
                "16",
                "--checkpoint-batch-size",
                "1000",
                "--checkpoint-revision",
                "inputs-v1",
            )
        )
        == 0
    )

    positional, keyword = calls[0]
    assert positional == (
        str(input_path),
        "genome.fa",
        512,
        "songlab/model",
        output_path,
    )
    assert keyword["is_file"] is True
    assert keyword["training_arguments"].per_device_eval_batch_size == 16
    assert keyword["training_arguments"].bf16_full_eval is False
    assert keyword["training_arguments"].torch_compile is False
    assert keyword["checkpoint_arguments"].checkpoint_batch_size == 1000
    assert keyword["checkpoint_arguments"].checkpoint_revision == "inputs-v1"


def test_inference_input_kind_is_inferred_from_the_path(tmp_path: Path) -> None:
    local_input = tmp_path / "variants.vcf.gz"
    local_input.touch()

    assert cli._input_is_file(str(local_input)) is True
    assert cli._input_is_file("songlab/variants") is False
    with pytest.raises(ValueError, match="Local input file does not exist"):
        cli._input_is_file(str(tmp_path / "missing.parquet"))


def test_invalid_inference_option_combination_has_concise_cli_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = cli.main(
        (
            "ss",
            "vep",
            "--input-path",
            "input",
            "--genome-path",
            "genome",
            "--window-size",
            "512",
            "--model-path",
            "model",
            "--output-path",
            "output",
            "--checkpoint-dir",
            "checkpoints",
        )
    )

    captured = capsys.readouterr()
    assert status == 2
    assert "checkpoint_batch_size" in captured.err
    assert "Traceback" not in captured.err


def test_training_dispatch_accepts_one_yaml_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "gpn.star.train",
        SimpleNamespace(main=calls.append),
    )

    assert cli.main(("star", "train", "profile.yaml")) == 0
    assert calls == [Path("profile.yaml")]


@pytest.mark.parametrize(
    "module",
    (
        "gpn.ss.run_vep",
        "gpn.ss.get_logits",
        "gpn.ss.get_embeddings",
        "gpn.msa.vep",
        "gpn.star.vep",
    ),
)
def test_retired_operation_module_paths_are_absent(module: str) -> None:
    assert util.find_spec(module) is None
