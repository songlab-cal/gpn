import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

import gpn.cli as cli

ROOT = Path(__file__).parents[1]


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return subprocess.run(
        [sys.executable, "-m", "gpn.cli", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )


def test_console_script_is_packaged() -> None:
    scripts = {
        entry_point.name: entry_point.value
        for entry_point in metadata.entry_points(group="console_scripts")
    }

    assert scripts["gpn"] == "gpn.cli:main"


def test_top_level_help_describes_only_maintained_groups() -> None:
    result = _run_cli("--help")

    assert result.returncode == 0, result.stderr
    assert "{ss,msa,star}" in result.stdout
    assert "GPN-MSA inference" in result.stdout
    assert "gpn.ss" not in result.stdout


@pytest.mark.parametrize("group", ("ss", "msa", "star"))
def test_group_help_is_lazy(group: str) -> None:
    script = f"""
import sys
from gpn.cli import main
try:
    main([{group!r}, '--help'])
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'datasets' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_msa_training_is_not_a_command() -> None:
    result = _run_cli("msa", "train")

    assert result.returncode == 2
    assert "invalid choice: 'train'" in result.stderr


@pytest.mark.parametrize(
    ("arguments", "expected_help"),
    (
        (("ss", "train", "--help"), "--dataset_name"),
        (("ss", "vep", "--help"), "variants_path"),
        (("ss", "logits", "--help"), "positions_path"),
        (("ss", "embedding", "--help"), "windows_path"),
        (("msa", "vep", "--help"), "msa_path"),
        (("star", "train", "--help"), "--msa_path"),
        (("star", "vep", "--help"), "--checkpoint-batch-size"),
    ),
)
def test_leaf_help_is_forwarded_offline(
    arguments: tuple[str, ...],
    expected_help: str,
) -> None:
    result = _run_cli(*arguments)

    assert result.returncode == 0, result.stderr
    assert expected_help in result.stdout
    assert result.stdout.startswith(f"usage: gpn {' '.join(arguments[:-1])}")


@pytest.mark.parametrize(
    ("arguments", "expected_module", "expected_command", "expected_forwarded"),
    (
        (
            ("ss", "train", "profile.json"),
            "gpn.ss.train",
            None,
            ["profile.json"],
        ),
        (
            (
                "ss",
                "vep",
                "variants.parquet",
                "--split",
                "validation",
                "--is-file",
            ),
            "gpn.ss.run_vep",
            None,
            ["variants.parquet", "--split", "validation", "--is-file"],
        ),
        (
            ("msa", "logits", "positions.parquet", "msa.zarr", "512", "model"),
            "gpn.msa.inference",
            "logits",
            ["positions.parquet", "msa.zarr", "512", "model"],
        ),
        (
            (
                "star",
                "embedding",
                "windows.parquet",
                "msa",
                "128",
                "model",
                "out.parquet",
                "--no-fp16",
            ),
            "gpn.star.inference",
            "embedding",
            [
                "windows.parquet",
                "msa",
                "128",
                "model",
                "out.parquet",
                "--no-fp16",
            ],
        ),
    ),
)
def test_dispatch_is_lazy_and_forwards_exact_arguments(
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
    expected_module: str,
    expected_command: str | None,
    expected_forwarded: list[str],
) -> None:
    calls = []

    def fake_main(argv, **kwargs):
        calls.append((argv, kwargs, sys.argv[0]))

    imported = []

    def fake_import_module(name):
        imported.append(name)
        return SimpleNamespace(main=fake_main)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)
    original_prog = sys.argv[0]

    assert cli.main(arguments) == 0
    assert imported == [expected_module]
    assert calls[0][0] == expected_forwarded
    assert calls[0][1] == (
        {} if expected_command is None else {"command": expected_command}
    )
    assert calls[0][2].startswith("gpn ")
    assert sys.argv[0] == original_prog


def test_dispatch_does_not_hide_unexpected_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ModuleNotFoundError("unexpected internal import")
    error.name = "unexpected_dependency"

    def fail_import(name):
        raise error

    monkeypatch.setattr(cli.importlib, "import_module", fail_import)

    with pytest.raises(ModuleNotFoundError) as raised:
        cli.main(("ss", "vep", "--help"))

    assert raised.value is error


@pytest.mark.parametrize(
    "module",
    (
        "gpn.ss.run_vep",
        "gpn.msa.inference",
        "gpn.star.inference",
    ),
)
def test_legacy_inference_module_help_remains_available(module: str) -> None:
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
