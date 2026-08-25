import ast
import hashlib
import importlib.util
import json
import re
from datetime import date
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).parents[1]
COLABS = ROOT / "colabs"
MODEL_DEMOS = {
    "gpn_demo.ipynb": "gpn",
    "phylogpn_demo.ipynb": "phylogpn",
    "gpn_star_demo.ipynb": "gpn_star",
}
WORKFLOWS = {"gpn_star_precomputed_scores.ipynb"}
NOTEBOOKS = set(MODEL_DEMOS) | WORKFLOWS
BASELINE = json.loads(
    (ROOT / "tests" / "fixtures" / "published_model_baseline.json").read_text()
)
FORBIDDEN_TEXT = (
    "/accounts/",
    "/scratch/",
    "os.system",
    "import gpn.model",
    "import gpn.star.model",
    "trust_remote_code",
)
BASELINE_SHA256 = hashlib.sha256(
    (ROOT / "tests" / "fixtures" / "published_model_baseline.json").read_bytes()
).hexdigest()


def _load_notebook(name):
    return json.loads((COLABS / name).read_text())


def _load_refresh_notebooks(monkeypatch):
    monkeypatch.syspath_prepend(ROOT)
    spec = importlib.util.spec_from_file_location(
        "gpn_refresh_notebooks", ROOT / "docs" / "refresh_notebooks.py"
    )
    assert spec is not None
    assert spec.loader is not None
    refresh_notebooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refresh_notebooks)
    return refresh_notebooks


def _cell(notebook, cell_id):
    return next(cell for cell in notebook["cells"] if cell["id"] == cell_id)


def _plain_output(notebook, cell_id):
    parts = []
    for output in _cell(notebook, cell_id).get("outputs", []):
        plain = output.get("data", {}).get("text/plain", "")
        parts.append("".join(plain) if isinstance(plain, list) else plain)
    return "\n".join(parts)


def _nucleotide_table(text):
    rows = re.findall(
        r"^\s*\d+\s+([ACGT])\s+(-?\d+\.\d+)\s+(\d+\.\d+)",
        text,
        flags=re.MULTILINE,
    )
    return {
        base: (float(logit), float(probability)) for base, logit, probability in rows
    }


def _base_vectors(text):
    vectors = re.findall(r"'([ACGT])': tensor\(\[([^]]+)\]\)", text)
    return {
        base: [float(value) for value in values.split(",")] for base, values in vectors
    }


def test_canonical_notebook_set_is_maintained():
    actual = {path.name for path in COLABS.glob("*.ipynb")}
    validation_dates = {
        _load_notebook(name)["metadata"]["gpn"]["last_scientific_validation"]
        for name in MODEL_DEMOS
    }

    assert actual == NOTEBOOKS
    assert len(validation_dates) == 1
    date.fromisoformat(validation_dates.pop())


def test_demos_are_portable_static_notebooks():
    for name, family in MODEL_DEMOS.items():
        path = COLABS / name
        serialized = path.read_text()
        notebook = json.loads(serialized)
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        assert path.stat().st_size < 2 * 1024 * 1024
        assert notebook["nbformat"] == 4
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        assert (
            notebook["metadata"]["gpn"]["model_id"]
            == (BASELINE["models"][family]["model_id"])
        )
        assert (
            notebook["metadata"]["gpn"]["model_revision"]
            == (BASELINE["models"][family]["revision"])
        )
        assert notebook["metadata"]["gpn"]["baseline_sha256"] == BASELINE_SHA256
        assert set(notebook["metadata"]["gpn"]["output_environment"]) == {
            "device",
            "dtype",
            "gpn",
            "python",
            "torch",
            "transformers",
        }
        assert "colab.research.google.com" not in source
        for forbidden in FORBIDDEN_TEXT:
            assert forbidden not in serialized

        install_cells = [
            cell
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "pip install" in "".join(cell["source"])
        ]
        assert len(install_cells) == 1
        assert "skip-execution" in install_cells[0]["metadata"]["tags"]
        assert "remove-input" in install_cells[0]["metadata"]["tags"]

        executable_cells = [
            cell
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "skip-execution" not in cell.get("metadata", {}).get("tags", [])
        ]
        for cell in executable_cells:
            ast.parse("".join(cell["source"]), filename=f"{path}:{cell['id']}")

        error_outputs = [
            output
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert error_outputs == []
        committed_outputs = [
            output for cell in notebook["cells"] for output in cell.get("outputs", [])
        ]
        assert committed_outputs
        if family in {"gpn", "gpn_star"}:
            assert any(
                "image/png" in output.get("data", {}) for output in committed_outputs
            )
        stderr_outputs = [
            output
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
            if output.get("output_type") == "stream" and output.get("name") == "stderr"
        ]
        assert stderr_outputs == []


def test_precomputed_score_workflow_is_portable_and_pinned():
    path = COLABS / "gpn_star_precomputed_scores.ipynb"
    serialized = path.read_text()
    notebook = json.loads(serialized)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["metadata"]["gpn"] == {
        "benchmark_id": "songlab/omim_traitgym",
        "benchmark_revision": "9317562efb8c61f31bb5fc62a19f731b2f8b4384",
        "dataset_id": "songlab/gpn-star-scores",
        "dataset_revision": "5c799b2ec6aa089f0caa8294ae72adb4510f81ae",
    }
    assert '"hf://datasets/songlab/omim_traitgym@"' in source
    assert "9317562efb8c61f31bb5fc62a19f731b2f8b4384" in source
    assert '"hf://datasets/songlab/gpn-star-scores@"' in source
    assert "5c799b2ec6aa089f0caa8294ae72adb4510f81ae" in source
    assert 'KEYS = ["chrom", "pos", "ref", "alt"]' in source
    assert "average_precision_score" in source
    assert "global AUPRC" in source
    assert "variants.height == 3_380" in source
    assert 'get_column("label").sum() == 338' in source
    assert 'annotated.get_column("label").to_numpy()' in source
    assert 'annotated.get_column("effect_score").to_numpy()' in source
    assert "partition_by" not in source
    assert "group_by" not in source
    assert '-pl.col("llr_calibrated")' in source
    assert 'filter(pl.col("pos").is_in(positions))' in source
    for forbidden in FORBIDDEN_TEXT + (
        "AutoModel",
        "AutoTokenizer",
        "from_pretrained",
        "hf_hub_download",
        "snapshot_download",
        "predictions/GPN-Star-M447.parquet",
        "all.zarr",
        "ukb_finemapped",
    ):
        assert forbidden not in serialized
    assert "import gpn" not in source
    assert "whole-genome" in source
    assert _cell(notebook, "scores-input")["outputs"]
    assert _cell(notebook, "scores-query")["outputs"]
    auprc_outputs = _cell(notebook, "scores-auprc")["outputs"]
    assert auprc_outputs == [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": ["Genome-wide join global AUPRC: 0.7644\n"],
        }
    ]
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        if "skip-execution" in cell.get("metadata", {}).get("tags", []):
            continue
        ast.parse("".join(cell["source"]), filename=f"{path}:{cell['id']}")


def test_gpn_committed_outputs_match_approved_baseline():
    notebook = _load_notebook("gpn_demo.ipynb")
    expected = BASELINE["models"]["gpn"]["expected"]
    observed = _nucleotide_table(_plain_output(notebook, "gpn-mlm"))

    assert list(observed) == expected["nucleotide_order"]
    for index, base in enumerate(expected["nucleotide_order"]):
        assert observed[base][0] == pytest.approx(expected["logits"][index], abs=1e-4)
        assert observed[base][1] == pytest.approx(
            expected["probabilities"][index], abs=1e-4
        )


def test_gpn_star_committed_outputs_match_approved_baseline():
    notebook = _load_notebook("gpn_star_demo.ipynb")
    expected = BASELINE["models"]["gpn_star"]["expected"]
    observed = _nucleotide_table(_plain_output(notebook, "star-model"))

    assert list(observed) == expected["nucleotide_order"]
    for index, base in enumerate(expected["nucleotide_order"]):
        assert observed[base][0] == pytest.approx(expected["logits"][index], abs=1e-4)
        assert observed[base][1] == pytest.approx(
            expected["probabilities"][index], abs=1e-4
        )

    raw_rows = re.findall(
        r"^\s*\d+\s+A>([CGT])\s+[CGT]\s+(-?\d+\.\d+)",
        _plain_output(notebook, "star-llr"),
        flags=re.MULTILINE,
    )
    observed_raw = {alternate: float(value) for alternate, value in raw_rows}
    assert observed_raw == pytest.approx(expected["llr_alt_minus_ref"], abs=1e-4)

    calibrated_rows = re.findall(
        r"^\s*\d+\s+A>([CGT])\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
        _plain_output(notebook, "star-calibration"),
        flags=re.MULTILINE,
    )
    observed_calibrated = {
        alternate: float(calibrated)
        for alternate, _raw, _neutral, calibrated in calibrated_rows
    }
    assert observed_calibrated == pytest.approx(expected["llr_calibrated"], abs=1e-4)

    source = "".join(_cell(notebook, "star-fixture-download")["source"])
    fixture_sha256 = BASELINE["alignment_fixture"]["sha256"]
    assert "/main/tests/fixtures/" not in source
    assert fixture_sha256 in source
    assert notebook["metadata"]["gpn"]["fixture_sha256"] == fixture_sha256


def test_phylogpn_committed_outputs_match_approved_baseline():
    notebook = _load_notebook("phylogpn_demo.ipynb")
    expected = BASELINE["models"]["phylogpn"]["expected"]

    observed_logits = _base_vectors(_plain_output(notebook, "phylo-rates"))
    observed_probabilities = _base_vectors(
        _plain_output(notebook, "phylo-probabilities")
    )
    for index, base in enumerate(expected["nucleotide_order"]):
        expected_logits = [row[index] for row in expected["first_sequence_logits"]]
        expected_probabilities = [
            row[index] for row in expected["first_sequence_probabilities"]
        ]
        assert observed_logits[base] == pytest.approx(expected_logits, abs=1e-4)
        assert observed_probabilities[base] == pytest.approx(
            expected_probabilities, abs=1e-4
        )

    llr_match = re.search(
        r"tensor\((-?\d+\.\d+)\)", _plain_output(notebook, "phylo-llr")
    )
    assert llr_match is not None
    assert float(llr_match.group(1)) == pytest.approx(
        expected["c_to_t_llr_position_one_zero_based"], abs=1e-4
    )


def test_sphinx_notebook_copy_is_deterministic(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "gpn_prepare_notebooks", ROOT / "docs" / "prepare_notebooks.py"
    )
    assert spec is not None
    assert spec.loader is not None
    prepare_notebooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_notebooks)

    monkeypatch.setattr(prepare_notebooks, "DESTINATION", tmp_path)
    prepare_notebooks.main()

    for name in NOTEBOOKS:
        assert (tmp_path / name).read_bytes() == (COLABS / name).read_bytes()


def test_refresh_consumes_kernel_provenance_without_committing_helper_cell(
    monkeypatch,
):
    refresh_notebooks = _load_refresh_notebooks(monkeypatch)

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("Quick start"),
            refresh_notebooks._provenance_cell(),
        ],
        metadata={"gpn": {}},
    )
    payload = {
        "last_scientific_validation": "2026-08-23",
        "model_id": "songlab/example-model",
        "model_revision": "0123456789abcdef",
        "output_environment": {
            "device": "cuda:0",
            "dtype": "torch.float32",
            "gpn": "0.9.0",
            "python": "3.13.2",
            "torch": "2.13.0+cu130",
            "transformers": "5.15.0",
        },
    }
    notebook.cells[-1]["outputs"] = [
        nbformat.v4.new_output(
            output_type="stream",
            name="stdout",
            text=(
                refresh_notebooks._PROVENANCE_PREFIX
                + json.dumps(payload, sort_keys=True)
                + "\n"
            ),
        )
    ]

    observed = refresh_notebooks._consume_provenance(notebook)
    refresh_notebooks._update_provenance(notebook, observed)

    assert [cell["cell_type"] for cell in notebook.cells] == ["markdown"]
    assert notebook.metadata["gpn"] == payload


def test_refresh_provenance_cell_records_loaded_model_identity(monkeypatch, capsys):
    refresh_notebooks = _load_refresh_notebooks(monkeypatch)

    class FakeModel:
        config = type(
            "Config",
            (),
            {
                "_commit_hash": "0123456789abcdef",
                "_name_or_path": "songlab/example-model",
            },
        )()

        @staticmethod
        def parameters():
            parameter = type(
                "Parameter", (), {"device": "cuda:0", "dtype": "torch.float32"}
            )()
            return iter([parameter])

    namespace = {
        "MODEL_ID": "songlab/example-model",
        "MODEL_REVISION": "0123456789abcdef",
        "model": FakeModel(),
    }
    exec(refresh_notebooks._provenance_cell().source, namespace)
    output = capsys.readouterr().out.strip()
    payload = json.loads(output.removeprefix(refresh_notebooks._PROVENANCE_PREFIX))

    assert payload["model_id"] == namespace["MODEL_ID"]
    assert payload["model_revision"] == namespace["MODEL_REVISION"]


def test_refresh_provenance_cell_rejects_revision_mismatch(monkeypatch):
    refresh_notebooks = _load_refresh_notebooks(monkeypatch)

    class FakeModel:
        config = type(
            "Config",
            (),
            {
                "_commit_hash": "resolved-revision",
                "_name_or_path": "songlab/example-model",
            },
        )()

        @staticmethod
        def parameters():
            return iter(())

    namespace = {
        "MODEL_ID": "songlab/example-model",
        "MODEL_REVISION": "requested-revision",
        "model": FakeModel(),
    }
    with pytest.raises(RuntimeError, match="resolved-revision != requested-revision"):
        exec(refresh_notebooks._provenance_cell().source, namespace)


def test_refresh_provenance_cell_rejects_model_identity_mismatch(monkeypatch):
    refresh_notebooks = _load_refresh_notebooks(monkeypatch)

    class FakeModel:
        config = type(
            "Config",
            (),
            {
                "_commit_hash": "0123456789abcdef",
                "_name_or_path": "songlab/loaded-model",
            },
        )()

        @staticmethod
        def parameters():
            return iter(())

    namespace = {
        "MODEL_ID": "songlab/declared-model",
        "MODEL_REVISION": "0123456789abcdef",
        "model": FakeModel(),
    }
    with pytest.raises(
        RuntimeError, match="songlab/loaded-model != songlab/declared-model"
    ):
        exec(refresh_notebooks._provenance_cell().source, namespace)


def test_refresh_provenance_cell_requires_resolved_revision(monkeypatch):
    refresh_notebooks = _load_refresh_notebooks(monkeypatch)

    class FakeModel:
        config = type(
            "Config",
            (),
            {"_commit_hash": None, "_name_or_path": "songlab/example-model"},
        )()

        @staticmethod
        def parameters():
            return iter(())

    namespace = {
        "MODEL_ID": "songlab/example-model",
        "MODEL_REVISION": "0123456789abcdef",
        "model": FakeModel(),
    }
    with pytest.raises(RuntimeError, match="resolved commit hash"):
        exec(refresh_notebooks._provenance_cell().source, namespace)


def test_refresh_rejects_unexpected_text_beside_expected_stderr(monkeypatch):
    refresh_notebooks = _load_refresh_notebooks(monkeypatch)

    expected = (
        "[transformers] ConvNetModel LOAD REPORT songlab/gpn-brassicales "
        "cls.decoder.{0, 2, 3}.weight UNEXPECTED "
        "can be ignored when loading from different task/architecture"
    )
    clean_cell = nbformat.v4.new_code_cell(
        outputs=[
            nbformat.v4.new_output(output_type="stream", name="stderr", text=expected)
        ]
    )
    refresh_notebooks._remove_expected_stderr(
        clean_cell, notebook_name="quick_start.ipynb"
    )
    assert clean_cell.outputs == []

    mixed_cell = nbformat.v4.new_code_cell(
        outputs=[
            nbformat.v4.new_output(
                output_type="stream",
                name="stderr",
                text=f"{expected}\nUnexpected warning",
            )
        ]
    )
    with pytest.raises(RuntimeError, match="Unexpected warning"):
        refresh_notebooks._remove_expected_stderr(
            mixed_cell, notebook_name="quick_start.ipynb"
        )
