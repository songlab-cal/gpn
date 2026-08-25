import argparse
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

import numpy as np
import pytest
import torch

from gpn.scoring import log_likelihood_ratio, nucleotide_probabilities

FIXTURE_DIR = Path(__file__).parent / "fixtures"
BASELINE_PATH = FIXTURE_DIR / "published_model_baseline.json"
ALIGNMENT_PATH = FIXTURE_DIR / "hg38_chr6_31575665_31575793_multiz100way.npz"
LABELS_SHA256 = "e9ae64092ac6b05abf012b15205376c2383b002e63a635caf6caa249575b7b19"
TEST_SHA256 = "baa4ba0e62175fc71391f2ceca753dbc612df34fc7d498f5cf12c9939ec78701"
SPECIES_99_SHA256 = "ebf42541d808853468b9ce6087409a909d5046c88e75b5b767c567c59fac2399"
SPECIES_89_SHA256 = "0f2d9a52aabcd45646b55453c137a072611c6d826f968914c31f8ba82f7918af"
SOURCE_SLICE_SHA256 = "fb3271e52670bcfe44347b6e7051c9ef8f15aaf2f37b92848da346c28b54ca8e"

MODEL_REVISIONS = {
    "gpn": ("songlab/gpn-brassicales", "eb9c35d0d18571abe84390d22e74f2b21d319ce3"),
    "gpn_msa": (
        "songlab/gpn-msa-sapiens",
        "4a7d4f75449cb2abd560b2af024d76f99233c6db",
    ),
    "gpn_star": (
        "songlab/gpn-star-hg38-v100-200m",
        "0c949f132d35619a3eb188b402848c998a3313ae",
    ),
    "phylogpn": (
        "songlab/PhyloGPN",
        "3556db4c469e67d25f0f7a0a6653b48be3eebf51",
    ),
    "sorghum_expression": (
        "songlab/gpn-brassicales-gxa-sorghum-v1",
        "53209151b497d4840d50526d44c0460b6e6768b7",
    ),
}

NOTEBOOK_LOGITS = {
    "gpn": [4.8509, -2.3441, -1.7025, -1.5116],
    "gpn_msa": [6.0621, -1.5262, -0.6878, -0.5625],
    "gpn_star": [7.7062, -1.9220, -0.6912, -2.6507],
}

PHYLOGPN_PUBLISHED_LOGITS = [
    [1.6576689, 1.1715941, -0.1791951, 4.0944095],
    [4.0177917, 0.4637241, 1.1368530, 1.2592138],
    [1.0111851, 0.9405909, -0.0489903, 3.8542891],
    [3.9505949, 0.4236194, 0.8729535, 1.4166427],
    [3.7107053, 0.8936814, 0.3608846, 1.8472843],
    [3.6846304, 0.7399911, 0.7772030, 1.4588658],
]


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text())


def test_alignment_fixture_is_small_and_self_consistent(baseline):
    fixture_metadata = baseline["alignment_fixture"]
    digest = hashlib.sha256(ALIGNMENT_PATH.read_bytes()).hexdigest()
    assert digest == fixture_metadata["sha256"]
    assert ALIGNMENT_PATH.stat().st_size < 50_000
    assert fixture_metadata["source"]["raw_slice_sha256"] == SOURCE_SLICE_SHA256
    assert fixture_metadata["source"] == {
        "kind": "preexisting_local_zarr",
        "public_archive": "99.zarr.tar.gz",
        "public_archive_sha256": (
            "4dad7da04db9c804032c0c4c7bbefea58f694fc911e962d28c8df87f356ce4ad"
        ),
        "public_archive_size_bytes": 42_269_901_437,
        "public_dataset_id": "songlab/multiz100way-pigz",
        "public_dataset_revision": "6a9d42a35e7debbba845979dea6064f14d5cb3f9",
        "raw_slice_sha256": SOURCE_SLICE_SHA256,
    }
    assert fixture_metadata["species_99_sha256"] == SPECIES_99_SHA256
    assert fixture_metadata["species_89_sha256"] == SPECIES_89_SHA256

    with np.load(ALIGNMENT_PATH, allow_pickle=False) as alignment:
        msa = alignment["gpn_msa_tokens"]
        star = alignment["gpn_star_v100_tokens"]

    assert msa.shape == (128, 90)
    assert star.shape == (128, 100)
    assert msa.dtype == star.dtype == np.uint8
    np.testing.assert_array_equal(
        msa[:3, :3],
        [[1, 1, 0], [2, 3, 0], [1, 1, 1]],
    )
    np.testing.assert_array_equal(
        msa[-3:, :3],
        [[4, 4, 4], [2, 4, 2], [2, 2, 2]],
    )
    np.testing.assert_array_equal(msa[76:79, 0], [1, 4, 3])
    np.testing.assert_array_equal(star[76:79, 0], [1, 4, 3])
    np.testing.assert_array_equal(msa[:, 0], star[:, 0])

    full_species = fixture_metadata["gpn_star_v100_species"]
    reduced_species = fixture_metadata["gpn_msa_species"]
    subset_indices = [full_species.index(species) for species in reduced_species]
    assert subset_indices == [0, *range(11, 100)]
    np.testing.assert_array_equal(msa, star[:, subset_indices])
    assert (
        hashlib.sha256(("\n".join(full_species[1:]) + "\n").encode()).hexdigest()
        == SPECIES_99_SHA256
    )
    assert (
        hashlib.sha256(("\n".join(reduced_species[1:]) + "\n").encode()).hexdigest()
        == SPECIES_89_SHA256
    )


def test_model_revisions_are_immutable_and_named(baseline):
    assert baseline["schema_version"] == 3
    for name, (model_id, revision) in MODEL_REVISIONS.items():
        record = baseline["models"][name]
        assert record["model_id"] == model_id
        assert record["revision"] == revision
        assert len(revision) == 40


def test_model_baseline_has_reproducible_source_and_generation_provenance(baseline):
    environment = baseline["environment"]
    assert environment["gpn_version"] == "0.9.0"
    assert environment["gpn_source_commit"] == (
        "1dc8f776953d8f74eb0a3ac0a277a72ca38581a2"
    )
    assert environment["gpn_source_tree"] == (
        "949c3f16dbffb12c1eacca17c165ba6a494e7a52"
    )
    assert environment["working_tree_changes"] is False
    generation = baseline["generation"]
    assert "regenerate_published_model_baseline.py" in generation["command"]
    assert "--output /tmp/" in generation["command"]
    assert generation["approval_reason"]
    assert generation["generator_sha256"] == (
        "82e5a3d43385a39ee10f5bef8948bb7026482842decd51ef169d7c5527ce7ca5"
    )
    assert generation["review_candidate_sha256"] == (
        "972b893cf2b98a6de538399cc1cd98f3252386762490fc8f87bb7dcddfa5c26f"
    )


def test_baseline_generator_rejects_a_gpn_import_outside_the_checkout(
    monkeypatch,
    tmp_path,
):
    generator_path = FIXTURE_DIR / "regenerate_published_model_baseline.py"
    spec = importlib.util.spec_from_file_location("baseline_generator", generator_path)
    assert spec is not None
    assert spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    generator._require_local_gpn_source()
    monkeypatch.setattr(
        generator.gpn, "__file__", str(tmp_path / "gpn" / "__init__.py")
    )
    with pytest.raises(SystemExit, match="imported gpn came from"):
        generator._require_local_gpn_source()


def test_baseline_generator_records_reproducible_quoted_command(tmp_path):
    generator_path = FIXTURE_DIR / "regenerate_published_model_baseline.py"
    spec = importlib.util.spec_from_file_location("baseline_generator", generator_path)
    assert spec is not None
    assert spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    template = generator.ROOT / "tests" / "fixtures" / "custom template.json"
    output = tmp_path / "candidate; review.json"
    args = argparse.Namespace(
        template=template,
        output=output,
        approval_reason="Changed calibration; preserve this text literally",
    )

    assert shlex.split(generator._generation_command(args)) == [
        "uv",
        "run",
        "--extra",
        "inference",
        "python",
        "tests/fixtures/regenerate_published_model_baseline.py",
        "--template",
        "tests/fixtures/custom template.json",
        "--output",
        str(output),
        "--approval-reason",
        args.approval_reason,
    ]


@pytest.mark.parametrize("model_name", NOTEBOOK_LOGITS)
def test_likelihoods_reproduce_quick_start_outputs(baseline, model_name):
    expected = baseline["models"][model_name]["expected"]
    logits = torch.tensor(expected["logits"])

    torch.testing.assert_close(
        logits,
        torch.tensor(NOTEBOOK_LOGITS[model_name]),
        rtol=0,
        atol=5e-4,
    )
    torch.testing.assert_close(
        nucleotide_probabilities(logits),
        torch.tensor(expected["probabilities"]),
        rtol=1e-6,
        atol=1e-7,
    )
    for alternate_index, alternate in enumerate("CGT", start=1):
        actual = log_likelihood_ratio(logits, 0, alternate_index)
        assert actual.item() == pytest.approx(
            expected["llr_alt_minus_ref"][alternate],
            abs=1e-6,
        )


def test_gpn_star_calibration_matches_quick_start(baseline):
    expected = baseline["models"]["gpn_star"]["expected"]
    assert expected["input"]["pentanucleotide"] == "CCATG"
    logits = torch.tensor(expected["logits"])
    raw_llr = logits[1:] - logits[0]
    neutral_means = torch.tensor(list(expected["llr_neutral_mean"].values()))
    calibrated = raw_llr - neutral_means
    torch.testing.assert_close(
        calibrated,
        torch.tensor(list(expected["llr_calibrated"].values())),
        rtol=0,
        atol=1e-6,
    )
    assert calibrated.tolist() == pytest.approx(
        [-8.196719, -7.101466, -8.904929],
        abs=5e-4,
    )
    assert expected["calibration"] == {
        "filename": "calibration_table/llr.parquet",
        "sha256": "162ed3d1bc05208a955f0440c4a010c82213632c41870307e701ba011775a5e4",
    }


def test_phylogpn_likelihood_uses_current_published_weights(baseline):
    expected = baseline["models"]["phylogpn"]["expected"]
    logits = torch.tensor(expected["first_sequence_logits"])
    torch.testing.assert_close(
        logits,
        torch.tensor(PHYLOGPN_PUBLISHED_LOGITS),
        rtol=0,
        atol=5e-4,
    )
    torch.testing.assert_close(
        nucleotide_probabilities(logits),
        torch.tensor(expected["first_sequence_probabilities"]),
        rtol=1e-6,
        atol=1e-7,
    )
    c_to_t = log_likelihood_ratio(logits[1], 1, 3)
    assert c_to_t.item() == pytest.approx(0.7954897, abs=5e-4)
    assert c_to_t.item() == pytest.approx(
        expected["c_to_t_llr_position_one_zero_based"],
        abs=1e-6,
    )


def test_sorghum_expression_output_has_published_label_order(baseline):
    expected = baseline["models"]["sorghum_expression"]["expected"]
    values = np.asarray(expected["predicted_log1p_expression"])
    labels = expected["label_order"]

    input_record = expected["input"]
    assert input_record["dataset_revision"] == (
        "0545539b3229946b90c1073c99a97bfb9f95cd83"
    )
    assert input_record["labels_sha256"] == LABELS_SHA256
    assert input_record["test_sha256"] == TEST_SHA256
    labels_content = ("\n".join(labels) + "\n").encode()
    assert hashlib.sha256(labels_content).hexdigest() == LABELS_SHA256
    assert len(input_record["sequence"]) == 512
    assert len(labels) == len(set(labels)) == 26
    assert values.shape == (26,)
    assert np.isfinite(values).all()
    assert (values >= 0).all()
