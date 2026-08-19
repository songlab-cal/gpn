import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from gpn import register_auto_classes

pytestmark = pytest.mark.published_models

FIXTURE_DIR = Path(__file__).parent / "fixtures"
BASELINE = json.loads((FIXTURE_DIR / "published_model_baseline.json").read_text())
ALIGNMENT_PATH = FIXTURE_DIR / "hg38_chr6_31575665_31575793_multiz100way.npz"
DNA_VOCAB = "-ACGT?"
MASK_TOKEN_ID = DNA_VOCAB.index("?")
NUCLEOTIDE_INDICES = [DNA_VOCAB.index(nucleotide) for nucleotide in "ACGT"]


@pytest.fixture(scope="module", autouse=True)
def register_models() -> None:
    register_auto_classes("gpn", "phylogpn", "star")


def load_kwargs(record: dict) -> dict:
    kwargs = {"revision": record["revision"]}
    if cache_dir := os.environ.get("HF_HOME"):
        kwargs["cache_dir"] = cache_dir
    return kwargs


def assert_matches_baseline(actual: torch.Tensor, expected) -> None:
    torch.testing.assert_close(
        actual.detach().cpu().to(torch.float32),
        torch.tensor(expected, dtype=torch.float32),
        rtol=1e-4,
        atol=1e-4,
    )


def download_dataset_file(record: dict, filename: str) -> Path:
    input_record = record["expected"]["input"]
    kwargs = {
        "repo_type": "dataset",
        "revision": input_record["dataset_revision"],
    }
    if cache_dir := os.environ.get("HF_HOME"):
        kwargs["cache_dir"] = cache_dir
    return Path(hf_hub_download(input_record["dataset_id"], filename, **kwargs))


def test_published_gpn_likelihood() -> None:
    record = BASELINE["models"]["gpn"]
    expected = record["expected"]
    tokenizer = AutoTokenizer.from_pretrained(record["model_id"], **load_kwargs(record))
    model = AutoModelForMaskedLM.from_pretrained(
        record["model_id"],
        **load_kwargs(record),
    ).eval()
    input_ids = tokenizer(
        expected["input"]["sequence"],
        return_tensors="pt",
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    position = expected["input"]["masked_position_zero_based"]
    input_ids[0, position] = tokenizer.mask_token_id

    with torch.inference_mode():
        output = model(input_ids=input_ids).logits
    indices = [tokenizer.get_vocab()[nucleotide] for nucleotide in "acgt"]
    assert_matches_baseline(output[0, position, indices], expected["logits"])


def test_published_gpn_msa_likelihood() -> None:
    record = BASELINE["models"]["gpn_msa"]
    expected = record["expected"]
    with np.load(ALIGNMENT_PATH, allow_pickle=False) as alignment:
        tokens = alignment["gpn_msa_tokens"].astype(np.int64)
    msa = torch.from_numpy(tokens).unsqueeze(0)
    input_ids = msa[:, :, 0].clone()
    aux_features = msa[:, :, 1:]
    position = expected["input"]["masked_position_zero_based"]
    input_ids[0, position] = MASK_TOKEN_ID
    model = AutoModelForMaskedLM.from_pretrained(
        record["model_id"],
        **load_kwargs(record),
    ).eval()

    with torch.inference_mode():
        output = model(input_ids=input_ids, aux_features=aux_features).logits
    assert_matches_baseline(
        output[0, position, NUCLEOTIDE_INDICES],
        expected["logits"],
    )


def test_published_gpn_star_likelihood() -> None:
    record = BASELINE["models"]["gpn_star"]
    expected = record["expected"]
    with np.load(ALIGNMENT_PATH, allow_pickle=False) as alignment:
        tokens = alignment["gpn_star_v100_tokens"].astype(np.int64)
    msa = torch.from_numpy(tokens).unsqueeze(0)
    input_ids = msa[:, :, :1].clone()
    position = expected["input"]["masked_position_zero_based"]
    input_ids[0, position, 0] = MASK_TOKEN_ID
    model = AutoModelForMaskedLM.from_pretrained(
        record["model_id"],
        **load_kwargs(record),
    ).eval()

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            source_ids=msa,
            target_species=np.array([[0]], dtype=int),
        ).logits
    logits = output[0, position, 0, NUCLEOTIDE_INDICES]
    assert_matches_baseline(logits, expected["logits"])

    import pandas as pd

    table_path = Path(
        hf_hub_download(
            record["model_id"],
            expected["calibration"]["filename"],
            **load_kwargs(record),
        )
    )
    assert (
        hashlib.sha256(table_path.read_bytes()).hexdigest()
        == (expected["calibration"]["sha256"])
    )
    table = pd.read_parquet(table_path).set_index("pentanuc_mut")
    pentanucleotide = expected["input"]["pentanucleotide"]
    raw_llr = logits[1:] - logits[0]
    for index, alternate in enumerate("CGT"):
        neutral_mean = table.loc[
            f"{pentanucleotide}_{alternate}",
            "llr_neutral_mean",
        ]
        assert neutral_mean == pytest.approx(
            expected["llr_neutral_mean"][alternate],
            abs=1e-7,
        )
        calibrated = raw_llr[index].item() - neutral_mean
        assert calibrated == pytest.approx(
            expected["llr_calibrated"][alternate],
            abs=1e-4,
        )


def test_published_phylogpn_likelihood_without_remote_code() -> None:
    record = BASELINE["models"]["phylogpn"]
    expected = record["expected"]
    tokenizer = AutoTokenizer.from_pretrained(record["model_id"], **load_kwargs(record))
    model = AutoModel.from_pretrained(record["model_id"], **load_kwargs(record)).eval()
    pad_size = expected["input"]["symmetric_padding_each_side"]
    sequences = expected["input"]["sequences"]
    padded = [
        tokenizer.pad_token * pad_size + sequence + tokenizer.pad_token * pad_size
        for sequence in sequences
    ]
    input_ids = tokenizer(padded, return_tensors="pt", padding=True)["input_ids"]

    with torch.inference_mode():
        output = model(input_ids=input_ids)
    first_logits = torch.stack(
        [output[nucleotide][0, : len(sequences[0])] for nucleotide in "ACGT"],
        dim=-1,
    )
    assert_matches_baseline(first_logits, expected["first_sequence_logits"])


def test_published_sorghum_expression_output() -> None:
    record = BASELINE["models"]["sorghum_expression"]
    expected = record["expected"]
    tokenizer = AutoTokenizer.from_pretrained(record["model_id"], **load_kwargs(record))
    model = AutoModelForSequenceClassification.from_pretrained(
        record["model_id"],
        **load_kwargs(record),
    ).eval()
    input_ids = tokenizer(
        expected["input"]["sequence"],
        return_tensors="pt",
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]

    with torch.inference_mode():
        output = model(input_ids=input_ids).logits[0]
    assert_matches_baseline(output, expected["predicted_log1p_expression"])

    import pandas as pd

    labels_path = download_dataset_file(record, expected["input"]["labels_file"])
    test_path = download_dataset_file(record, expected["input"]["test_file"])
    assert (
        hashlib.sha256(labels_path.read_bytes()).hexdigest()
        == (expected["input"]["labels_sha256"])
    )
    assert (
        hashlib.sha256(test_path.read_bytes()).hexdigest()
        == (expected["input"]["test_sha256"])
    )
    assert labels_path.read_text().splitlines() == expected["label_order"]

    row = pd.read_parquet(test_path).iloc[expected["input"]["row_index"]]
    assert str(row["chrom"]) == expected["input"]["chrom"]
    assert int(row["start"]) == expected["input"]["start_zero_based"]
    assert int(row["end"]) == expected["input"]["end_zero_based_exclusive"]
    assert row["strand"] == expected["input"]["strand"]
    assert row["seq"] == expected["input"]["sequence"]
