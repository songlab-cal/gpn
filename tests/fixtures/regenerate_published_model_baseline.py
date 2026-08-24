"""Regenerate published-model numerical baselines from immutable Hub revisions.

This intentionally downloads model artifacts but reads the MSA only from the tiny
checked-in fixture. It writes a review candidate and never overwrites the canonical
baseline in place.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import gpn
from gpn import register_auto_classes

ROOT = Path(__file__).parents[2]
FIXTURE_DIR = Path(__file__).parent
CANONICAL_BASELINE = FIXTURE_DIR / "published_model_baseline.json"
ALIGNMENT_PATH = FIXTURE_DIR / "hg38_chr6_31575665_31575793_multiz100way.npz"
DNA_VOCAB = "-ACGT?"
NUCLEOTIDE_INDICES = [DNA_VOCAB.index(nucleotide) for nucleotide in "ACGT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=CANONICAL_BASELINE,
        help="Baseline whose pinned inputs and revisions should be reused",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Review-candidate JSON path (the canonical fixture is refused)",
    )
    parser.add_argument(
        "--approval-reason",
        required=True,
        help="Scientific reason for proposing changed expectations",
    )
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_committed_source() -> tuple[str, str]:
    working_tree_changes = _git("status", "--porcelain")
    if working_tree_changes:
        raise SystemExit(
            "Refusing to generate scientific expectations from working-tree changes; "
            "use a clean checkout of the exact committed source."
        )
    return _git("rev-parse", "HEAD"), _git("rev-parse", "HEAD^{tree}")


def _require_local_gpn_source() -> None:
    imported_package = Path(gpn.__file__).resolve()
    expected_package = (ROOT / "src" / "gpn").resolve()
    if not imported_package.is_relative_to(expected_package):
        raise SystemExit(
            "Refusing to bind outputs to this repository because imported gpn came "
            f"from {imported_package}, not {expected_package}."
        )


def _portable_path(path: Path) -> str:
    """Render checkout-local paths relative to the repository root."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _generation_command(args: argparse.Namespace) -> str:
    return shlex.join(
        [
            "uv",
            "run",
            "--extra",
            "inference",
            "python",
            "tests/fixtures/regenerate_published_model_baseline.py",
            "--template",
            _portable_path(args.template),
            "--output",
            _portable_path(args.output),
            "--approval-reason",
            args.approval_reason,
        ]
    )


def _load_kwargs(record: dict) -> dict[str, str]:
    return {"revision": record["revision"]}


def _float_list(value: torch.Tensor) -> list:
    return value.detach().cpu().to(torch.float32).tolist()


def _probabilities(logits: torch.Tensor) -> list:
    return _float_list(torch.softmax(logits.to(torch.float32), dim=-1))


def _llr_by_alternate(logits: torch.Tensor) -> dict[str, float]:
    llr = logits[1:] - logits[0]
    return {
        alternate: float(value)
        for alternate, value in zip("CGT", _float_list(llr), strict=True)
    }


def _download_dataset_file(record: dict, filename: str) -> Path:
    input_record = record["expected"]["input"]
    return Path(
        hf_hub_download(
            input_record["dataset_id"],
            filename,
            repo_type="dataset",
            revision=input_record["dataset_revision"],
        )
    )


def _regenerate_gpn(record: dict) -> None:
    expected = record["expected"]
    tokenizer = AutoTokenizer.from_pretrained(
        record["model_id"], **_load_kwargs(record)
    )
    model = AutoModelForMaskedLM.from_pretrained(
        record["model_id"],
        **_load_kwargs(record),
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
    logits = output[0, position, indices]
    expected["logits"] = _float_list(logits)
    expected["probabilities"] = _probabilities(logits)
    expected["llr_alt_minus_ref"] = _llr_by_alternate(logits)


def _regenerate_gpn_msa(record: dict, alignment: np.lib.npyio.NpzFile) -> None:
    expected = record["expected"]
    msa = torch.from_numpy(
        alignment["gpn_msa_tokens"].astype(np.int64),
    ).unsqueeze(0)
    input_ids = msa[:, :, 0].clone()
    aux_features = msa[:, :, 1:]
    position = expected["input"]["masked_position_zero_based"]
    input_ids[0, position] = DNA_VOCAB.index("?")
    model = AutoModelForMaskedLM.from_pretrained(
        record["model_id"],
        **_load_kwargs(record),
    ).eval()
    with torch.inference_mode():
        output = model(input_ids=input_ids, aux_features=aux_features).logits
    logits = output[0, position, NUCLEOTIDE_INDICES]
    expected["logits"] = _float_list(logits)
    expected["probabilities"] = _probabilities(logits)
    expected["llr_alt_minus_ref"] = _llr_by_alternate(logits)


def _regenerate_gpn_star(record: dict, alignment: np.lib.npyio.NpzFile) -> None:
    import pandas as pd

    expected = record["expected"]
    msa = torch.from_numpy(
        alignment["gpn_star_v100_tokens"].astype(np.int64),
    ).unsqueeze(0)
    input_ids = msa[:, :, :1].clone()
    position = expected["input"]["masked_position_zero_based"]
    input_ids[0, position, 0] = DNA_VOCAB.index("?")
    model = AutoModelForMaskedLM.from_pretrained(
        record["model_id"],
        **_load_kwargs(record),
    ).eval()
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            source_ids=msa,
            target_species=np.array([[0]], dtype=int),
        ).logits
    logits = output[0, position, 0, NUCLEOTIDE_INDICES]
    expected["logits"] = _float_list(logits)
    expected["probabilities"] = _probabilities(logits)
    expected["llr_alt_minus_ref"] = _llr_by_alternate(logits)

    table_path = Path(
        hf_hub_download(
            record["model_id"],
            expected["calibration"]["filename"],
            **_load_kwargs(record),
        )
    )
    expected["calibration"]["sha256"] = hashlib.sha256(
        table_path.read_bytes()
    ).hexdigest()
    table = pd.read_parquet(table_path).set_index("pentanuc_mut")
    pentanucleotide = expected["input"]["pentanucleotide"]
    raw_llr = logits[1:] - logits[0]
    neutral_means = {
        alternate: float(
            table.loc[f"{pentanucleotide}_{alternate}", "llr_neutral_mean"]
        )
        for alternate in "CGT"
    }
    expected["llr_neutral_mean"] = neutral_means
    expected["llr_calibrated"] = {
        alternate: float(raw_llr[index]) - neutral_means[alternate]
        for index, alternate in enumerate("CGT")
    }


def _regenerate_phylogpn(record: dict) -> None:
    expected = record["expected"]
    tokenizer = AutoTokenizer.from_pretrained(
        record["model_id"], **_load_kwargs(record)
    )
    model = AutoModel.from_pretrained(record["model_id"], **_load_kwargs(record)).eval()
    pad_size = expected["input"]["symmetric_padding_each_side"]
    sequences = expected["input"]["sequences"]
    padded = [
        tokenizer.pad_token * pad_size + sequence + tokenizer.pad_token * pad_size
        for sequence in sequences
    ]
    input_ids = tokenizer(padded, return_tensors="pt", padding=True)["input_ids"]
    with torch.inference_mode():
        output = model(input_ids=input_ids)
    logits = torch.stack(
        [output[nucleotide][0, : len(sequences[0])] for nucleotide in "ACGT"],
        dim=-1,
    )
    expected["first_sequence_logits"] = _float_list(logits)
    expected["first_sequence_probabilities"] = _probabilities(logits)
    expected["c_to_t_llr_position_one_zero_based"] = float(logits[1, 3] - logits[1, 1])


def _regenerate_sorghum(record: dict) -> None:
    import pandas as pd

    expected = record["expected"]
    input_record = expected["input"]
    labels_path = _download_dataset_file(record, input_record["labels_file"])
    test_path = _download_dataset_file(record, input_record["test_file"])
    labels = labels_path.read_text().splitlines()
    row = pd.read_parquet(test_path).iloc[input_record["row_index"]]
    input_record.update(
        {
            "chrom": str(row["chrom"]),
            "end_zero_based_exclusive": int(row["end"]),
            "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            "sequence": row["seq"],
            "start_zero_based": int(row["start"]),
            "strand": row["strand"],
            "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        }
    )
    expected["label_order"] = labels

    tokenizer = AutoTokenizer.from_pretrained(
        record["model_id"], **_load_kwargs(record)
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        record["model_id"],
        **_load_kwargs(record),
    ).eval()
    input_ids = tokenizer(
        input_record["sequence"],
        return_tensors="pt",
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    with torch.inference_mode():
        output = model(input_ids=input_ids).logits[0]
    expected["predicted_log1p_expression"] = _float_list(output)


def main() -> None:
    args = parse_args()
    if args.output.resolve() == CANONICAL_BASELINE.resolve():
        raise SystemExit(
            "Refusing to overwrite the canonical baseline; review a diff first."
        )
    source_commit, source_tree = _require_committed_source()
    _require_local_gpn_source()
    baseline = json.loads(args.template.read_text())
    register_auto_classes("ss", "msa", "star", "phylo")

    with np.load(ALIGNMENT_PATH, allow_pickle=False) as alignment:
        _regenerate_gpn(baseline["models"]["gpn"])
        _regenerate_gpn_msa(baseline["models"]["gpn_msa"], alignment)
        _regenerate_gpn_star(baseline["models"]["gpn_star"], alignment)
    _regenerate_phylogpn(baseline["models"]["phylogpn"])
    _regenerate_sorghum(baseline["models"]["sorghum_expression"])

    baseline["schema_version"] = 3
    baseline["environment"] = {
        "device": "cpu",
        "dtype": "float32",
        "gpn_source_commit": source_commit,
        "gpn_source_tree": source_tree,
        "gpn_version": importlib.metadata.version("gpn"),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "working_tree_changes": False,
    }
    baseline["generation"] = {
        "approval_reason": args.approval_reason,
        "command": _generation_command(args),
        "method": "Direct inference from every pinned checkpoint revision",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    print(f"Wrote review candidate to {args.output}")


if __name__ == "__main__":
    main()
