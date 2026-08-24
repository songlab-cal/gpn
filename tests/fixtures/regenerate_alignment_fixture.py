"""Regenerate the tiny published-model alignment fixture from a local Zarr.

This utility intentionally requires an already-present multiz100way store. It never
downloads or builds an alignment dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).parent
DEFAULT_METADATA = HERE / "published_model_baseline.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "msa_zarr",
        type=Path,
        help="path to an existing 100-species multiz100way Zarr store",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="baseline metadata containing coordinates and species order",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination .npz path (use a temporary path when only checking)",
    )
    return parser.parse_args()


def tokenize(raw: np.ndarray, vocabulary: list[str]) -> np.ndarray:
    unknown_index = vocabulary.index("-")
    table = np.full(256, unknown_index, dtype=np.uint8)
    for index, nucleotide in enumerate(vocabulary):
        table[ord(nucleotide)] = index
    return table[np.char.upper(raw).view(np.uint8)]


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.metadata.read_text())
    metadata = baseline["alignment_fixture"]
    root = zarr.open(args.msa_zarr, mode="r")
    raw = np.asarray(
        root[metadata["chrom"]][
            metadata["start_zero_based"] : metadata["end_zero_based_exclusive"]
        ]
    ).view("S1")
    if raw.shape != (128, 100):
        raise ValueError(f"Expected a (128, 100) slice, got {raw.shape}")

    raw_sha256 = hashlib.sha256(raw.tobytes(order="C")).hexdigest()
    expected_raw_sha256 = metadata["source"]["raw_slice_sha256"]
    if expected_raw_sha256 and raw_sha256 != expected_raw_sha256:
        raise ValueError(
            "The source slice does not match the recorded fixture provenance: "
            f"{raw_sha256} != {expected_raw_sha256}"
        )

    full_species = metadata["gpn_star_v100_species"]
    reduced_species = metadata["gpn_msa_species"]
    subset_indices = [full_species.index(species) for species in reduced_species]
    if subset_indices != [0, *range(11, 100)]:
        raise ValueError("Unexpected GPN-MSA species subset or ordering")

    tokens = tokenize(raw, metadata["token_vocabulary"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        gpn_msa_tokens=tokens[:, subset_indices],
        gpn_star_v100_tokens=tokens,
    )
    fixture_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    if fixture_sha256 != metadata["sha256"]:
        raise ValueError(
            "Regenerated fixture does not match the committed artifact: "
            f"{fixture_sha256} != {metadata['sha256']}"
        )

    print(f"raw slice sha256: {raw_sha256}")
    print(f"fixture sha256: {fixture_sha256}")


if __name__ == "__main__":
    main()
