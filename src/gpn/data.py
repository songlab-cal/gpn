"""Dataset and nucleotide-array utilities shared by model families."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from jaxtyping import UInt8


def load_table(path: str) -> pd.DataFrame:
    """Read one supported genomic table into a DataFrame."""

    if path.endswith(".parquet"):
        frame = pd.read_parquet(path)
    elif "csv" in path:
        frame = pd.read_csv(path)
    elif "tsv" in path:
        frame = pd.read_csv(path, sep="\t")
    elif "vcf" in path:
        frame = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
            usecols=[0, 1, 3, 4],
            dtype={0: str},
        ).rename(columns={0: "chrom", 1: "pos", 3: "ref", 4: "alt"})
    elif "gtf" in path or "gff" in path:
        frame = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
            dtype={"chrom": str},
            names=[
                "chrom",
                "source",
                "feature",
                "start",
                "end",
                "score",
                "strand",
                "frame",
                "attribute",
            ],
        )
        frame.start -= 1
    else:
        raise ValueError(f"Unsupported input table format: {path}")
    frame.chrom = frame.chrom.astype(str)
    return frame


def load_dataset_from_file_or_dir(
    path: str,
    split: str = "test",
    is_file: bool = False,
    **kwargs: Any,
) -> Dataset:
    """Load a local table or a named Hugging Face dataset split."""

    if is_file:
        return Dataset.from_pandas(load_table(path))
    return load_dataset(path, split=split, **kwargs)


class Tokenizer:
    """Tokenize nucleotide byte arrays with the shared six-token vocabulary."""

    def __init__(self, vocab: str = "-ACGT?"):
        unknown = vocab.index("-")
        self.table: UInt8[np.ndarray, "... byte"] = np.full(
            (256,), unknown, dtype=np.uint8
        )
        for index, character in enumerate(vocab):
            self.table[ord(character)] = index
        self.vocab = vocab
        self.mask_token = "?"
        self.pad_token = "-"

    def __call__(self, value: Any) -> Any:
        return self.table[np.char.upper(value).view(np.uint8)]

    def __len__(self) -> int:
        return len(self.vocab)

    def mask_token_id(self) -> int:
        return self.vocab.index("?")

    def unk_token_id(self) -> int:
        return self.vocab.index("-")

    def pad_token_id(self) -> int:
        return self.vocab.index("-")

    def nucleotide_token_id_start(self) -> int:
        return self.vocab.index("A")

    def nucleotide_token_id_end(self) -> int:
        return self.vocab.index("T") + 1


class ReverseComplementer:
    """Reverse-complement DNA represented as NumPy byte arrays."""

    def __init__(self) -> None:
        complement_mapping = {
            b"A": b"T",
            b"T": b"A",
            b"C": b"G",
            b"G": b"C",
            b"a": b"t",
            b"t": b"a",
            b"c": b"g",
            b"g": b"c",
        }
        self.table: np.ndarray = np.array(
            [
                complement_mapping.get(chr(index).encode(), chr(index).encode())
                for index in range(256)
            ],
            dtype="|S1",
        )

    def __call__(self, value: Any, position_axis: int = -1) -> Any:
        return self.table[np.flip(value, axis=position_axis).view(np.uint8)]


def slice_alignment(
    array: Any,
    start: int,
    end: int,
    n_species: int,
) -> np.ndarray:
    """Slice a reference-indexed alignment and gap-pad out-of-bounds positions."""

    if end <= start:
        raise ValueError("Alignment interval end must be greater than start")
    result: np.ndarray = np.full((end - start, n_species), b"-", dtype="|S1")
    source_start = max(start, 0)
    source_stop = min(end, array.shape[0])
    if source_start < source_stop:
        destination_start = source_start - start
        destination_stop = destination_start + source_stop - source_start
        result[destination_start:destination_stop] = np.asarray(
            array[source_start:source_stop]
        ).view("|S1")
    return result
