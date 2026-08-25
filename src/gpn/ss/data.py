"""Reference-genome access for single-species GPN."""

import gzip
from typing import Any

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


def load_fasta(path: str, subset_chroms: set[str] | None = None) -> pd.Series:
    opener = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    with opener as handle:
        return pd.Series(
            {
                record.id: str(record.seq)
                for record in SeqIO.parse(handle, "fasta")
                if subset_chroms is None or record.id in subset_chroms
            }
        )


class Genome:
    def __init__(self, path: str, subset_chroms: set[str] | None = None):
        self._genome = load_fasta(path, subset_chroms=subset_chroms)
        self.chrom_sizes = {
            chrom: len(sequence) for chrom, sequence in self._genome.items()
        }

    def get_seq(self, chrom: str, start: int, end: int, strand: str = "+") -> str:
        chrom_size = self.chrom_sizes[chrom]
        sequence = self._genome[chrom][max(start, 0) : min(end, chrom_size)]
        if start < 0:
            sequence = "N" * (-start) + sequence
        if end > chrom_size:
            sequence += "N" * (end - chrom_size)
        if strand == "-":
            sequence = str(Seq(sequence).reverse_complement())
        return sequence

    def get_seq_fwd_rev(self, chrom: str, start: int, end: int) -> tuple[str, str]:
        forward = self.get_seq(chrom, start, end)
        return forward, str(Seq(forward).reverse_complement())


def token_input_id(token: str, tokenizer: Any, n_prefix: int = 0) -> int:
    return tokenizer(token)["input_ids"][n_prefix]
