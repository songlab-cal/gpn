from Bio import SeqIO
from Bio.Seq import Seq
from functools import reduce
import gzip
import numpy as np
import pandas as pd
import pyranges as pr
from tqdm import tqdm
import xml.etree.ElementTree as ET


tqdm.pandas()


SPLIT_CHROMS = config["split_chroms"]
SPLITS = list(SPLIT_CHROMS.keys())
CHROMS = np.concatenate(list(SPLIT_CHROMS.values()))


def tss_pos_bin(pos: int) -> int:
    """
    Bin the TSS position.
    e.g. if bin size is 100, then positions 0-99 will be binned to 50,
    positions 100-199 will be binned to 150, etc.
    """
    bin_size = config["tss_pos_bin_size"]
    return (pos // bin_size) * bin_size + bin_size // 2


def get_assay_name(assay):
    if assay.text in config["assay_skip_renaming"]:
        return assay.text
    return assay.get("technical_replicate_id", assay.text)


def avg_correlation(df, method):
    if df.shape[1] < 2: return np.nan
    corr_matrix = df.corr(method=method)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Compute the average correlation
    return upper_triangle.stack().mean()


def load_fasta(path, subset_chroms=None):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as handle:
        genome = pd.Series(
            {
                rec.id: str(rec.seq)
                for rec in SeqIO.parse(handle, "fasta")
                if subset_chroms is None or rec.id in subset_chroms
            }
        )
    return genome


class Genome:
    def __init__(self, path, subset_chroms=None):
        self._genome = load_fasta(path, subset_chroms=subset_chroms)
        self.chrom_sizes = {chrom: len(seq) for chrom, seq in self._genome.items()}

    def __call__(self, chrom, start, end, strand="+"):
        chrom_size = self.chrom_sizes[chrom]
        seq = self._genome[chrom][max(start,0):min(end,chrom_size)]

        if start < 0: seq = "N" * (-start) + seq  # left padding
        if end > chrom_size: seq = seq + "N" * (end - chrom_size)  # right padding

        if strand == "-":
            seq = str(Seq(seq).reverse_complement())
        return seq