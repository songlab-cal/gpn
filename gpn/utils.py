import gzip
from Bio import SeqIO
import pandas as pd


def load_fasta(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as handle:
        return SeqIO.to_dict(SeqIO.parse(handle, "fasta"))


# Some standard formats
def load_table(path):
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif 'csv' in path:
        df = pd.read_csv(path)
    elif 'tsv' in path:
        df = pd.read_csv(path, sep='\t')
    elif 'vcf' in path:
        df = pd.read_csv(
            path, sep="\t", header=None, comment="#", usecols=[0,1,2,3,4],
        ).rename(cols={0: 'chrom', 1: 'pos', 2: 'id', 3: 'ref', 4: 'alt'})
        df.pos -= 1
    elif 'gtf' in path or 'gff' in path:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
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
    df.chrom = df.chrom.astype(str)
    return df


def load_repeatmasker(path):
    df = pd.read_csv(path, sep="\t").rename(
        columns=dict(genoName="chrom", genoStart="start", genoEnd="end")
    )
    df.chrom = df.chrom.astype(str)
    return df


class Genome:
    def __init__(self, path):
        self.genome = load_fasta(path)

    def get_window_seq(self, window):
        seq = self.genome[window.chrom][window.start:window.end].seq
        if window.strand == "-":
            seq = seq.reverse_complement()
        return str(seq)
