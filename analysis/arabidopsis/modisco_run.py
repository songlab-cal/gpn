import bioframe as bf
#from gpn.utils import Genome
import modiscolite
import numpy as np
import pandas as pd
from scipy.special import softmax
from tqdm import tqdm
tqdm.pandas()

import gzip
from Bio import SeqIO, bgzf
from Bio.Seq import Seq

def load_fasta(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as handle:
        genome = pd.Series({rec.id: str(rec.seq) for rec in SeqIO.parse(handle, "fasta")})
    return genome


class Genome:
    def __init__(self, path):
        self._genome = load_fasta(path)

    def get_seq(self, chrom, start, end, strand="+"):
        seq = self._genome[chrom][start:end]
        if strand == "-":
            seq = str(Seq(seq).reverse_complement())
        return seq


nucleotides = list("ACGT")
nucleotides_ref_1hot = ["ref_1hot_"+nuc for nuc in nucleotides]
pred = pd.read_parquet(snakemake.input[0])
pred.pos -= 1  # 1-based -> 0-based
pred.loc[:, nucleotides] = softmax(
    pd.read_parquet(snakemake.input[1])[nucleotides].values, axis=1
) - 1/4  # centering
genome = Genome(snakemake.input[2])
pred["ref_nuc"] = pred.progress_apply(
    lambda row: genome.get_seq(row.chrom, row.pos, row.pos+1).upper(), axis=1
)
for nuc in nucleotides:
    pred[f"ref_1hot_{nuc}"] = pred.progress_apply(
        lambda row: 1.0 if row.ref_nuc==nuc else 0.0, axis=1
    )
pred = pred.set_index(["chrom", "pos"], drop=False)
pred["start"] = pred.pos
pred["end"] = pred.pos + 1
pred_spans = bf.merge(bf.expand(pred, pad=10))

def get_sequences(span):
    pos = np.arange(span.start, span.end)
    chrom = [span.chrom] * len(pos)
    return pred.reindex(zip(chrom, pos))[nucleotides_ref_1hot].fillna(1/4).values

def get_attributions(span):
    pos = np.arange(span.start, span.end)
    chrom = [span.chrom] * len(pos)
    return pred.reindex(zip(chrom, pos))[nucleotides].fillna(0).values

sequences = np.expand_dims(
    np.concatenate(pred_spans.progress_apply(get_sequences, axis=1).values), 0
)
attributions = np.expand_dims(
    np.concatenate(pred_spans.progress_apply(get_attributions, axis=1).values), 0
)
pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
    hypothetical_contribs=attributions, 
    one_hot=sequences,
    max_seqlets_per_metacluster=100_000,  # default is 2000
    sliding_window_size=20,
    flank_size=5,
    target_seqlet_fdr=0.05,
    n_leiden_runs=3,  # default is 2
    verbose=True,
)
modiscolite.io.save_hdf5(snakemake.output[0], pos_patterns, neg_patterns)
