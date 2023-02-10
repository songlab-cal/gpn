import argparse
from Bio.Seq import Seq
import pandas as pd
from tqdm import tqdm

from .utils import Genome, load_table


tqdm.pandas()


def get_seq(variant, genome, window_size):
    ref_str = variant.ref
    alt_str = variant.alt
    seq = genome.get_seq(variant.chrom, variant.start, variant.end, variant.strand).upper()
    assert len(seq) == window_size
    window_pos = window_size // 2
    if variant.strand == "-":
        window_pos = window_size - window_pos - 1
        ref_str = str(Seq(ref_str).reverse_complement())
        alt_str = str(Seq(alt_str).reverse_complement())
    assert seq[window_pos] == ref_str
    seq_list = list(seq)
    seq_list[window_pos] = "[MASK]"
    seq = "".join(seq_list)
    return seq, window_pos, ref_str, alt_str


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x : x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers


def get_seq_dnabert(variant, genome, window_size):
    K = 6
    assert K==6
    assert window_size % 2 == 0
    ref_str = variant.ref
    alt_str = variant.alt
    seq = genome.get_window_seq(variant).upper()
    assert len(seq) == window_size
    kmer_pos = K // 2
    if variant.strand == "-":
        ref_str = str(Seq(ref_str).reverse_complement())
        alt_str = str(Seq(alt_str).reverse_complement())
        kmer_pos = K - kmer_pos - 1
    seq_list = seq2kmer(seq, K).split(" ")
    window_pos = len(seq_list) // 2
    ref_kmer_str = seq_list[window_pos]
    assert ref_kmer_str[kmer_pos] == ref_str
    alt_kmer_str = list(ref_kmer_str) # copy
    alt_kmer_str[kmer_pos] = alt_str
    alt_kmer_str = "".join(alt_kmer_str)
    range_low = 2 if variant.strand == "+" else 3
    range_high = 4 if variant.strand == "+" else 3
    for i in range(window_pos-range_low, window_pos+range_high):
        seq_list[i] = "[MASK]"
    seq = " ".join(seq_list)
    pos = 1+window_pos  # to account for CLS token
    return seq, pos, ref_kmer_str, alt_kmer_str


def make_dataset(variants, genome, window_size, get_seq_fn=get_seq):
    variants["start"] = variants.pos - window_size//2
    variants["end"] = variants.start + window_size
    variants["strand"] = "+"
    variants_neg = variants.copy()
    variants_neg.strand = "-"
    variants = pd.concat([variants, variants_neg], ignore_index=True)
    variants[["seq", "pos", "ref", "alt"]] = variants.progress_apply(
        lambda v: get_seq_fn(v, genome, window_size), axis=1
    ).tolist()
    variants = variants[["seq", "pos", "ref", "alt"]]
    return variants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create VEP dataset.")
    parser.add_argument(
        "variants_path", type=str,
        help="Variants path (csv,tsv,parquet,vcf as long as it has chrom,pos,ref,alt)"
    )
    parser.add_argument("fasta_path", help="Genome fasta path", type=str)
    parser.add_argument("window_size", help="Window size", type=int)
    parser.add_argument("output_path", help="Output path (parquet)", type=str)
    parser.add_argument("--dnabert", help="DNABERT style", action="store_true")
    args = parser.parse_args()

    variants = load_table(args.variants_path)[["chrom", "pos", "ref", "alt"]]
    print("Loading genome...")
    genome = Genome(args.fasta_path)
    print("Loading genome... Done.")

    d = make_dataset(
        variants, genome, args.window_size,
        get_seq if not args.dnabert else get_seq_dnabert
    )
    d.to_parquet(args.output_path, index=False)
