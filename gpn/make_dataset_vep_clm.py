import argparse
from Bio.Seq import Seq
import pandas as pd
from tqdm import tqdm

from .utils import Genome, load_table


tqdm.pandas()


def get_seqs(variant, genome, window_size):
    ref_str = variant.ref
    alt_str = variant.alt
    seq_ref = genome.get_window_seq(variant).upper()
    assert len(seq_ref) == window_size
    window_pos = window_size // 2
    if variant.strand == "-":
        window_pos = window_size - window_pos - 1
        ref_str = str(Seq(ref_str).reverse_complement())
        alt_str = str(Seq(alt_str).reverse_complement())
    assert seq_ref[window_pos] == ref_str
    seq_alt = list(seq_ref)
    seq_alt[window_pos] = alt_str
    seq_alt = "".join(seq_alt)
    return seq_ref, seq_alt


def make_dataset(variants, genome, window_size):
    variants["start"] = variants.pos - window_size//2
    variants["end"] = variants.start + window_size
    variants["strand"] = "+"
    variants_neg = variants.copy()
    variants_neg.strand = "-"
    variants = pd.concat([variants, variants_neg], ignore_index=True)
    variants[["seq_ref", "seq_alt"]] = variants.progress_apply(
        lambda v: get_seqs(v, genome, window_size), axis=1
    ).tolist()
    variants = variants[["seq_ref", "seq_alt"]]
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
    args = parser.parse_args()

    variants = load_table(args.variants_path)[["chrom", "pos", "ref", "alt"]]
    print("Loading genome...")
    genome = Genome(args.fasta_path)
    print("Loading genome... Done.")

    d = make_dataset(variants, genome, args.window_size)
    d.to_parquet(args.output_path, index=False)
