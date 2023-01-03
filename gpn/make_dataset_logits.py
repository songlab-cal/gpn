import argparse
from Bio.Seq import Seq
import pandas as pd
from tqdm import tqdm

from .utils import Genome, load_table


tqdm.pandas()


def get_seq(variant, genome, window_size):
    seq = genome.get_window_seq(variant).upper()
    assert len(seq) == window_size
    window_pos = window_size // 2
    if variant.strand == "-":
        window_pos = window_size - window_pos - 1
    seq_list = list(seq)
    seq_list[window_pos] = "[MASK]"
    seq = "".join(seq_list)
    return seq, window_pos


def make_dataset(variants, genome, window_size, get_seq_fn=get_seq):
    variants["start"] = variants.pos - window_size//2
    variants["end"] = variants.start + window_size
    variants["strand"] = "+"
    variants_neg = variants.copy()
    variants_neg.strand = "-"
    variants = pd.concat([variants, variants_neg], ignore_index=True)
    variants[["seq", "pos"]] = variants.progress_apply(
        lambda v: get_seq_fn(v, genome, window_size), axis=1
    ).tolist()
    variants = variants[["seq", "pos"]]
    return variants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create logits dataset.")
    parser.add_argument(
        "variants_path", type=str,
        help="Variants path (csv,tsv,parquet,vcf as long as it has chrom,pos)"
    )
    parser.add_argument("fasta_path", help="Genome fasta path", type=str)
    parser.add_argument("window_size", help="Window size", type=int)
    parser.add_argument("output_path", help="Output path (parquet)", type=str)
    args = parser.parse_args()

    variants = load_table(args.variants_path)[["chrom", "pos"]]
    print("Loading genome...")
    genome = Genome(args.fasta_path)
    print("Loading genome... Done.")

    d = make_dataset(variants, genome, args.window_size, get_seq)
    d.to_parquet(args.output_path, index=False)
