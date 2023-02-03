import argparse
from Bio import SeqIO
import bioframe as bf
from gpn.utils import load_table, Genome
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def get_gtf_features(gtf, feature):
    gtf_features = gtf[gtf.feature == feature]
    return bf.merge(bf.sanitize_bedframe(gtf_features[["chrom", "start", "end"]]))


def intersect_intervals(a, b):
    return bf.overlap(a, b, how="inner", return_overlap=True)[
        ["chrom", "overlap_start", "overlap_end"]
    ].rename(columns=dict(overlap_start="start", overlap_end="end"))


def add_flank(intervals, flank):
    return bf.merge(bf.expand(intervals, pad=feature_flank)).drop(columns="n_intervals")


def filter_length(intervals, min_interval_len):
    return intervals[intervals.end-intervals.start>=min_interval_len]


def filter_defined(intervals, genome, include_flank=None):
    defined = genome.get_defined_intervals()
    if include_flank is not None:
        defined = add_flank(defined, include_flank)
    return intersect_intervals(intervals, defined)


def filter_unmasked(intervals, genome, include_flank=None):
    unmasked = genome.get_unmasked_intervals()
    if include_flank is not None:
        unmasked = add_flank(unmasked, include_flank)
    return intersect_intervals(intervals, unmasked)


def filter_gtf_features(intervals, gtf, feature, include_flank=None):
    gtf_features = get_gtf_features(gtf, feature)
    if include_flank is not None:
        gtf_features = add_flank(gtf_features, include_flank)
    return intersect_intervals(intervals, gtf_features)


def main(args):
    genome = Genome(args.fasta_path)
    if args.input_intervals_path is None:
        print("All intervals")
        intervals = genome.get_all_intervals()
    else:
        print("User-defined intervals")
        intervals = read_table(args.input_intervals_path)
    intervals = bf.merge(bf.sanitize_bedframe(intervals))
    print(intervals.shape)

    if args.filter_gtf_features is not None:
        gtf = load_table(args.gtf_path)
        intervals = filter_gtf_features(
            intervals, gtf, args.filter_feature, args.gtf_features_include_flank
        )
        print(intervals.shape)
    if args.filter_defined:
        intervals = filter_defined(intervals, genome, args.defined_include_flank)
        print(intervals.shape)
    if args.filter_unmasked:
        intervals = filter_unmasked(intervals, genome, args.unmasked_include_flank)
        print(intervals.shape)
    if args.min_interval_len:
        intervals = filter_length(intervals, args.min_interval_len)
    print(intervals)
    intervals.to_parquet(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define genomic intervals for language modeling."
    )
    parser.add_argument("fasta_path", help="Genome fasta path", type=str)
    parser.add_argument("output_path", help="Output path", type=str)
    parser.add_argument(
        "--input-intervals-path",
        help="Input intervals path. If ommitted, will use full chromosomes in fasta.",
        type=str,
    )
    parser.add_argument("--min-interval-len", help="Minimum interval length", type=int)
    parser.add_argument(
        "--filter-defined",
        help="Keep only ACGTacgt",
        action="store_true",
    )
    parser.add_argument(
        "--defined-include-flank",
        help="Flank of defined regions included",
        type=int,
    )
    parser.add_argument(
        "--filter-unmasked",
        help="Keep only ACGT",
        action="store_true",
    )
    parser.add_argument(
        "--unmasked-include-flank",
        help="Flank of unmasked regions included",
        type=int,
    )
    parser.add_argument(
        "--filter-gtf-features",
        help="Filter to a specific feature of GTF in GTF_PATH, e.g. exon, CDS. Could also be a custom feature annotation such as promoter, enhancer, etc.",
        type=str,
    )
    parser.add_argument(
        "--gtf-feature-include-flank",
        help="Flank of GTF features included",
        type=int,
    )
    parser.add_argument("--gtf-path", help="GTF path", type=str)
    args = parser.parse_args()
    main(args)
