import argparse
from Bio import SeqIO
import bioframe as bf
from gpn.utils import load_table, Genome
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


# TODO: maybe call it I or Is, or ivals instead of intervals


def get_annotation_features(annotation, feature):
    annotation_features = annotation[annotation.feature == feature]
    return bf.merge(bf.sanitize_bedframe(annotation_features[["chrom", "start", "end"]]))


def intersect_intervals(a, b):
    return bf.overlap(a, b, how="inner", return_overlap=True)[
        ["chrom", "overlap_start", "overlap_end"]
    ].rename(columns=dict(overlap_start="start", overlap_end="end"))


def union_intervals(a, b):
    return bf.merge(pd.concat([a, b], ignore_index=True)).drop(columns="n_intervals")


def intervals_size(intervals):
    return (intervals.end-intervals.start).sum()


def add_flank(intervals, flank):
    return bf.merge(bf.expand(intervals, pad=flank)).drop(columns="n_intervals")


def add_jitter(intervals, magnitude, seed=42):
    # After using this function, we recommend intersecting with
    # Genome.get_all_intervals(), to avoid getting out of chromosome bounds
    # or smaller subsets such as Genome.get_defined_intervals()
    rng = np.random.default_rng(seed)
    jitter = rng.integers(-magnitude, magnitude, size=len(intervals), endpoint=True)
    new_intervals = intervals.copy()
    new_intervals.start += jitter
    new_intervals.end += jitter
    return bf.merge(new_intervals)


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


def filter_annotation_features(intervals, annotation, feature, include_flank=None):
    annotation_features = get_annotation_features(annotation, feature)
    if include_flank is not None:
        annotation_features = add_flank(annotation_features, include_flank)
    return intersect_intervals(intervals, annotation_features)


def get_promoters(annotation, upstream_size):
    # not exactly getting promoters, just gettting regions upstream of TSS
    
    def get_promoter(transcript):
        if transcript.strand == "+":
            start, end = transcript.start-upstream_size, transcript.start
        else:
            start, end = transcript.end, transcript.end+upstream_size
        return pd.Series(dict(chrom=transcript.chrom, start=start, end=end))

    transcripts = annotation.query('feature=="transcript"')
    promoters = transcripts.apply(get_promoter, axis=1)
    return bf.merge(promoters)


def get_random_intervals(intervals, size, n, seed=42):
    rng = np.random.default_rng(seed)
    interval_size = (intervals.end-intervals.start).values
    # the number of random intervals that can be generated per interval
    # e.g. if target size is 512, an interval of size 512 can produce 1 interval,
    # and interval of size 513 can produce 2 intervals
    interval_w = 1 + interval_size - size
    interval_p = interval_w / interval_w.sum()
    rand_interval_index = rng.choice(len(intervals), p=interval_p, size=n)

    rand_intervals = []
    for i in range(n):
        interval = intervals.iloc[rand_interval_index[i]]
        start = rng.integers((interval.end-interval.start) - size, endpoint=True)
        end = start + size
        rand_intervals.append([interval.chrom, start, end])
    rand_intervals = pd.DataFrame(rand_intervals, columns=["chrom", "start", "end"])
    return bf.merge(rand_intervals).drop(columns="n_intervals")


def main(args):
    if args.input_intervals_path is None:
        print("All intervals")
        genome = Genome(args.fasta_path)
        intervals = genome.get_all_intervals()
    else:
        print("User-defined intervals")
        intervals = load_table(args.input_intervals_path)
    intervals = bf.merge(bf.sanitize_bedframe(intervals))
    print(intervals.shape)
    if args.min_interval_len:
        intervals = filter_length(intervals, args.min_interval_len)

    if args.filter_annotation_features is not None:
        annotation = load_table(args.annotation_path)
        intervals = filter_annotation_features(
            intervals, annotation, args.filter_annotation_features, args.annotation_features_include_flank
        )
        print(intervals.shape)
    if args.filter_defined:
        if genome is None: genome = Genome(args.fasta_path)
        intervals = filter_defined(intervals, genome, args.defined_include_flank)
        print(intervals.shape)
    if args.filter_unmasked:
        if genome is None: genome = Genome(args.fasta_path)
        intervals = filter_unmasked(intervals, genome, args.unmasked_include_flank)
        print(intervals.shape)
    if args.min_interval_len:
        intervals = filter_length(intervals, args.min_interval_len)
    print(intervals)
    intervals.to_parquet(args.output_path, index=False)


# TODO: consider removing this. It's just too complicated and better to have 
# the user manually call the functions from python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define genomic intervals for language modeling."
    )
    parser.add_argument("output_path", help="Output path", type=str)
    parser.add_argument("--fasta-path", help="Genome fasta path", type=str)
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
        "--filter-annotation-features",
        help="Filter to a specific feature of annotation in ANNOTATION_PATH, e.g. exon, CDS. Could also be a custom feature annotation such as promoter, enhancer, etc.",
        type=str,
    )
    parser.add_argument(
        "--annotation-features-include-flank",
        help="Flank of annotation features included",
        type=int,
    )
    parser.add_argument("--annotation-path", help="annotation path", type=str)
    args = parser.parse_args()
    main(args)
