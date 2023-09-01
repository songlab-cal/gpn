import argparse
import bioframe as bf

from .data import (
    load_table, Genome, filter_defined, filter_unmasked, filter_length,
    filter_annotation_features, add_jitter, add_flank,
)


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
            intervals, annotation, args.filter_annotation_features,
            args.annotation_features_include_flank,
            args.annotation_features_add_jitter,
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
    parser.add_argument(
        "--annotation-features-add-jitter",
        help="Add jitter to annotation features",
        type=int,
    )
    args = parser.parse_args()
    main(args)
