import argparse
from Bio import SeqIO
import bioframe as bf
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

tqdm.pandas()


DEFINED_SYMBOLS = list("ACGTacgt")
MASKED_SYMBOLS = list("acgt")


def filter_undefined(intervals, genome, min_contig_len):
    print("filter_undefined")

    def find_defined(interval):
        seq = np.array(
            list(str(genome[interval.chrom][interval.start : interval.end].seq))
        )
        intervals = interval.to_frame().T
        intervals = bf.sanitize_bedframe(intervals)
        undefined = pd.DataFrame(
            dict(start=interval.start + np.where(~np.isin(seq, DEFINED_SYMBOLS))[0])
        )
        if len(undefined) > 0:
            undefined["chrom"] = interval.chrom
            undefined["end"] = undefined.start + 1
            undefined = bf.merge(undefined)
            intervals = bf.subtract(intervals, undefined)
        intervals = intervals[intervals.end - intervals.start >= min_contig_len]
        return intervals[["chrom", "start", "end"]]

    intervals = pd.concat(
        intervals.parallel_apply(find_defined, axis=1).values, ignore_index=True
    )
    return intervals


def filter_masked(intervals, genome, min_contig_len, mask_incl_context):
    print("filter_masked")

    def find_unmasked(interval):
        seq = np.array(
            list(str(genome[interval.chrom][interval.start : interval.end].seq))
        )
        intervals = interval.to_frame().T
        intervals = bf.sanitize_bedframe(intervals)
        masked = pd.DataFrame(
            dict(start=interval.start + np.where(np.isin(seq, MASKED_SYMBOLS))[0])
        )
        if len(masked > 0):
            masked["chrom"] = interval.chrom
            masked["end"] = masked.start + 1
            masked = bf.merge(masked)
            masked.start += mask_incl_context
            masked.end -= mask_incl_context
            masked = masked.query("end-start > 0")
            if len(masked) > 0:
                intervals = bf.subtract(intervals, masked)
        intervals = intervals[intervals.end - intervals.start >= min_contig_len]
        return intervals

    intervals = pd.concat(
        intervals.parallel_apply(find_unmasked, axis=1).values, ignore_index=True
    )
    return intervals


def filter_feature(
    intervals, gtf, feature, min_contig_len, feature_flank, filter_protein_coding
):
    gtf = gtf[(gtf.feature == feature) & (gtf.chrom.isin(intervals.chrom.unique()))]
    if filter_protein_coding:
        gtf["transcript_biotype"] = gtf.attribute.str.extract(
            r'transcript_biotype "([^;]*)";'
        )
        gtf = gtf[gtf.transcript_biotype=="protein_coding"]
    gtf = gtf[["chrom", "start", "end"]]
    gtf = bf.sanitize_bedframe(gtf)
    gtf = bf.merge(gtf)
    gtf = bf.expand(gtf, pad=feature_flank)
    gtf = bf.merge(gtf)
    intervals = bf.overlap(intervals, gtf, how="inner", return_overlap=True)[
        ["chrom", "overlap_start", "overlap_end"]
    ].rename(columns=dict(overlap_start="start", overlap_end="end"))
    intervals = intervals[intervals.end - intervals.start >= min_contig_len]
    return intervals


def main(args):
    with gzip.open(args.fasta_path, "rt") if args.fasta_path.endswith(".gz") else open(
        args.fasta_path
    ) as handle:
        genome = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
    if args.input_intervals_path is None:
        intervals = pd.DataFrame(
            [[chrom, 0, len(record)] for chrom, record in genome.items()],
            columns=["chrom", "start", "end"],
        )
    else:
        intervals = pd.read_csv(args.input_intervals_path, sep="\t")
    intervals.chrom = intervals.chrom.astype(str)
    intervals = bf.sanitize_bedframe(intervals)
    intervals = bf.merge(intervals)
    print(intervals.shape)

    if args.filter_feature is not None:
        gtf = pd.read_csv(
            args.gtf_path,
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
        gtf.chrom = gtf.chrom.astype(str)
        intervals = filter_feature(
            intervals, gtf, args.filter_feature, args.min_contig_len,
            args.feature_flank, args.filter_protein_coding,
        )
        print(intervals.shape)
    if args.filter_undefined:
        intervals = filter_undefined(intervals, genome, args.min_contig_len)
        print(intervals.shape)
    if args.filter_masked:
        intervals = filter_masked(
            intervals, genome, args.min_contig_len, args.mask_incl_context
        )
        print(intervals.shape)
    print(intervals)
    intervals.to_csv(args.output_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define genomic intervals for language modeling."
    )
    parser.add_argument("fasta_path", help="Genome fasta path", type=str)
    parser.add_argument(
        "input_intervals_path",
        help="Input intervals path. If ommitted, will use full chromosomes in fasta.",
        type=str,
    )
    parser.add_argument("output_path", help="Output path", type=str)
    parser.add_argument("--min-contig-len", help="Minimum contig length", type=int, default=128)
    parser.add_argument(
        "--mask-incl-context",
        help="Border of masked regions included as context",
        type=int,
    )
    parser.add_argument(
        "--filter-undefined",
        help="Exclude undefined nucleotides (e.g. N)",
        action="store_true",
    )
    parser.add_argument(
        "--filter-masked",
        help="Exclude masked nucleotides (represented in lowercase)",
        action="store_true",
    )
    parser.add_argument(
        "--filter-feature",
        help="Filter to a specific feature of GTF in GTF_PATH",
        type=str,
    )
    parser.add_argument(
        "--feature-flank",
        help="Add flank to GTF feature.",
        type=int,
    )
    parser.add_argument("--gtf-path", help="GTF path", type=str)
    parser.add_argument("--filter-protein-coding", action="store_true")
    args = parser.parse_args()
    main(args)
