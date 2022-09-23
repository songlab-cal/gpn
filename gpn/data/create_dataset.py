import argparse
from Bio import SeqIO
import gzip
from tqdm import tqdm


tqdm.pandas()


def get_contig_windows(contig, window_size, step_size):
    windows = pd.DataFrame(
        dict(start=np.arange(contig.start, contig.end - window_size, step_size))
    )
    windows["end"] = windows.start + window_size
    windows["chrom"] = contig.chrom
    windows["strand"] = "+"
    windows_neg = windows.copy()
    windows_neg.strand = "-"
    windows = pd.concat([windows, windows_neg], ignore_index=True)
    return windows


def get_window_seq(window, genome):
    seq = genome[window.chrom][window.start:window.end].seq
    if window.strand == "-":
        seq = seq.reverse_complement()
    return str(seq)


def main(args):
    print(args)
    with gzip.open(args.fasta_path, "rt") if args.fasta_path.endswith(".gz") else open(
        args.fasta_path
    ) as handle:
        genome = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
    df = pd.read_csv(args.intervals_path, sep="\t")
    df.chrom = df.chrom.astype(str)
    if args.split_into_windows:
        df = pd.concat(
            df.apply(
                lambda contig: get_contig_windows(
                    contig, args.window_size, args.step_size
                ),
                axis=1,
            ).values,
            ignore_index=True,
        )
        df["seq"] = df.progress_apply(
            lambda window: get_window_seq(window, genome), axis=1
        )
    else:
        df["seq"] = df.progres_apply(
            lambda row: str(genome[row.chrom][row.start:row.end].seq), axis=1
        )
    df = df.sample(frac=1.0, random_state=42)
    print(df)
    df.to_parquet(data_args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset in parquet format.")
    parser.add_argument("--fasta-path", help="Genome fasta path", type=str)
    parser.add_argument("--intervals-path", help="Intervals path", type=str)
    parser.add_argument("--output-path", help="Output path", type=str)
    parser.add_argument(
        "--split-into-windows",
        help="Split into windows of window_size and step_size",
        action="store_true",
    )
    parser.add_argument("--window-size", help="Window size", type=int)
    parser.add_argument("--step-size", help="Step size", type=int)
    args = parser.parse_args()
    main(args)
