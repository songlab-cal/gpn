import argparse
import numpy as np
import pandas as pd
#from pandarallel import pandarallel
#pandarallel.initialize(progress_bar=True)
from tqdm import tqdm
tqdm.pandas()

from .utils import Genome, load_table


def make_windows(intervals, window_size, step_size):
    return pd.concat(
        intervals.progress_apply(
            lambda interval: get_interval_windows(interval, window_size, step_size), axis=1,
        ).values,
        ignore_index=True,
    )


def get_interval_windows(interval, window_size, step_size):
    windows = pd.DataFrame(
        dict(start=np.arange(interval.start, interval.end-window_size+1, step_size))
    )
    windows["end"] = windows.start + window_size
    windows["chrom"] = interval.chrom
    windows = windows[["chrom", "start", "end"]]  # just re-ordering
    windows["strand"] = "+"
    windows_neg = windows.copy()
    windows_neg.strand = "-"
    return  pd.concat([windows, windows_neg], ignore_index=True)


def get_seq(intervals, genome):
    intervals["seq"] = intervals.progress_apply(
        lambda i: genome.get_seq(i.chrom, i.start, i.end, i.strand),
        axis=1,
    )
    return intervals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MLM dataset.")
    parser.add_argument("intervals_path", help="Intervals path", type=str)
    parser.add_argument("fasta_path", help="Genome fasta path", type=str)
    parser.add_argument("output_path", help="Output path", type=str)
    parser.add_argument("--window_size", help="Window size", type=int)
    parser.add_argument("--step_size", help="Step size", type=int)
    args = parser.parse_args()
    print(args)

    intervals = load_table(args.intervals_path)
    if args.window_size is not None:
        print("Making windows...")
        assert args.step_size is not None
        intervals = make_windows(intervals, args.window_size, args.step_size)
    if "strand" not in intervals.columns:
        intervals["strand"] = "+"
    print("Loading genome...")
    genome = Genome(args.fasta_path)
    print("Loading genome... Done.")
    intervals = get_seq(intervals, genome)
    intervals = intervals.sample(frac=1.0, random_state=42)
    print(intervals)
    intervals.to_parquet(args.output_path, index=False)
