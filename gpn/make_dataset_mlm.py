import argparse
import numpy as np
import pandas as pd

from .data import Genome, load_table, make_windows, get_interval_windows, get_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MLM dataset.")
    parser.add_argument("intervals_path", help="Intervals path", type=str)
    parser.add_argument("fasta_path", help="Genome fasta path", type=str)
    parser.add_argument("output_path", help="Output path", type=str)
    parser.add_argument("--window_size", help="Window size", type=int)
    parser.add_argument("--step_size", help="Step size", type=int)
    parser.add_argument('--add_rc', help="Add reverse complement as data augmentation", action='store_true')
    args = parser.parse_args()
    print(args)

    intervals = load_table(args.intervals_path)
    if args.window_size is not None:
        print("Making windows...")
        assert args.step_size is not None
        intervals = make_windows(intervals, args.window_size, args.step_size, args.add_rc)
    if "strand" not in intervals.columns:
        intervals["strand"] = "+"
    print("Loading genome...")
    genome = Genome(args.fasta_path)
    print("Loading genome... Done.")
    intervals = get_seq(intervals, genome)
    intervals = intervals.sample(frac=1.0, random_state=42)
    print(intervals)
    intervals.to_parquet(args.output_path, index=False)
