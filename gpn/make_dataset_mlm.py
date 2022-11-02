import argparse
import numpy as np
import pandas as pd
#from pandarallel import pandarallel
#pandarallel.initialize(progress_bar=True)
from tqdm import tqdm
tqdm.pandas()

from .utils import Genome, load_table


def get_contig_windows(contig, window_size, step_size):
    window_no_loss_flank = window_size // 8
    max_no_loss_window = 0.95
    windows = pd.DataFrame(
        dict(start=np.arange(contig.start, contig.end-window_size+1, step_size))
    )
    windows["end"] = windows.start + window_size
    windows["chrom"] = contig.chrom
    windows["strand"] = "+"

    def prepare_no_loss_mask(w):
        m = contig.no_loss_mask[w.start-contig.start:w.end-contig.start].copy()
        m[:window_no_loss_flank] = True
        m[-window_no_loss_flank:] = True
        return m

    if 'no_loss_mask' in contig: 
        windows["no_loss_mask"] = windows.apply(prepare_no_loss_mask, axis=1)
        windows = windows[windows.no_loss_mask.apply(np.mean) <= max_no_loss_window]
    windows_neg = windows.copy()
    windows_neg.strand = "-"
    if 'no_loss_mask' in contig:
        windows_neg.no_loss_mask = windows_neg.no_loss_mask.apply(np.flip)
    windows = pd.concat([windows, windows_neg], ignore_index=True)
    return windows


def make_dataset(intervals, genome, window_size, step_size):
    intervals.chrom = intervals.chrom.astype(str)
    windows = pd.concat(
        intervals.progress_apply(
            lambda contig: get_contig_windows(contig, window_size, step_size), axis=1,
        ).values,
        ignore_index=True,
    )
    windows["seq"] = windows.progress_apply(genome.get_window_seq, axis=1)
    windows = windows.sample(frac=1.0, random_state=42)
    return windows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MLM dataset.")
    parser.add_argument("intervals_path", help="Intervals path", type=str)
    parser.add_argument("fasta_path", help="Genome fasta path", type=str)
    parser.add_argument("window_size", help="Window size", type=int)
    parser.add_argument("step_size", help="Step size", type=int)    
    parser.add_argument("output_path", help="Output path", type=str)
    args = parser.parse_args()
    print(args)

    intervals = load_table(args.intervals_path)
    print("Loading genome...")
    genome = Genome(args.fasta_path)
    print("Loading genome... Done.")
    d = make_dataset(intervals, genome, args.window_size, args.step_size)
    d.to_parquet(args.output_path, index=False)
