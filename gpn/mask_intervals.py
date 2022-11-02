import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import Genome, load_table


tqdm.pandas()


def get_intervals_mask(intervals, genome, mask_lowercase, mask_flank):
    intervals["strand"] = "+"
    intervals["seq"] = intervals.progress_apply(
        lambda i: np.array(list(genome.get_window_seq(i))),
        axis=1,
    )
    def get_mask_flank(seq):
        mask = np.zeros_like(seq, dtype=bool)
        if mask_flank > 0:
            mask[:mask_flank] = True
            mask[-mask_flank:] = True  # mask[-0:] == mask
        return mask
    intervals["no_loss_mask"] = intervals.seq.progress_apply(get_mask_flank)
    if mask_lowercase:
        intervals.no_loss_mask = intervals.progress_apply(
            lambda i: i.no_loss_mask | np.char.islower(i.seq),
            axis=1,
        )
    intervals.drop(columns=["seq", "strand"], inplace=True)
    return intervals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask genomic intervals so no loss is computed."
    )
    parser.add_argument("intervals_path", help="Intervals path.", type=str)
    parser.add_argument("fasta_path", help="Soft-masked genome fasta path", type=str)
    parser.add_argument("output_path", help="Output path", type=str)
    parser.add_argument("--mask-lowercase", help="Mask lowercase", action="store_true")
    parser.add_argument(
        "--mask-flank", help="Mask flank by this amount.", type=int, default=0,
    )
    args = parser.parse_args()
    print(args)

    intervals = load_table(args.intervals_path)
    intervals.chrom = intervals.chrom.astype(str)
    print("Loading genome...")
    genome = Genome(args.fasta_path)
    print("Loading genome... Done.")
    intervals = get_intervals_mask(
        intervals, genome, args.mask_lowercase, args.mask_flank,
    )
    intervals.to_parquet(args.output_path, index=False)
