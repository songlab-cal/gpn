import bioframe as bf
from itertools import combinations
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

AUTOSOMES = [str(i) for i in range(1, 23)]
SEX_CHROMS = ["X", "Y"]
CHROMS = AUTOSOMES + SEX_CHROMS

CENTER_WINDOW_SIZE = 100


def make_windows(intervals, window_size, step_size):
    intervals = intervals[intervals.end - intervals.start >= window_size]
    return pd.concat(
        intervals.progress_apply(
            lambda interval: get_interval_windows(interval, window_size, step_size),
            axis=1,
        ).values,
        ignore_index=True,
    )


def get_interval_windows(interval, window_size, step_size):
    windows = pd.DataFrame(
        dict(start=np.arange(interval.start, interval.end - window_size + 1, step_size))
    )
    windows["end"] = windows.start + window_size
    windows["chrom"] = interval.chrom
    windows = windows[["chrom", "start", "end"]]  # just re-ordering
    if "label" in interval:
        windows["label"] = interval.label
    return windows


class BigWigInMemory:
    def __init__(self, path, subset_chroms=None, fill_nan=None):
        import pyBigWig

        with pyBigWig.open(path) as bw:
            chrom_len = bw.chroms()
            chroms = subset_chroms if subset_chroms is not None else chrom_len.keys()
            print("Loading data...")
            self.data = pd.Series(
                {
                    chrom: bw.values(chrom, 0, chrom_len[chrom], numpy=True)
                    for chrom in tqdm(chroms)
                }
            )
            if fill_nan is not None:
                print(f"Filling NaNs with {fill_nan}...")
                self.data = self.data.apply(lambda x: np.nan_to_num(x, nan=fill_nan))

    def __call__(self, chrom, start, end):
        return self.data[chrom][start:end]
