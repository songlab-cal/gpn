from gpn.star.data import (
    Genome,
    GenomeMSA,
    filter_length,
    make_windows,
)
import numpy as np
import pandas as pd
import pyBigWig
from tqdm import tqdm

tqdm.pandas()


CHROMS = [str(i) for i in range(1, 11)]