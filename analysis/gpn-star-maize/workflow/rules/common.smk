from Bio import Phylo
from gpn.star.data import (
    BigWig,
    Genome,
    GenomeMSA,
    filter_length,
    make_windows,
)
from gpn.star.utils import get_llr
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


SPLIT_CHROMS = {
    "train": [str(i) for i in range(1, 9)],
    "validation": ["9"],
    "test": ["10"],
}
SPLITS = SPLIT_CHROMS.keys()
CHROMS = np.concatenate(list(SPLIT_CHROMS.values()))


def cluster_clades(phylo_dist_pairwise, threshold):
    N = phylo_dist_pairwise.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i+1, N):
            if phylo_dist_pairwise[i, j] <= threshold:
                G.add_edge(i, j)
    clade_dict = {i: nodes for i, nodes in enumerate(list(nx.connected_components(G)))}
    return clade_dict