"""Local Zarr alignment access for deprecated GPN-MSA inference."""

from typing import Any

import numpy as np
import zarr

from gpn.data import ReverseComplementer, Tokenizer, slice_alignment


class GenomeMSA:
    def __init__(
        self,
        path: str,
        subset_chroms: list[str] | None = None,
        in_memory: bool = False,
    ):
        self.reverse_complementer = ReverseComplementer()
        self.tokenizer = Tokenizer()
        store = zarr.open(path, mode="r")
        chroms = list(store.keys())
        if subset_chroms is not None:
            chroms = [chrom for chrom in chroms if chrom in subset_chroms]
        if in_memory:
            self.data = {chrom: store[chrom][:] for chrom in chroms}
        else:
            self.data = {chrom: store[chrom] for chrom in chroms}

    def get_msa(self, chrom: str, start: int, end: int) -> Any:
        array = self.data[chrom]
        return slice_alignment(array, start, end, array.shape[1])

    def get_msa_batch_fwd_rev(
        self,
        chroms: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        *,
        tokenize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        forward = np.array(
            [
                self.get_msa(str(chroms[index]), int(starts[index]), int(ends[index]))
                for index in range(len(chroms))
            ]
        )
        reverse = self.reverse_complementer(forward, position_axis=1)
        if tokenize:
            forward = self.tokenizer(forward)
            reverse = self.tokenizer(reverse)
        return forward, reverse
