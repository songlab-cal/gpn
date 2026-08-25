"""Local whole-genome MSA access for GPN-Star training and inference."""

from typing import Any

import numpy as np
import zarr

from gpn.data import ReverseComplementer, Tokenizer, slice_alignment


class GenomeMSA:
    def __init__(
        self,
        path: str,
        n_species: int,
        subset_chroms: list[str] | None = None,
        in_memory: bool = False,
    ):
        self.reverse_complementer = ReverseComplementer()
        self.tokenizer = Tokenizer()
        self.n_species = n_species
        store = zarr.open(path, mode="r")
        chroms = list(store.keys())
        if subset_chroms is not None:
            chroms = [chrom for chrom in chroms if chrom in subset_chroms]
        if in_memory:
            self.data = {chrom: store[chrom][:] for chrom in chroms}
        else:
            self.data = {chrom: store[chrom] for chrom in chroms}

    def get_msa(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        strand: str = "+",
        tokenize: bool = False,
    ) -> Any:
        msa = slice_alignment(self.data[chrom], start, end, self.n_species)
        if strand == "-":
            msa = self.reverse_complementer(msa, position_axis=0)
        return self.tokenizer(msa) if tokenize else msa

    def get_msa_fwd_rev(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        tokenize: bool = False,
    ) -> tuple[Any, Any]:
        forward = self.get_msa(chrom, start, end)
        reverse = self.reverse_complementer(forward, position_axis=0)
        if tokenize:
            forward = self.tokenizer(forward)
            reverse = self.tokenizer(reverse)
        return forward, reverse

    def _normalize_batch(
        self,
        values: list[Any],
        starts: list[int] | np.ndarray,
        ends: list[int] | np.ndarray,
        *,
        tokenize: bool,
    ) -> np.ndarray:
        try:
            return np.array(values)
        except ValueError:
            batch_size = len(values)
            length: int = int(ends[0] - starts[0])
            dtype: Any = np.uint8 if tokenize else "S1"
            result = np.zeros((batch_size, length, self.n_species), dtype=dtype)
            if not tokenize:
                result[:] = b"-"
            for index, value in enumerate(values):
                result[index, :, 0] = value[:length, 0]
            return result

    def get_msa_batch(
        self,
        chroms: list[str],
        starts: list[int],
        ends: list[int],
        strands: list[str],
        *,
        tokenize: bool = False,
    ) -> np.ndarray:
        values = [
            self.get_msa(
                str(chroms[index]),
                int(starts[index]),
                int(ends[index]),
                strand=strands[index],
                tokenize=tokenize,
            )
            for index in range(len(chroms))
        ]
        return self._normalize_batch(values, starts, ends, tokenize=tokenize)

    def get_msa_batch_fwd_rev(
        self,
        chroms: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        *,
        tokenize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        pairs = [
            self.get_msa_fwd_rev(
                str(chroms[index]),
                int(starts[index]),
                int(ends[index]),
                tokenize=tokenize,
            )
            for index in range(len(chroms))
        ]
        forward, reverse = zip(*pairs, strict=True)
        return (
            self._normalize_batch(list(forward), starts, ends, tokenize=tokenize),
            self._normalize_batch(list(reverse), starts, ends, tokenize=tokenize),
        )
