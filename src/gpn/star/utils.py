from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int, Num
from numpy.lib.stride_tricks import sliding_window_view
from torch import Tensor


def max_smooth(
    arr: Num[np.ndarray, "batch position"], window_size: int
) -> Num[np.ndarray, "batch position"]:
    # assert window_size is odd
    assert window_size % 2 == 1

    # pad the array with zeros on the right side along the L dimension
    padded_arr = np.pad(
        arr,
        ((0, 0), (window_size // 2, window_size // 2)),
        mode="constant",
        constant_values=0,
    )

    # create a view of the array with a sliding window
    windowed_arr = sliding_window_view(padded_arr, window_shape=(1, window_size))

    # find the max in each window, and reshape the result to 2D
    return np.max(windowed_arr, axis=-1).reshape(arr.shape)


def calculate_clade_avg_nuc_freq(
    T: Int[Tensor, "batch position species nucleotide"],
    labels: Int[Tensor, "... species"],
) -> Float[Tensor, "batch position nucleotide"]:
    C = labels.unique().size(0)

    labels_onehot = F.one_hot(labels, num_classes=C).float()
    counts = torch.einsum("blnv,nc->blcv", T.float(), labels_onehot)
    counts_per_group = labels_onehot.sum(dim=0).clamp(min=1)
    counts_per_group = counts_per_group.view(1, 1, C, 1)
    freqs = counts / counts_per_group
    avg_freqs = freqs.mean(dim=2)

    return avg_freqs


def sample_nuc_from_freq(
    avg_freqs: Float[Tensor, "batch position nucleotide"], N: int
) -> Int[Tensor, "batch position sample"]:
    probs = avg_freqs / avg_freqs.sum(dim=2, keepdim=True)  # Shape: (B, L, V)
    B, L, V = probs.shape

    probs_flat = probs.view(B * L, V)  # Shape: (B*L, V)
    samples_flat = torch.multinomial(
        probs_flat, num_samples=N, replacement=True
    )  # Shape: (B*L, N)
    samples = samples_flat.view(B, L, N)

    return samples


def get_all_species_mask(
    clade_mask: Bool[Tensor, "batch position selected_clade"],
    clade_indices: Int[Tensor, "batch selected_clade"],
    species_clade_indices: Int[Tensor, "... species"],
) -> Bool[Tensor, "batch position species"]:
    N = species_clade_indices.shape[0]

    clade_indices_expanded = clade_indices.unsqueeze(2)  # Shape: (B, C, 1)
    species_clade_indices_expanded = species_clade_indices.view(
        1, 1, N
    )  # Shape: (1, 1, C)
    match = (
        clade_indices_expanded == species_clade_indices_expanded
    ).float()  # Shape: (B, C, N)

    # Permute clade_mask to shape (B, C, L) and convert to float
    clade_mask_permuted = clade_mask.permute(0, 2, 1).float()  # Shape: (B, C, L)

    # Perform batch matrix multiplication: (B, N, C) x (B, C, L) -> (B, N, L)
    species_mask = torch.bmm(
        match.permute(0, 2, 1), clade_mask_permuted
    )  # Shape: (B, N, L)

    # Permute dimensions to get shape (B, L, N) and convert to boolean
    species_mask = species_mask.permute(0, 2, 1).bool()

    return species_mask


def find_directory_sum_paths(path_str: str | Path) -> dict[int, str]:
    # Preserve the logical directory name: SCF layouts commonly use a `100`
    # symlink whose target is named `99` (99 aligned species plus the target).
    root = Path(path_str).expanduser().absolute()
    if not root.exists():
        raise FileNotFoundError(f"MSA path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"MSA path is not a directory: {root}")

    direct_store = root / "all.zarr"
    if direct_store.is_dir():
        if not root.name.isdigit():
            raise ValueError(
                "A direct GPN-Star MSA directory must have a numeric species-count "
                f"name, but found {root.name!r}: {root}"
            )
        return {int(root.name): str(direct_store)}

    stores = [
        (int(child.name), child / "all.zarr")
        for child in root.iterdir()
        if child.is_dir() and child.name.isdigit() and (child / "all.zarr").is_dir()
    ]
    if not stores:
        raise ValueError(
            "No GPN-Star MSA stores found. Expected either a numeric directory "
            f"containing all.zarr or numeric children containing all.zarr: {root}"
        )
    return {
        species_count: str(store)
        for species_count, store in sorted(stores, reverse=True)
    }


def normalize_logits(logits: pd.DataFrame) -> pd.DataFrame:
    logits_array = logits.values

    exp_logits = np.exp(logits_array)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    normalized_logits = np.log(probs)

    return pd.DataFrame(normalized_logits, columns=logits.columns, index=logits.index)


def get_entropy(logits: pd.DataFrame) -> np.ndarray:
    logits_array = logits.values

    probs = np.exp(logits_array)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return entropy


def get_llr(
    logits: pd.DataFrame,
    ref: Sequence[str],
    alt: Sequence[str],
) -> np.ndarray:
    ref_logits = logits.values[
        np.arange(len(ref)), [logits.columns.get_loc(r) for r in ref]
    ]
    alt_logits = logits.values[
        np.arange(len(alt)), [logits.columns.get_loc(a) for a in alt]
    ]

    return alt_logits - ref_logits
