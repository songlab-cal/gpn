import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn.functional as F
import os

def max_smooth(arr, window_size):
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


def calculate_clade_avg_nuc_freq(T, labels):
    C = labels.unique().size(0)

    labels_onehot = F.one_hot(labels, num_classes=C).float()
    counts = torch.einsum('blnv,nc->blcv', T.float(), labels_onehot)
    counts_per_group = labels_onehot.sum(dim=0).clamp(min=1)
    counts_per_group = counts_per_group.view(1, 1, C, 1)
    freqs = counts / counts_per_group
    avg_freqs = freqs.mean(dim=2)

    return avg_freqs

def sample_nuc_from_freq(avg_freqs, N):
    probs = avg_freqs / avg_freqs.sum(dim=2, keepdim=True)  # Shape: (B, L, V)
    B, L, V = probs.shape
    
    probs_flat = probs.view(B * L, V)  # Shape: (B*L, V)
    samples_flat = torch.multinomial(probs_flat, num_samples=N, replacement=True)  # Shape: (B*L, N)
    samples = samples_flat.view(B, L, N)

    return samples

def get_all_species_mask(clade_mask, clade_indices, species_clade_indices):

    N = species_clade_indices.shape[0]

    clade_indices_expanded = clade_indices.unsqueeze(2)  # Shape: (B, C, 1)
    species_clade_indices_expanded = species_clade_indices.view(1, 1, N)  # Shape: (1, 1, C)
    match = (clade_indices_expanded == species_clade_indices_expanded).float()  # Shape: (B, C, N)

    # Permute clade_mask to shape (B, C, L) and convert to float
    clade_mask_permuted = clade_mask.permute(0, 2, 1).float()  # Shape: (B, C, L)

    # Perform batch matrix multiplication: (B, N, C) x (B, C, L) -> (B, N, L)
    species_mask = torch.bmm(match.permute(0, 2, 1), clade_mask_permuted)  # Shape: (B, N, L)

    # Permute dimensions to get shape (B, L, N) and convert to boolean
    species_mask = species_mask.permute(0, 2, 1).bool()

    return species_mask


def find_directory_sum_paths(path_str):
    dirs = os.listdir(path_str)
    _, final_dirname = os.path.split(path_str)
    if 'all.zarr' in dirs:
        return {int(final_dirname): os.path.join(path_str, 'all.zarr')}
    else:
        dirs = sorted(dirs, reverse=True)
        return {int(name): os.path.join(path_str, name, 'all.zarr') for name in dirs}
    

def normalize_logits(logits):
    logits_array = logits.values
    
    exp_logits = np.exp(logits_array)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    normalized_logits = np.log(probs)
    
    return pd.DataFrame(normalized_logits, columns=logits.columns, index=logits.index)

def get_entropy(logits):
    logits_array = logits.values
    
    probs = np.exp(logits_array)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return entropy