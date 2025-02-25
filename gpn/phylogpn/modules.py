from functools import cached_property
from typing import List, Optional

import torch
from torch import nn
from torch.nn.utils import parametrize


def check_if_involution(indices: List[int]) -> bool:
    return all(indices[indices[idx]] == idx for idx in range(len(indices)))


def get_conv1d_output_length(
    input_length: int, kernel_size: int, stride_size: int = 1, pad_size: int = 0, dilation_rate: int = 1
) -> int:
    return (input_length + 2 * pad_size - dilation_rate * (kernel_size - 1) - 1) // stride_size + 1


def get_involution_indices(size: int) -> List[int]:
    return list(reversed(range(size)))


class RCEWeight(nn.Module):
    def __init__(
        self, input_involution_indices: List[int], output_involution_indices: List[int]
    ):
        if not check_if_involution(input_involution_indices) or not check_if_involution(
                output_involution_indices):
            raise ValueError(
                "`input_involution_indices` and `output_involution_indices` must be involutions"
            )

        super().__init__()
        self._input_involution_indices = input_involution_indices
        self._output_involution_indices = output_involution_indices
        self._input_involution_index_tensor = None
        self._output_involution_index_tensor = None
        self._device = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._device != x.device:
            self._input_involution_index_tensor = torch.tensor(self._input_involution_indices, device=x.device)
            self._output_involution_index_tensor = torch.tensor(self._output_involution_indices, device=x.device)
            self._device = x.device

        output_involution_indices = self._output_involution_index_tensor
        input_involution_indices = self._input_involution_index_tensor
        return (x + x[output_involution_indices][:, input_involution_indices].flip(2)) / 2


class IEBias(nn.Module):
    def __init__(self, involution_indices: List[int]):
        if not check_if_involution(involution_indices):
            raise ValueError("`involution_indices` must be an involution")

        super().__init__()
        self._involution_indices = involution_indices
        self._involution_index_tensor = None
        self._device = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._device != x.device:
            self._involution_index_tensor = torch.tensor(self._involution_indices, device=x.device)
            self._device = x.device

        involution_indices = self._involution_index_tensor
        return (x + x[involution_indices]) / 2


class IEWeight(nn.Module):
    def __init__(
        self, input_involution_indices: List[int], output_involution_indices: List[int]
    ):
        if not check_if_involution(input_involution_indices) or not check_if_involution(
                output_involution_indices):
            raise ValueError(
                "`input_involution_indices` and `output_involution_indices` must be involutions"
            )

        super().__init__()
        self._input_involution_indices = input_involution_indices
        self._output_involution_indices = output_involution_indices
        self._input_involution_index_tensor = None
        self._output_involution_index_tensor = None
        self._device = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._device != x.device:
            self._input_involution_index_tensor = torch.tensor(self._input_involution_indices, device=x.device)
            self._output_involution_index_tensor = torch.tensor(self._output_involution_indices, device=x.device)
            self._device = x.device

        output_involution_indices = self._output_involution_index_tensor
        input_involution_indices = self._input_involution_index_tensor
        return (x + x[input_involution_indices][:, output_involution_indices]) / 2


class RCEByteNetBlock(nn.Module):
    def __init__(self, outer_involution_indices: List[int], inner_dim: int, kernel_size: int, dilation_rate: int = 1):
        outer_dim = len(outer_involution_indices)

        if outer_dim % 2 != 0:
            raise ValueError("`outer_involution_indices` must have an even length")

        if inner_dim % 2 != 0:
            raise ValueError("`inner_dim` must be even")

        if kernel_size % 2 == 0:
            raise ValueError("`kernel_size` must be odd")

        super().__init__()
        inner_involution_indices = get_involution_indices(inner_dim)

        layers = [
            nn.GroupNorm(1, outer_dim),
            nn.GELU(),
            nn.Conv1d(outer_dim, inner_dim, kernel_size=1),
            nn.GroupNorm(1, inner_dim),
            nn.GELU(),
            nn.Conv1d(inner_dim, inner_dim, kernel_size, dilation=dilation_rate),
            nn.GroupNorm(1, inner_dim),
            nn.GELU(),
            nn.Conv1d(inner_dim, outer_dim, kernel_size=1)
        ]
        parametrize.register_parametrization(
            layers[2], "weight",
            RCEWeight(outer_involution_indices, inner_involution_indices)
        )
        parametrize.register_parametrization(
            layers[2], "bias",
            IEBias(inner_involution_indices)
        )
        parametrize.register_parametrization(
            layers[5], "weight",
            RCEWeight(inner_involution_indices, inner_involution_indices)
        )
        parametrize.register_parametrization(
            layers[5], "bias",
            IEBias(inner_involution_indices)
        )
        parametrize.register_parametrization(
            layers[8], "weight",
            RCEWeight(inner_involution_indices, outer_involution_indices)
        )
        parametrize.register_parametrization(
            layers[8], "bias",
            IEBias(outer_involution_indices)
        )
        self.layers = nn.Sequential(*layers)
        self._kernel_size = kernel_size
        self._dilation_rate = dilation_rate

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def dilation_rate(self):
        return self._dilation_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_length = x.shape[2]
        output_length = get_conv1d_output_length(input_length, self.kernel_size, dilation_rate=self.dilation_rate)
        a = (input_length - output_length) // 2
        
        if a == 0:
            return self.layers(x) + x
        
        return self.layers(x) + x[:, :, a:-a]
    
class PhyloGPN(nn.Module):
    def __init__(
        self,
        input_involution_indices: List[int],
        output_involution_indices: List[int],
        dilation_rates: List[int],
        outer_dim: int,
        inner_dim: int,
        kernel_size: int,
        pad_token_idx: Optional[int] = None,
    ):
        if pad_token_idx is not None and input_involution_indices[pad_token_idx] != pad_token_idx:
            raise ValueError("`input_involution_indices[pad_token_idx]` must be equal to `pad_token_idx`")

        super().__init__()
        vocab_size = len(input_involution_indices)
        outer_involution_indices = get_involution_indices(outer_dim)

        self.embedding = nn.Embedding(vocab_size, outer_dim, padding_idx=pad_token_idx)
        parametrize.register_parametrization(
            self.embedding, "weight",
            IEWeight(input_involution_indices, outer_involution_indices)
        )
        nn.init.normal_(self.embedding.weight, std=2**0.5)
        self.embedding.weight.data[self.embedding.padding_idx].zero_()
        self.embedding.requires_grad = False

        blocks = []
        receptive_field_size = 1

        for r in dilation_rates:
            blocks.append(RCEByteNetBlock(outer_involution_indices, inner_dim, kernel_size, dilation_rate=r))
            receptive_field_size += (kernel_size - 1) * r

        self.blocks = nn.Sequential(*blocks)

        output_dim = len(output_involution_indices)
        self.output_layers = nn.Sequential(
            nn.GroupNorm(1, outer_dim), nn.GELU(), nn.Conv1d(outer_dim, output_dim, kernel_size=1)
        )
        parametrize.register_parametrization(
            self.output_layers[-1], "weight",
            RCEWeight(outer_involution_indices, output_involution_indices)
        )
        parametrize.register_parametrization(
            self.output_layers[-1], "bias", IEBias(output_involution_indices)
        )

        self._embedding_involution_indices = outer_involution_indices

    @property
    def embedding_involution_indices(self):
        return self._embedding_involution_indices

    def get_embeddings(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_tensor).swapaxes(1, 2)
        return self.output_layers[0](self.blocks(x)).swapaxes(1, 2)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.get_embeddings(input_tensor).swapaxes(1, 2)
        return self.output_layers[1:](x).swapaxes(1, 2)