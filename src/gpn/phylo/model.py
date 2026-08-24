"""Maintained PhyloGPN configuration, tokenizer, and inference model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn.utils import parametrize
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer


class PhyloGPNConfig(PretrainedConfig):
    """Configuration for the published PhyloGPN convolutional model."""

    model_type = "phylogpn"

    def __init__(
        self,
        outer_dim: int = 960,
        inner_dim: int = 480,
        kernel_size: int = 5,
        stack_size: int = 2,
        num_stacks: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.outer_dim = outer_dim
        self.inner_dim = inner_dim
        self.kernel_size = kernel_size
        self.stack_size = stack_size
        self.num_stacks = num_stacks


class PhyloGPNTokenizer(PreTrainedTokenizer):
    """Single-nucleotide tokenizer used by the published PhyloGPN model."""

    model_input_names: ClassVar[list[str]] = ["input_ids"]

    def __init__(
        self,
        model_max_length: int | None = None,
        unk_token: str = "N",
        pad_token: str = "-",
        bos_token: str | None = None,
        eos_token: str | None = None,
        sep_token: str | None = None,
        cls_token: str | None = None,
        mask_token: str | None = None,
        split_special_tokens: bool = True,
        **kwargs: Any,
    ) -> None:
        self._tokens = tuple("ACGTN-")
        self._vocab = {token: index for index, token in enumerate(self._tokens)}
        super().__init__(
            model_max_length=model_max_length,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token,
            split_special_tokens=split_special_tokens,
            add_prefix_space=kwargs.pop("add_prefix_space", False),
            padding_side=kwargs.pop("padding_side", "right"),
            **kwargs,
        )

    def _tokenize(self, sequence: str) -> list[str]:
        return list(sequence)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab["N"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._tokens[index]

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None,
    ) -> tuple[str, ...]:
        return ()


def _is_involution(indices: Sequence[int]) -> bool:
    return all(indices[indices[index]] == index for index in range(len(indices)))


def _reverse_indices(size: int) -> list[int]:
    return list(reversed(range(size)))


class _ReverseComplementEquivariantWeight(nn.Module):
    def __init__(
        self,
        input_involution_indices: Sequence[int],
        output_involution_indices: Sequence[int],
    ) -> None:
        super().__init__()
        if not _is_involution(input_involution_indices) or not _is_involution(
            output_involution_indices
        ):
            raise ValueError("Input and output indices must be involutions")

        self._input_indices = tuple(input_involution_indices)
        self._output_indices = tuple(output_involution_indices)
        self._input_index_tensor: Tensor | None = None
        self._output_index_tensor: Tensor | None = None
        self._device: torch.device | None = None

    def forward(self, weight: Float[Tensor, "output input kernel"]) -> Tensor:
        if self._device != weight.device:
            self._input_index_tensor = torch.tensor(
                self._input_indices,
                device=weight.device,
            )
            self._output_index_tensor = torch.tensor(
                self._output_indices,
                device=weight.device,
            )
            self._device = weight.device

        assert self._input_index_tensor is not None
        assert self._output_index_tensor is not None
        reverse_complement = weight[self._output_index_tensor][
            :, self._input_index_tensor
        ].flip(2)
        return (weight + reverse_complement) / 2


class _InvariantBias(nn.Module):
    def __init__(self, involution_indices: Sequence[int]) -> None:
        super().__init__()
        if not _is_involution(involution_indices):
            raise ValueError("Bias indices must be an involution")

        self._indices = tuple(involution_indices)
        self._index_tensor: Tensor | None = None
        self._device: torch.device | None = None

    def forward(self, bias: Float[Tensor, " output"]) -> Tensor:
        if self._device != bias.device:
            self._index_tensor = torch.tensor(self._indices, device=bias.device)
            self._device = bias.device

        assert self._index_tensor is not None
        return (bias + bias[self._index_tensor]) / 2


class _InvariantEmbeddingWeight(nn.Module):
    def __init__(
        self,
        input_involution_indices: Sequence[int],
        output_involution_indices: Sequence[int],
    ) -> None:
        super().__init__()
        if not _is_involution(input_involution_indices) or not _is_involution(
            output_involution_indices
        ):
            raise ValueError("Input and output indices must be involutions")

        self._input_indices = tuple(input_involution_indices)
        self._output_indices = tuple(output_involution_indices)
        self._input_index_tensor: Tensor | None = None
        self._output_index_tensor: Tensor | None = None
        self._device: torch.device | None = None

    def forward(self, weight: Float[Tensor, "input output"]) -> Tensor:
        if self._device != weight.device:
            self._input_index_tensor = torch.tensor(
                self._input_indices,
                device=weight.device,
            )
            self._output_index_tensor = torch.tensor(
                self._output_indices,
                device=weight.device,
            )
            self._device = weight.device

        assert self._input_index_tensor is not None
        assert self._output_index_tensor is not None
        reverse_complement = weight[self._input_index_tensor][
            :, self._output_index_tensor
        ]
        return (weight + reverse_complement) / 2


class _RCEByteNetBlock(nn.Module):
    def __init__(
        self,
        outer_involution_indices: Sequence[int],
        inner_dim: int,
        kernel_size: int,
        dilation_rate: int = 1,
    ) -> None:
        super().__init__()
        outer_dim = len(outer_involution_indices)
        if outer_dim % 2 != 0:
            raise ValueError("Outer dimension must be even")
        if inner_dim % 2 != 0:
            raise ValueError("Inner dimension must be even")
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        inner_involution_indices = _reverse_indices(inner_dim)
        self.layer_norm1 = nn.LayerNorm(outer_dim)
        self.conv1 = nn.Conv1d(outer_dim, inner_dim, kernel_size=1)
        self.layer_norm2 = nn.LayerNorm(inner_dim)
        self.conv2 = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size,
            dilation=dilation_rate,
        )
        self.layer_norm3 = nn.LayerNorm(inner_dim)
        self.conv3 = nn.Conv1d(inner_dim, outer_dim, kernel_size=1)
        self.gelu = nn.GELU()

        parametrize.register_parametrization(
            self.conv1,
            "weight",
            _ReverseComplementEquivariantWeight(
                outer_involution_indices,
                inner_involution_indices,
            ),
        )
        parametrize.register_parametrization(
            self.conv1,
            "bias",
            _InvariantBias(inner_involution_indices),
        )
        parametrize.register_parametrization(
            self.conv2,
            "weight",
            _ReverseComplementEquivariantWeight(
                inner_involution_indices,
                inner_involution_indices,
            ),
        )
        parametrize.register_parametrization(
            self.conv2,
            "bias",
            _InvariantBias(inner_involution_indices),
        )
        parametrize.register_parametrization(
            self.conv3,
            "weight",
            _ReverseComplementEquivariantWeight(
                inner_involution_indices,
                outer_involution_indices,
            ),
        )
        parametrize.register_parametrization(
            self.conv3,
            "bias",
            _InvariantBias(outer_involution_indices),
        )
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

    def forward(self, inputs: Float[Tensor, "batch channel position"]) -> Tensor:
        trim_size = (self.kernel_size - 1) * self.dilation_rate // 2
        x = self.layer_norm1(inputs.swapaxes(1, 2)).swapaxes(1, 2)
        x = self.gelu(x)
        x = self.conv1(x)
        x = self.layer_norm2(x.swapaxes(1, 2)).swapaxes(1, 2)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.layer_norm3(x.swapaxes(1, 2)).swapaxes(1, 2)
        x = self.gelu(x)
        x = self.conv3(x)

        if trim_size == 0:
            return x + inputs
        return x + inputs[:, :, trim_size:-trim_size]


class _RCEByteNet(nn.Module):
    def __init__(
        self,
        input_involution_indices: Sequence[int],
        output_involution_indices: Sequence[int],
        dilation_rates: Sequence[int],
        outer_dim: int,
        inner_dim: int,
        kernel_size: int,
        pad_token_index: int | None = None,
    ) -> None:
        super().__init__()
        if (
            pad_token_index is not None
            and input_involution_indices[pad_token_index] != pad_token_index
        ):
            raise ValueError("Pad token index must be invariant")

        outer_involution_indices = _reverse_indices(outer_dim)
        self.embedding = nn.Embedding(
            len(input_involution_indices),
            outer_dim,
            padding_idx=pad_token_index,
        )
        parametrize.register_parametrization(
            self.embedding,
            "weight",
            _InvariantEmbeddingWeight(
                input_involution_indices,
                outer_involution_indices,
            ),
        )
        nn.init.normal_(self.embedding.weight, std=2**0.5)
        if self.embedding.padding_idx is not None:
            self.embedding.weight.data[self.embedding.padding_idx].zero_()

        self.blocks = nn.Sequential(
            *(
                _RCEByteNetBlock(
                    outer_involution_indices,
                    inner_dim,
                    kernel_size,
                    dilation_rate=rate,
                )
                for rate in dilation_rates
            )
        )
        self.output_layer_norm = nn.LayerNorm(outer_dim)
        self.output_gelu = nn.GELU()
        self.output_conv = nn.Conv1d(
            outer_dim,
            len(output_involution_indices),
            kernel_size=1,
        )
        parametrize.register_parametrization(
            self.output_conv,
            "weight",
            _ReverseComplementEquivariantWeight(
                outer_involution_indices,
                output_involution_indices,
            ),
        )
        parametrize.register_parametrization(
            self.output_conv,
            "bias",
            _InvariantBias(output_involution_indices),
        )

    def get_embeddings(
        self,
        input_ids: Int[Tensor, "batch position"],
    ) -> Float[Tensor, "batch position channel"]:
        x = self.embedding(input_ids).swapaxes(1, 2)
        x = self.blocks(x)
        return self.output_layer_norm(x.swapaxes(1, 2))

    def forward(
        self,
        input_ids: Int[Tensor, "batch position"],
    ) -> Float[Tensor, "batch position nucleotide"]:
        x = self.get_embeddings(input_ids).swapaxes(1, 2)
        x = self.output_gelu(x)
        return self.output_conv(x).swapaxes(1, 2)


class PhyloGPNModel(PreTrainedModel):
    """Reverse-complement-equivariant PhyloGPN inference model."""

    config_class = PhyloGPNConfig

    def __init__(self, config: PhyloGPNConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        dilation_rates = config.num_stacks * [
            config.kernel_size**index for index in range(config.stack_size)
        ]
        self._model = _RCEByteNet(
            input_involution_indices=[3, 2, 1, 0, 4, 5],
            output_involution_indices=[3, 2, 1, 0],
            dilation_rates=dilation_rates,
            outer_dim=config.outer_dim,
            inner_dim=config.inner_dim,
            kernel_size=config.kernel_size,
            pad_token_index=5,
        )
        # Required by modern Transformers loaders to collect tied-weight and
        # device-placement metadata after all child modules exist.
        self.post_init()

    def get_embeddings(
        self,
        input_ids: Int[Tensor, "batch position"],
    ) -> Float[Tensor, "batch position channel"]:
        return self._model.get_embeddings(input_ids)

    def forward(
        self,
        input_ids: Int[Tensor, "batch position"],
        **_: Any,
    ) -> dict[str, Float[Tensor, "batch position"]]:
        output = self._model(input_ids)
        return {
            nucleotide: output[:, :, index] for index, nucleotide in enumerate("ACGT")
        }
