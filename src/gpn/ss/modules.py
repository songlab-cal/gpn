from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from transformers.modeling_outputs import (
    BaseModelOutput,
)


class EncoderConfig(Protocol):
    """GPN configuration fields consumed by convolutional encoders."""

    dilation_max: int
    dilation_base: int
    dilation_cycle: int
    dilation_double_every: int
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    first_kernel_size: int
    rest_kernel_size: int
    hidden_dropout_prob: float
    bias: bool
    depthwise: bool
    slim: bool


class TransposeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, x: Float[Tensor, "batch first second"]
    ) -> Float[Tensor, "batch second first"]:
        x = torch.transpose(x, 1, 2)
        return x


# class ConvLayer(nn.Module):
#    def __init__(
#        self,
#        hidden_size=None,
#        **kwargs,
#    ):
#        super().__init__()
#        self.conv = nn.Sequential(
#            TransposeLayer(),
#            nn.Conv1d(
#                in_channels=hidden_size,
#                out_channels=hidden_size,
#                padding="same",
#                **kwargs,
#            ),
#            TransposeLayer(),
#            nn.GELU(),
#            nn.LayerNorm(hidden_size),
#        )
#        self.ffn = nn.Sequential(
#            nn.Linear(hidden_size, hidden_size),
#            nn.GELU(),
#            nn.LayerNorm(hidden_size),
#        )
#
#    def forward(self, x):
#        x = x + self.conv(x)
#        x = x + self.ffn(x)
#        return x


# class ConvLayer(nn.Module):
#    def __init__(
#        self,
#        hidden_size=None,
#        intermediate_size=None,
#        **kwargs,
#    ):
#        super().__init__()
#        self.conv = nn.Sequential(
#            TransposeLayer(),
#            nn.Conv1d(
#                in_channels=hidden_size,
#                out_channels=hidden_size,
#                padding="same",
#                **kwargs,
#            ),
#            TransposeLayer(),
#            nn.GELU(),
#            nn.Linear(hidden_size, hidden_size),
#        )
#        self.conv_ln = nn.LayerNorm(hidden_size)
#        self.ffn = nn.Sequential(
#            nn.Linear(hidden_size, intermediate_size),
#            nn.GELU(),
#            nn.Linear(intermediate_size, hidden_size),
#        )
#        self.ffn_ln = nn.LayerNorm(hidden_size)
#
#    def forward(self, x):
#        x = self.conv_ln(x + self.conv(x))
#        x = self.ffn_ln(x + self.ffn(x))
#        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        hidden_dropout_prob: float | None = None,
        bias: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if hidden_size is None or intermediate_size is None:
            raise ValueError("hidden_size and intermediate_size are required")
        if hidden_dropout_prob is None:
            raise ValueError("hidden_dropout_prob is required")
        self.conv = nn.Sequential(
            nn.LayerNorm(hidden_size, bias=bias),
            TransposeLayer(),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                padding="same",
                bias=bias,
                **kwargs,
            ),
            TransposeLayer(),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
            nn.Dropout(hidden_dropout_prob),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size, bias=bias),
            nn.Linear(hidden_size, intermediate_size, bias=bias),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size, bias=bias),
            nn.Dropout(hidden_dropout_prob),
        )

    def forward(
        self, x: Float[Tensor, "... position hidden"]
    ) -> Float[Tensor, "... position hidden"]:
        x = x + self.conv(x)
        x = x + self.ffn(x)
        return x


class ByteNetLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int | None = None,
        slim: bool = False,
        bias: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if hidden_size is None:
            raise ValueError("hidden_size is required")
        intermediate_size = hidden_size // 2 if slim else hidden_size
        self.layer = nn.Sequential(
            nn.LayerNorm(hidden_size, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_size, intermediate_size, bias=bias),
            nn.LayerNorm(intermediate_size, bias=bias),
            nn.GELU(),
            TransposeLayer(),
            nn.Conv1d(
                in_channels=intermediate_size,
                out_channels=intermediate_size,
                padding="same",
                bias=bias,
                **kwargs,
            ),
            TransposeLayer(),
            nn.LayerNorm(intermediate_size, bias=bias),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size, bias=bias),
        )

    def forward(
        self, x: Float[Tensor, "... position hidden"]
    ) -> Float[Tensor, "... position hidden"]:
        x = x + self.layer(x)
        return x


class OneHotEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size is None:
            raise ValueError("hidden_size is required")
        self.hidden_size = hidden_size

    def forward(
        self, x: Int[Tensor, "... position"]
    ) -> Float[Tensor, "... position hidden"]:
        return F.one_hot(x, num_classes=self.hidden_size).float()


def get_dilation_schedule(config: EncoderConfig) -> list[int]:
    return [
        min(
            config.dilation_max,
            config.dilation_base
            ** ((i % config.dilation_cycle) // config.dilation_double_every),
        )
        for i in range(config.num_hidden_layers)
    ]


class ConvNetEncoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        dilation_schedule = get_dilation_schedule(config)
        print(f"{dilation_schedule=}")
        self.layer = nn.Sequential(
            *[
                ConvLayer(
                    hidden_size=config.hidden_size,
                    kernel_size=config.first_kernel_size
                    if i == 0
                    else config.rest_kernel_size,
                    dilation=dilation_schedule[i],
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    bias=config.bias,
                    intermediate_size=config.intermediate_size,
                    groups=1
                    if (not config.depthwise or i == 0)
                    else config.hidden_size,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self, hidden_states: Float[Tensor, "... position hidden"]
    ) -> BaseModelOutput:
        hidden_states = self.layer(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class ByteNetEncoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        dilation_schedule = get_dilation_schedule(config)
        print(f"{dilation_schedule=}")
        self.layer = nn.Sequential(
            *[
                ByteNetLayer(
                    hidden_size=config.hidden_size,
                    kernel_size=config.first_kernel_size
                    if i == 0
                    else config.rest_kernel_size,
                    dilation=dilation_schedule[i],
                    bias=config.bias,
                    groups=1
                    if (not config.depthwise or i == 0)
                    else config.hidden_size,
                    slim=config.slim,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self, hidden_states: Float[Tensor, "... position hidden"]
    ) -> BaseModelOutput:
        hidden_states = self.layer(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(input_size, bias=False),
            nn.Linear(input_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )
        if input_size != output_size:
            self.shortcut = nn.Linear(input_size, output_size, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Float[Tensor, "... input"]) -> Float[Tensor, "... output"]:
        return self.shortcut(x) + self.layer(x)


class CNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        kernel_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kernel_size is None:
            raise ValueError("kernel_size is required")
        self.layer = nn.Sequential(
            nn.LayerNorm(input_size, bias=False),
            TransposeLayer(),
            nn.Conv1d(
                input_size,
                hidden_size,
                kernel_size,
                bias=False,
                padding="same",
                **kwargs,
            ),
            TransposeLayer(),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )
        if input_size != output_size:
            self.shortcut = nn.Linear(input_size, output_size, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(
        self, x: Float[Tensor, "... position input"]
    ) -> Float[Tensor, "... position output"]:
        return self.shortcut(x) + self.layer(x)
