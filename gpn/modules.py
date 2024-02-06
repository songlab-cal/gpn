import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class TransposeLayer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        hidden_size=None,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            TransposeLayer(),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                padding="same",
                **kwargs,
            ),
            TransposeLayer(),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.ffn(x)
        return x


class OneHotEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        return F.one_hot(x, num_classes=self.hidden_size).float()


def get_dilation_schedule(config):
    return [
        min(
            config.dilation_max,
            config.dilation_base
            ** ((i % config.dilation_cycle) // config.dilation_double_every),
        )
        for i in range(config.n_layers)
    ]


class ConvNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        dilation_schedule = get_dilation_schedule(config)
        self.layer = nn.Sequential(
            *[
                ConvLayer(
                    hidden_size=config.hidden_size,
                    kernel_size=config.kernel_size,
                    dilation=dilation_schedule[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(self, hidden_states):
        hidden_states = self.layer(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)
