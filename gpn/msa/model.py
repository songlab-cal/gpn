from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel, BertConfig, BertLayer
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutput


class MSAConvNetConfig(PretrainedConfig):
    model_type = "MSAConvNet"

    def __init__(
        self,
        vocab_size=8,
        n_rows=7,
        hidden_size=256,
        n_layers=9,
        kernel_size=9,
        dilation_double_every=1,
        dilation_max=9999,
        dilation_cycle=4,
        initializer_range=0.02,
        transformer_n_heads=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_rows = n_rows
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation_double_every = dilation_double_every
        self.dilation_max = dilation_max
        self.dilation_cycle = dilation_cycle
        self.transformer_n_heads = transformer_n_heads
        self.initializer_range = initializer_range


class MSAConvNetPreTrainedModel(PreTrainedModel):
    config_class = MSAConvNetConfig
    base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class TransposeLayer(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self, hidden_size=None, **kwargs,
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
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        b, r, c, h = x.shape
        x = rearrange(x, "b r c h -> (b r) c h")
        x = x + self.conv(x)
        x = rearrange(x, "(b r) c h -> b r c h", b=b)
        x = x + self.ffn(x)
        return x


# inspired by Convolutional GNN:
# https://youtu.be/uF53xsT7mjc?t=1269
class SetConvLayer(nn.Module):
    def __init__(
        self, hidden_size=None, **kwargs,
    ):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size),
        )
        self.ffn3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        b, r, c, h = x.shape
        v = self.ffn1(x)  # should try skipping this
        v = reduce(v, "b r c h -> b r h", "mean")
        v = self.ffn2(v)
        v = repeat(v, "b r h -> b r c h", c=c)
        x = x + v
        x = x + self.ffn3(x)
        return x


class RowAttentionLayer(nn.Module):
    def __init__(
        self, hidden_size=None, nhead=None, **kwargs,
    ):
        super().__init__()
        config = BertConfig(
            hidden_size=hidden_size,
            intermediate_size=4*hidden_size,
            num_attention_heads=nhead,
        )
        self.layer = BertLayer(config)

    def forward(self, x):
        b, r, c, h = x.shape
        x = rearrange(x, "b r c h -> (b c) r h")
        x = self.layer(x)[0]
        x = rearrange(x, "(b c) r h -> b r c h", b=b)
        return x


class OneHotEmbedding(nn.Module):
    def __init__(
        self, hidden_size=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        return F.one_hot(x, num_classes=self.hidden_size).float()


def get_dilation_schedule(config):
    return [
        min(
            config.dilation_max,
            2 ** ((i % config.dilation_cycle) // config.dilation_double_every),
        )
        for i in range(config.n_layers)
    ]


class MSAConvNetModel(MSAConvNetPreTrainedModel):
    def __init__(
        self, config, **kwargs,
    ):
        super().__init__(config)
        self.config = config

        self.embedding = OneHotEmbedding(config.hidden_size)
        self.row_embedding = nn.Embedding(config.n_rows, config.hidden_size)

        self.dilation_schedule = get_dilation_schedule(config)
        print(self.dilation_schedule)
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(
                    ConvLayer(
                        hidden_size=config.hidden_size,
                        kernel_size=config.kernel_size,
                        dilation=self.dilation_schedule[i],
                    ),
                    #SetConvLayer(hidden_size=config.hidden_size),
                    RowAttentionLayer(hidden_size=config.hidden_size, nhead=config.transformer_n_heads),
                )
                for i in range(config.n_layers)
            ]
        )
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        x = self.embedding(input_ids)
        b, r, c, h = x.shape
        x += repeat(self.row_embedding.weight, "r h -> b r c h", b=b, c=c)
        x = self.encoder(x)
        return BaseModelOutput(last_hidden_state=x)


class MSAConvNetOnlyMLMHead(nn.Module):
    def __init__(
        self, config,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size),
        )

    def forward(self, hidden_state):
        return self.decoder(hidden_state)


class MSAConvNetForMaskedLM(MSAConvNetPreTrainedModel):
    def __init__(
        self, config,
    ):
        super().__init__(config)
        self.config = config
        self.model = MSAConvNetModel(config)
        self.cls = MSAConvNetOnlyMLMHead(config)
        self.post_init()

    def forward(self, input_ids=None, labels=None, **kwargs):
        hidden_state = self.model(input_ids=input_ids, **kwargs).last_hidden_state
        logits = self.cls(hidden_state)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return MaskedLMOutput(loss=loss, logits=logits,)


# import torchinfo
# config = MSAConvNetConfig()
# model = MSAConvNetModel(config)
# model = MSAConvNetForMaskedLM(config)
# print(torchinfo.summary(model))
# x = torch.randint(config.vocab_size, (3, 6, 64))
# print(x.shape)
# y = model(input_ids=x).last_hidden_state
# y = model(input_ids=x).logits
# print(y.shape)
