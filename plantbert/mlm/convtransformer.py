import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel, BertConfig, BertModel
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutput


class ConvTransformerConfig(BertConfig):
    model_type = "ConvTransformer"

    def __init__(
        self,
        kernel_size=15,
        n_conv_layers=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.n_conv_layers = n_conv_layers


class ConvTransformerPreTrainedModel(PreTrainedModel):
    config_class = ConvTransformerConfig
    base_model_prefix = "model"
    #supports_gradient_checkpointing = True
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


class CompressionLayer(nn.Module):
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
                **kwargs,
            ),
            TransposeLayer(),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DecompressionLayer(nn.Module):
    def __init__(
        self,
        hidden_size=None,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            TransposeLayer(),
            nn.ConvTranspose1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                **kwargs,
            ),
            TransposeLayer(),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        x = self.conv(x)
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


class ConvTransformerModel(ConvTransformerPreTrainedModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Sequential(
            OneHotEmbedding(config.hidden_size),  # nn.Embedding(self.vocab_size, self.hidden_size)
            *[
                ConvLayer(
                    hidden_size=config.hidden_size,
                    kernel_size=config.kernel_size,
                )
            for i in range(config.n_conv_layers)]
        )
        self.compression = CompressionLayer(
            hidden_size=config.hidden_size,
            kernel_size=8,
            stride=8,
        )
        self.encoder = BertModel(config)
        self.decompression = DecompressionLayer(
            hidden_size=config.hidden_size,
            kernel_size=8,
            stride=8,
        )
        self.final_layer = ConvLayer(
            hidden_size=config.hidden_size,
            kernel_size=1,
        )
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        x = self.embedding(input_ids)
        residual = x
        #print(x.shape)
        x = self.compression(x)
        #print(x.shape)
        x = self.encoder(inputs_embeds=x).last_hidden_state
        #print(x.shape)
        x = self.decompression(x)
        #print(x.shape)
        x = x + residual
        #print(x.shape)
        x = self.final_layer(x)
        #print(x.shape)
        #raise Exception("debug")
        return BaseModelOutput(last_hidden_state=x)


class ConvTransformerOnlyMLMHead(nn.Module):
    def __init__(
        self,
        config,
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


class ConvTransformerForMaskedLM(ConvTransformerPreTrainedModel):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.config = config
        self.model = ConvTransformerModel(config)
        self.cls = ConvTransformerOnlyMLMHead(config)
        self.post_init()

    def forward(self, input_ids=None, labels=None, **kwargs):
        hidden_state = self.model(input_ids=input_ids, **kwargs).last_hidden_state
        logits = self.cls(hidden_state)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )



#config = ConvTransformerConfig(
#    vocab_size=6,
#    n_conv_layers=4,
#    kernel_size=9,
#    position_embedding_type="relative_key",
#)
#model = ConvTransformerForMaskedLM(config)
#x = torch.randint(low=0, high=5, size=(8, 512))
#y = model(input_ids=x)["logits"]
#print(x.shape, y.shape)