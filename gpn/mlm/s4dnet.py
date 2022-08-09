import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutput

from s4d import S4D


class S4DNetConfig(PretrainedConfig):
    model_type = "S4DNet"

    def __init__(
        self,
        vocab_size=6,
        hidden_size=256,
        n_layers=4,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range


class S4DNetPreTrainedModel(PreTrainedModel):
    config_class = S4DNetConfig
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


class OneHotEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        return F.one_hot(x, num_classes=self.hidden_size).float()


class S4DNetModel(S4DNetPreTrainedModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config

        #self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embedding = OneHotEmbedding(config.hidden_size)
        self.encoder = nn.Sequential(*[
            S4D(
                d_model=config.hidden_size,
                bidirectional=True,
                dropout=0.1,
                transposed=False,
            )
            for i in range(config.n_layers)
        ])
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        return BaseModelOutput(last_hidden_state=x)


class S4DNetOnlyMLMHead(nn.Module):
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


class S4DNetForMaskedLM(S4DNetPreTrainedModel):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.config = config
        self.model = S4DNetModel(config)
        self.cls = S4DNetOnlyMLMHead(config)
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