import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutput
from typing import Optional, Tuple, Union

from .modules import (
    TransposeLayer, ConvLayer, OneHotEmbedding, get_dilation_schedule,
    GPNEmbedding,
)


class ConvNetConfig(PretrainedConfig):
    model_type = "ConvNet"

    def __init__(
        self,
        vocab_size=7,
        hidden_size=512,
        n_layers=25,
        kernel_size=9,
        dilation_double_every=1,
        dilation_max=32,
        dilation_cycle=6,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation_double_every = dilation_double_every
        self.dilation_max = dilation_max
        self.dilation_cycle = dilation_cycle
        self.initializer_range = initializer_range


class ConvNetPreTrainedModel(PreTrainedModel):
    config_class = ConvNetConfig
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


class ConvNetModel(ConvNetPreTrainedModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config

        self.embedding = OneHotEmbedding(config.hidden_size)

        self.dilation_schedule = get_dilation_schedule(config)
        self.encoder = nn.Sequential(*[
            ConvLayer(
                hidden_size=config.hidden_size,
                kernel_size=config.kernel_size,
                dilation=self.dilation_schedule[i],
            )
            for i in range(config.n_layers)
        ])
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        return BaseModelOutput(last_hidden_state=x)


class ConvNetOnlyMLMHead(nn.Module):
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


class ConvNetForMaskedLM(ConvNetPreTrainedModel):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.config = config
        self.model = ConvNetModel(config)
        self.cls = ConvNetOnlyMLMHead(config)
        self.post_init()

    def forward(self, input_ids=None, labels=None, loss_weight=None, **kwargs):
        hidden_state = self.model(input_ids=input_ids, **kwargs).last_hidden_state
        logits = self.cls(hidden_state)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="none")
            labels = labels.view(-1)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels)
            loss_weight = loss_weight.view(-1)
            loss_weight[labels==-100] = 0.0
            loss = (loss * loss_weight / loss_weight.sum()).sum()
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )



AutoConfig.register("ConvNet", ConvNetConfig)
AutoModel.register(ConvNetConfig, ConvNetModel)
AutoModelForMaskedLM.register(ConvNetConfig, ConvNetForMaskedLM)


from transformers import RoFormerConfig, RoFormerModel, RoFormerForMaskedLM
from transformers.models.roformer.modeling_roformer import RoFormerEncoder, RoFormerOnlyMLMHead, RoFormerSinusoidalPositionalEmbedding


class GPNRoFormerConfig(RoFormerConfig):
    model_type = "GPNRoFormer"

    def __init__(
        self,
        n_aux_features=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_aux_features = n_aux_features


class GPNRoFormerPreTrainedModel(PreTrainedModel):
    config_class = GPNRoFormerConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RoFormerSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RoFormerEncoder):
            module.gradient_checkpointing = value


class GPNRoFormerModel(GPNRoFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding = GPNEmbedding(
            config.vocab_size, config.n_aux_features, config.hidden_size,
        )
        self.encoder = RoFormerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids=None, aux_features=None, **kwargs):
        x = self.embedding(input_ids, aux_features=aux_features)
        x = self.encoder(x, **kwargs)
        return x


class GPNRoFormerForMaskedLM(GPNRoFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.model = GPNRoFormerModel(config)
        self.cls = RoFormerOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        labels=None,
        loss_weight=None,
        **kwargs
    ):
        hidden_state = self.model(**kwargs).last_hidden_state
        logits = self.cls(hidden_state)
        loss = None
        if labels is not None and loss_weight is None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size), labels.view(-1)
            )
        if labels is not None and loss_weight is not None:
            loss_fct = CrossEntropyLoss(reduction="none")
            labels = labels.view(-1)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels)
            loss_weight = loss_weight.view(-1)
            loss_weight[labels==-100] = 0.0
            loss = (loss * loss_weight / loss_weight.sum()).sum()
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )


AutoConfig.register("GPNRoFormer", GPNRoFormerConfig)
AutoModel.register(GPNRoFormerConfig, GPNRoFormerModel)
AutoModelForMaskedLM.register(GPNRoFormerConfig, GPNRoFormerForMaskedLM)
