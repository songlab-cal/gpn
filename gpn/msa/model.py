"""Deprecated GPN-MSA model definitions for inference compatibility."""

import torch.nn as nn
from transformers import PreTrainedModel, RoFormerConfig
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.roformer.modeling_roformer import (
    RoFormerEncoder,
    RoFormerOnlyMLMHead,
    RoFormerSinusoidalPositionalEmbedding,
)

from ..embeddings import OneHotAuxEmbedding
from ..losses import masked_lm_loss


class GPNMSAConfig(RoFormerConfig):
    """Configuration for published GPN-MSA checkpoints.

    ``model_type`` intentionally retains the historical serialized value used by
    those checkpoints. Python callers use the family-oriented ``GPNMSA`` name.
    """

    model_type = "GPNRoFormer"

    def __init__(
        self,
        vocab_size=6,
        aux_features_vocab_size=5,
        n_aux_features=0,
        group_tokens=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.aux_features_vocab_size = aux_features_vocab_size
        self.n_aux_features = n_aux_features
        self.group_tokens = group_tokens


class GPNMSAPreTrainedModel(PreTrainedModel):
    config_class = GPNMSAConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
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


class GPNMSAModel(GPNMSAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding = OneHotAuxEmbedding(config)
        self.encoder = RoFormerEncoder(config)
        self.post_init()

    def forward(self, input_ids=None, input_probs=None, aux_features=None, **kwargs):
        x = self.embedding(
            input_ids=input_ids,
            input_probs=input_probs,
            aux_features=aux_features,
        )
        return self.encoder(x, **kwargs)


class GPNMSAForMaskedLM(GPNMSAPreTrainedModel):
    # Published checkpoints contain the canonical head bias but predate modern
    # Transformers alias metadata. The decoder weight remains independent.
    _tied_weights_keys = {
        "cls.predictions.decoder.bias": "cls.predictions.bias",
    }

    def __init__(self, config):
        super().__init__(config)
        self.model = GPNMSAModel(config)
        self.cls = RoFormerOnlyMLMHead(config)
        self.post_init()

    def forward(self, labels=None, output_probs=None, loss_weight=None, **kwargs):
        hidden_state = self.model(**kwargs).last_hidden_state
        logits = self.cls(hidden_state)
        loss = masked_lm_loss(
            logits, labels, output_probs, loss_weight, self.config.vocab_size
        )
        return MaskedLMOutput(loss=loss, logits=logits)
