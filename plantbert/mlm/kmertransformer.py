from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel, BertConfig, BertModel
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutput


class KmerTransformerConfig(BertConfig):
    model_type = "KmerTransformer"

    def __init__(
        self,
        K=5,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.K = K
        self.initializer_range = initializer_range


class KmerTransformerPreTrainedModel(PreTrainedModel):
    config_class = KmerTransformerConfig
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


class KmerEmbedding(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.register_buffer('conv_weight', torch.empty(1, 1, config.K, dtype=torch.double))
        for i in range(config.K):
            self.conv_weight[0, 0, config.K-i-1] = config.vocab_size**i
        self.embedding = nn.Embedding(config.vocab_size**config.K, config.hidden_size)

    def forward(self, input_ids=None):
        x = rearrange(input_ids, "B L -> B 1 L")
        x = x.to(self.conv_weight.dtype)
        x = F.conv1d(x, self.conv_weight, padding="same")
        x = x.long()
        x = rearrange(x, "B 1 L -> B L")
        x = self.embedding(x)
        return x


class KmerTransformerModel(KmerTransformerPreTrainedModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config
        self.embedding = KmerEmbedding(config)
        self.encoder = BertModel(config)
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.encoder(inputs_embeds=x)
        return x


class KmerTransformerOnlyMLMHead(nn.Module):
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


class KmerTransformerForMaskedLM(KmerTransformerPreTrainedModel):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.config = config
        self.model = KmerTransformerModel(config)
        self.cls = KmerTransformerOnlyMLMHead(config)
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
