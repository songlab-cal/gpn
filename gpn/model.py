import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from typing import Optional, Tuple, Union

from .modules import ConvNetEncoder
from transformers import RoFormerConfig
from transformers.models.roformer.modeling_roformer import (
    RoFormerEncoder,
    RoFormerSinusoidalPositionalEmbedding,
)


ENCODER_CLASS = {
    "convnet": ConvNetEncoder,
    "roformer": RoFormerEncoder,
}


class TransposeLinear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.t(self.weight), self.bias)


class GPNEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = None
        assert config.vocab_size + config.n_aux_features <= config.hidden_size

    def forward(self, input_ids=None, input_probs=None, aux_features=None):
        if input_ids is not None:
            res = F.one_hot(input_ids, num_classes=self.config.hidden_size).float()
        elif input_probs is not None:
            res = F.pad(input_probs, (0, self.config.hidden_size - self.config.vocab_size))
        else:
            raise Exception("Either input_ids or input_probs should be provided")

        if aux_features is not None:
            if self.config.aux_features_vocab_size is not None:
                aux_features = (
                    F.one_hot(
                        aux_features.long(), num_classes=self.config.aux_features_vocab_size
                    )
                    .reshape(input_ids.shape[0], input_ids.shape[1], -1)
                    .float()
                )
            res[
                :, :, self.config.vocab_size : self.config.vocab_size + self.config.n_aux_features
            ] = aux_features

        return res


class GPNEmbedding2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)

    def forward(self, input_ids):
        return self.word_embeddings(input_ids)


def compute_loss(logits, labels, output_probs, loss_weight, vocab_size):
    loss = None
    if labels is not None and loss_weight is None:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
    elif labels is not None and loss_weight is not None:
        loss_fct = CrossEntropyLoss(reduction="none")
        labels = labels.view(-1)
        loss = loss_fct(
            logits.view(-1, vocab_size), labels
        )  # what if we first exclude the ones with -100??
        loss_weight = loss_weight.view(-1)
        loss_weight[labels == -100] = 0.0
        loss = (loss * loss_weight / loss_weight.sum()).sum()
    elif output_probs is not None:
        loss_fct = CrossEntropyLoss(reduction="none")
        output_probs = output_probs.view(-1, vocab_size)
        exclude = (output_probs == 0.0).all(dim=-1)
        output_probs = output_probs[~exclude]
        logits = logits.view(-1, vocab_size)[~exclude]
        loss_weight = loss_weight.view(-1)[~exclude]
        loss = loss_fct(logits, output_probs)
        loss = (loss * loss_weight / loss_weight.sum()).sum()
    return loss


class MLMHead(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, bias=config.bias),
        )
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.bias)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, features, **kwargs):
        x = features.mean(axis=1)  # mean pooling
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


#class ClassificationHead(nn.Module):
#    """Head for sentence-level classification tasks."""
#
#    def __init__(self, config):
#        super().__init__()
#        intermediate_size = 64
#        kernel_size = 5
#        self.conv1 = nn.Sequential(
#            TransposeLayer(),
#            nn.Conv1d(
#                in_channels=config.hidden_size,
#                out_channels=intermediate_size,
#                kernel_size=kernel_size,
#                padding="same",
#            ),
#            TransposeLayer(),
#            nn.GELU(),
#        )
#        self.proj1 = nn.Linear(config.hidden_size, intermediate_size)
#        self.ln1 = nn.LayerNorm(config.intermediate_size)
#
#        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
#
#        self.config = config
#
#    def forward(self, features, **kwargs):
#        x = features.mean(axis=1)  # mean pooling
#        x = self.dropout(x)
#        x = self.dense(x)
#        x = ACT2FN[self.config.hidden_act](x)
#        x = self.dropout(x)
#        x = self.out_proj(x)
#        return x


class GPNConfig(RoFormerConfig):
    model_type = "GPN"

    def __init__(
        self,
        vocab_size=7,  # ss: 7, msa: 6
        aux_features_vocab_size=5,
        n_aux_features=0,
        encoder="convnet",  # convnet, roformer
        num_hidden_layers=25,  # roformer: 12
        hidden_size=512,  # roformer: 768
        intermediate_size=2048,  # roformer: 3072 (usually 4 * hidden_size)
        hidden_dropout_prob=0.0,
        bias=False,
        # convnet-specific
        first_kernel_size=15,
        rest_kernel_size=5,
        dilation_double_every=1,
        dilation_max=9999,
        dilation_cycle=6,
        dilation_base=2,
        pos_weight=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.aux_features_vocab_size = aux_features_vocab_size
        self.n_aux_features = n_aux_features
        self.encoder = encoder
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.bias = bias
        self.first_kernel_size = first_kernel_size
        self.rest_kernel_size = rest_kernel_size
        self.dilation_double_every = dilation_double_every
        self.dilation_max = dilation_max
        self.dilation_cycle = dilation_cycle
        self.dilation_base = dilation_base
        self.pos_weight = pos_weight


class GPNPreTrainedModel(PreTrainedModel):
    config_class = GPNConfig
    base_model_prefix = "model"
    # GB: won't try to support this for now
    # supports_gradient_checkpointing = True

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
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RoFormerEncoder):
            module.gradient_checkpointing = value


class GPNModel(GPNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = GPNEmbedding(config)
        self.encoder = ENCODER_CLASS[config.encoder](config)
        self.ln_f = nn.LayerNorm(config.hidden_size, bias=config.bias)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids=None, input_probs=None, aux_features=None, **kwargs):
        x = self.embeddings(
            input_ids=input_ids, input_probs=input_probs, aux_features=aux_features
        )
        x = self.encoder(x, **kwargs)


        # should be optional
        x = self.ln_f(x.last_hidden_state)
        x = BaseModelOutput(last_hidden_state=x)


        return x

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


class GPNForMaskedLM(GPNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = GPNModel(config)
        self.cls = MLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, labels=None, output_probs=None, loss_weight=None, **kwargs):
        hidden_state = self.model(**kwargs).last_hidden_state
        logits = self.cls(hidden_state)
        loss = compute_loss(
            logits, labels, output_probs, loss_weight, self.config.vocab_size
        )
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )

    def get_output_embeddings(self):
        # we want to prevent tying weights to None
        if self.model.get_input_embeddings() is not None:
            return self.cls.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.decoder = new_embeddings


class GPNForSequenceClassification(GPNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = GPNModel(config)
        self.classifier = ClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        aux_features=None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        hidden_state = self.model(
            input_ids=input_ids, aux_features=aux_features
        ).last_hidden_state
        logits = self.classifier(hidden_state)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


class GPNForTokenClassification(GPNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = config.pos_weight

        self.model = GPNModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(
            config.hidden_size,
            # for binary classification we'll use BCEWithLogitsLoss
            config.num_labels if config.num_labels > 2 else 1,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        aux_features=None,
        labels: Optional[torch.LongTensor] = None,
    ) -> TokenClassifierOutput:
        x = self.model(
            input_ids=input_ids, aux_features=aux_features
        ).last_hidden_state
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            assert self.num_labels == 2  # only binary implemented for now
            loss_fct = BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.pos_weight)
            )
            loss = loss_fct(torch.squeeze(logits), labels.float())

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )


AutoConfig.register("GPN", GPNConfig)
AutoModel.register(GPNConfig, GPNModel)
AutoModelForMaskedLM.register(GPNConfig, GPNForMaskedLM)
AutoModelForSequenceClassification.register(GPNConfig, GPNForSequenceClassification)
AutoModelForTokenClassification.register(GPNConfig, GPNForTokenClassification)

from .legacy import ConvNetConfig, ConvNetModel, ConvNetForMaskedLM
AutoConfig.register("ConvNet", ConvNetConfig)
AutoModel.register(ConvNetConfig, ConvNetModel)
AutoModelForMaskedLM.register(ConvNetConfig, ConvNetForMaskedLM)

from .legacy import GPNRoFormerConfig, GPNRoFormerModel, GPNRoFormerForMaskedLM
AutoConfig.register("GPNRoFormer", GPNRoFormerConfig)
AutoModel.register(GPNRoFormerConfig, GPNRoFormerModel)
AutoModelForMaskedLM.register(GPNRoFormerConfig, GPNRoFormerForMaskedLM)
