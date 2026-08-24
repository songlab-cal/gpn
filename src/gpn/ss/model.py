import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    RoFormerConfig,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.models.roformer.modeling_roformer import (
    RoFormerEncoder,
    RoFormerSinusoidalPositionalEmbedding,
)

from ..embeddings import OneHotAuxEmbedding
from ..losses import masked_lm_loss
from .modules import CNN, MLP, ByteNetEncoder, ConvNetEncoder

ENCODER_CLASS = {
    "bytenet": ByteNetEncoder,
    "convnet": ConvNetEncoder,
    "roformer": RoFormerEncoder,
}


class TransposeLinear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.t(self.weight), self.bias)


class GPNEmbedding2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )

    def forward(self, input_ids, **kwargs):
        return self.word_embeddings(input_ids)


EMBEDDING_CLASS = {
    "one_hot": OneHotAuxEmbedding,
    "embedding": GPNEmbedding2,
}


class MLMHead(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        if config.mlm_head_transform:
            self.transform = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias),
                nn.GELU(),
                nn.LayerNorm(config.hidden_size, bias=config.bias),
            )
        else:
            self.transform = nn.Identity()
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.bias
        )

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class StandardClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.ln = nn.LayerNorm(config.hidden_size, bias=config.bias)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(
            config.hidden_size, config.num_labels, bias=config.bias
        )

        self.config = config

    def forward(self, features, **kwargs):
        x = features.mean(axis=1)  # mean pooling
        x = self.ln(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LightweightCNNClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        intermediate_size = 64
        kernel_size = 3
        self.conv1 = CNN(
            config.hidden_size,
            intermediate_size,
            intermediate_size,
            kernel_size=kernel_size,
        )
        self.conv2 = CNN(
            intermediate_size,
            intermediate_size,
            intermediate_size,
            kernel_size=kernel_size,
        )
        self.mlp = MLP(
            intermediate_size,
            intermediate_size,
            intermediate_size,
        )
        self.ln = nn.LayerNorm(intermediate_size, bias=False)
        self.final = nn.Linear(intermediate_size, config.num_labels, bias=False)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.mean(axis=1)
        x = self.mlp(x)
        x = self.ln(x)
        x = self.final(x)
        return x


class LightweightMLPClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        intermediate_size = 64
        self.mlp1 = MLP(
            config.hidden_size,
            intermediate_size,
            intermediate_size,
        )
        self.mlp2 = MLP(
            intermediate_size,
            intermediate_size,
            intermediate_size,
        )
        self.ln = nn.LayerNorm(intermediate_size, bias=False)
        self.final = nn.Linear(intermediate_size, config.num_labels, bias=False)

    def forward(self, x, **kwargs):
        x = self.mlp1(x)
        x = x.mean(axis=1)
        x = self.mlp2(x)
        x = self.ln(x)
        x = self.final(x)
        return x


CLASSIFICATION_HEAD_CLASS = {
    "standard": StandardClassificationHead,
    "lightweight_mlp": LightweightMLPClassificationHead,
    "lightweight_cnn": LightweightCNNClassificationHead,
}


class GPNConfig(RoFormerConfig):
    model_type = "GPN"

    def __init__(
        self,
        vocab_size=7,  # ss: 7, msa: 6
        aux_features_vocab_size=5,
        n_aux_features=0,
        embedding="one_hot",  # one_hot, embedding
        encoder="convnet",  # convnet, roformer, bytenet
        num_hidden_layers=25,  # roformer: 12
        hidden_size=512,  # roformer: 768
        intermediate_size=2048,  # roformer: 3072 (usually 4 * hidden_size)
        hidden_dropout_prob=0.0,
        bias=False,
        tie_word_embeddings=False,
        mlm_head_transform=True,
        # bytenet-specific
        slim=False,
        # convnet-specific
        first_kernel_size=9,
        rest_kernel_size=5,
        dilation_double_every=1,
        dilation_max=9999,
        dilation_cycle=8,
        dilation_base=2,
        depthwise=False,
        # specific to head for downstream classification/regression task
        classification_head="standard",
        pos_weight=None,
        regression_softplus=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.aux_features_vocab_size = aux_features_vocab_size
        self.n_aux_features = n_aux_features
        self.embedding = embedding
        self.encoder = encoder
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.bias = bias
        self.tie_word_embeddings = tie_word_embeddings
        self.mlm_head_transform = mlm_head_transform
        self.slim = slim
        self.first_kernel_size = first_kernel_size
        self.rest_kernel_size = rest_kernel_size
        self.dilation_double_every = dilation_double_every
        self.dilation_max = dilation_max
        self.dilation_cycle = dilation_cycle
        self.dilation_base = dilation_base
        self.depthwise = depthwise
        self.classification_head = classification_head
        self.pos_weight = pos_weight
        self.regression_softplus = regression_softplus


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
        self.embeddings = EMBEDDING_CLASS[config.embedding](config)
        self.encoder = ENCODER_CLASS[config.encoder](config)
        self.ln_f = nn.LayerNorm(config.hidden_size, bias=config.bias)

        # Fix: https://github.com/huggingface/transformers/issues/40564
        self.accepts_loss_kwargs = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids=None, input_probs=None, aux_features=None, **kwargs):
        x = self.embeddings(
            input_ids=input_ids, input_probs=input_probs, aux_features=aux_features
        )
        x = self.encoder(x)

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

        # Fix: https://github.com/huggingface/transformers/issues/40564
        self.accepts_loss_kwargs = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, labels=None, output_probs=None, loss_weight=None, **kwargs):
        hidden_state = self.model(**kwargs).last_hidden_state
        logits = self.cls(hidden_state)
        loss = masked_lm_loss(
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
        self.classifier = CLASSIFICATION_HEAD_CLASS[config.classification_head](config)
        self.regression_softplus = config.regression_softplus

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        aux_features=None,
        labels: torch.LongTensor | None = None,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor]:
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
        if self.regression_softplus:
            logits = F.softplus(logits)

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
                if self.num_labels == 1:
                    loss = loss_fct(torch.squeeze(logits), labels)
                else:
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
        input_ids: torch.LongTensor | None = None,
        aux_features=None,
        labels: torch.LongTensor | None = None,
    ) -> TokenClassifierOutput:
        x = self.model(input_ids=input_ids, aux_features=aux_features).last_hidden_state
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            assert self.num_labels == 2  # only binary implemented for now
            loss_fct = BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
            loss = loss_fct(torch.squeeze(logits), labels.float())

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )


class ConvNetMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size),
        )

    def forward(self, hidden_state):
        return self.decoder(hidden_state)


class ConvNetClassificationHead(nn.Module):
    """Head used by the published sorghum sequence-classification models."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, features, **kwargs):
        x = features.mean(axis=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        return self.out_proj(x)


class ConvNetConfig(PretrainedConfig):
    """Configuration used by the original GPN and sorghum checkpoints."""

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
        dilation_base=2,
        initializer_range=0.02,
        n_aux_features=0,
        aux_features_vocab_size=5,
        hidden_dropout_prob=0.1,
        hidden_act="gelu",
        regression_softplus=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation_double_every = dilation_double_every
        self.dilation_max = dilation_max
        self.dilation_cycle = dilation_cycle
        self.dilation_base = dilation_base
        self.initializer_range = initializer_range
        self.n_aux_features = n_aux_features
        self.aux_features_vocab_size = aux_features_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.regression_softplus = regression_softplus


class ConvNetPreTrainedModel(PreTrainedModel):
    config_class = ConvNetConfig
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
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


class _TransposeLayer(nn.Module):
    def forward(self, x):
        return torch.transpose(x, 1, 2)


class _ConvLayer(nn.Module):
    def __init__(self, hidden_size=None, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            _TransposeLayer(),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                padding="same",
                **kwargs,
            ),
            _TransposeLayer(),
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
        return x + self.ffn(x)


def _get_dilation_schedule(config):
    return [
        min(
            config.dilation_base ** (i // config.dilation_double_every),
            config.dilation_max,
        )
        for i in range(config.dilation_cycle)
    ] * (config.n_layers // config.dilation_cycle + 1)


class ConvNetModel(ConvNetPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.embedding = OneHotAuxEmbedding(config)
        self.dilation_schedule = _get_dilation_schedule(config)
        self.encoder = nn.Sequential(
            *[
                _ConvLayer(
                    hidden_size=config.hidden_size,
                    kernel_size=config.kernel_size,
                    dilation=self.dilation_schedule[index],
                )
                for index in range(config.n_layers)
            ]
        )
        self.post_init()

    def forward(self, input_ids=None, input_probs=None, aux_features=None, **kwargs):
        x = self.embedding(
            input_ids=input_ids,
            input_probs=input_probs,
            aux_features=aux_features,
        )
        return BaseModelOutput(last_hidden_state=self.encoder(x))


class ConvNetForMaskedLM(ConvNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = ConvNetModel(config)
        self.cls = ConvNetMLMHead(config)
        self.post_init()

    def forward(self, labels=None, output_probs=None, loss_weight=None, **kwargs):
        hidden_state = self.model(**kwargs).last_hidden_state
        logits = self.cls(hidden_state)
        loss = masked_lm_loss(
            logits, labels, output_probs, loss_weight, self.config.vocab_size
        )
        return MaskedLMOutput(loss=loss, logits=logits)


class ConvNetForSequenceClassification(ConvNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = ConvNetModel(config)
        self.classifier = ConvNetClassificationHead(config)
        self.regression_softplus = config.regression_softplus
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor]:
        hidden_state = self.model(input_ids=input_ids).last_hidden_state
        logits = self.classifier(hidden_state)
        if self.regression_softplus:
            logits = F.softplus(logits)

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

        return SequenceClassifierOutput(loss=loss, logits=logits)
