import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutput
from typing import Optional, Tuple, Union

from .modules import TransposeLayer, ConvLayer, OneHotEmbedding, get_dilation_schedule


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
        print(self.dilation_schedule)
        #raise Exception("debug")
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

    def vep(self, pos=None, ref=None, alt=None, **kwargs):
        logits = self.forward(**kwargs).logits
        logits = logits[torch.arange(len(pos)), pos]
        logits_ref = logits[torch.arange(len(ref)), ref]
        logits_alt = logits[torch.arange(len(alt)), alt]
        llr = logits_alt - logits_ref
        return llr

    def get_logits(self, pos=None, **kwargs):
        logits = self.forward(**kwargs).logits
        logits = logits[torch.arange(len(pos)), pos]
        return logits


AutoConfig.register("ConvNet", ConvNetConfig)
AutoModel.register(ConvNetConfig, ConvNetModel)
AutoModelForMaskedLM.register(ConvNetConfig, ConvNetForMaskedLM)

from transformers import BertForMaskedLM, RoFormerForMaskedLM
BertForMaskedLM.vep = ConvNetForMaskedLM.vep  # so it works with DNABERT


# modifying to have weighted loss
def RoFormerForMaskedLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    loss_weight=None,
) -> Union[MaskedLMOutput, Tuple[torch.Tensor]]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
        config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
        loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.roformer(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = outputs[0]
    prediction_scores = self.cls(sequence_output)

    masked_lm_loss = None
    if labels is not None:
        logits = prediction_scores
        loss_fct = CrossEntropyLoss(reduction="none")
        labels = labels.view(-1)
        loss = loss_fct(logits.view(-1, self.config.vocab_size), labels)
        loss_weight = loss_weight.view(-1)
        loss_weight[labels==-100] = 0.0
        loss = (loss * loss_weight / loss_weight.sum()).sum()
        masked_lm_loss = loss

    if not return_dict:
        output = (prediction_scores,) + outputs[1:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    return MaskedLMOutput(
        loss=masked_lm_loss,
        logits=prediction_scores,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

RoFormerForMaskedLM.forward = RoFormerForMaskedLM_forward
RoFormerForMaskedLM.vep = ConvNetForMaskedLM.vep