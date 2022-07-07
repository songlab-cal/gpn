import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutput


class ConvNetConfig(PretrainedConfig):
    model_type = "ConvNet"

    def __init__(
        self,
        vocab_size=6,
        hidden_size=512,
        n_layers=30,
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
    base_model_prefix = "model" #"ConvNet"
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
        min(config.dilation_max, 2**((i%config.dilation_cycle)//config.dilation_double_every))
        for i in range(config.n_layers)
    ]


class ConvNetModel(ConvNetPreTrainedModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config

        #self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embedding = OneHotEmbedding(config.hidden_size)

        #self.n_species = 18
        #self.species_embedding = nn.Embedding(self.n_species, config.hidden_size)

        self.dilation_schedule = get_dilation_schedule(config)
        self.encoder = nn.Sequential(*[
            ConvLayer(
                hidden_size=config.hidden_size,
                kernel_size=config.kernel_size,
                dilation=self.dilation_schedule[i],
                #groups=self.hidden_size,  # depthwise convolution
            )
            for i in range(config.n_layers)
        ])
        self.post_init()

    #def forward(self, input_ids=None, species_id=None, **kwargs):
    def forward(self, input_ids=None, **kwargs):
        #B, L = input_ids.shape
        #if species_id is None:
        #    species_id = torch.zeros(B, dtype=torch.int64, device=input_ids.device)

        x = self.embedding(input_ids)

        #sp_embedding = self.species_embedding(species_id)
        #sp_embedding = sp_embedding.unsqueeze(1).repeat(1, L, 1)
        #x = x + sp_embedding

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

    def forward(self, input_ids=None, labels=None, **kwargs):
        #print(input_ids.shape)
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




    #def _set_gradient_checkpointing(self, module, value=False):
        #if isinstance(module, ConvNetEncoder):
        #    module.gradient_checkpointing = value

##model = ConvNetModel(vocab_size=5, n_layers=2, hidden_size=64)
#model = ConvNetForMaskedLM(vocab_size=5, n_layers=2, hidden_size=64)
#x = torch.randint(low=0, high=5, size=(8, 100))
##y = model(x)["hidden_state"]
#y = model(input_ids=x)["logits"]
#print(x.shape, y.shape)
