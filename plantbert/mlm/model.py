import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput


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


class ConvNetModel(nn.Module):
    def __init__(
        self,
        vocab_size=None,
        n_layers=None,
        hidden_size=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.kernel_size = 9

        #self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embedding = OneHotEmbedding(self.hidden_size)
        self.encoder = nn.Sequential(*[
            ConvLayer(
                hidden_size=self.hidden_size,
                kernel_size=self.kernel_size,
                dilation=min(i+1, 8),
            )
            for i in range(self.n_layers)
        ])

    def forward(self, input_ids=None):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        return {"hidden_states": x}


class ConvNetOnlyMLMHead(nn.Module):
    def __init__(
        self,
        vocab_size=None,
        hidden_size=None,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, hidden_states):
        return self.decoder(hidden_states)


class ConvNetForMaskedLM(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = kwargs["vocab_size"]
        self.hidden_size = kwargs["hidden_size"]
        self.model = ConvNetModel(**kwargs)
        self.cls = ConvNetOnlyMLMHead(vocab_size=self.vocab_size, hidden_size=self.hidden_size)

    def forward(self, labels=None, **kwargs):
        hidden_states = self.model(**kwargs)["hidden_states"]
        logits = self.cls(hidden_states)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )


##model = ConvNetModel(vocab_size=5, n_layers=2, hidden_size=64)
#model = ConvNetForMaskedLM(vocab_size=5, n_layers=2, hidden_size=64)
#x = torch.randint(low=0, high=5, size=(8, 100))
##y = model(x)["hidden_states"]
#y = model(input_ids=x)["logits"]
#print(x.shape, y.shape)
