import copy
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput

from gpn.model import GPNModel, GPNPreTrainedModel


class _Activation(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        if name not in ACT2FN:
            raise ValueError(f"Unsupported activation '{name}'. Available: {list(ACT2FN.keys())}")
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ACT2FN[self.name](x)


def _ensure_list_of_ints(value: Iterable[int] | None) -> List[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise TypeError(f"Expected an iterable of ints, got {type(value)}")


class ProjectionMLP(nn.Module):
    """
    Lightweight projection head inspired by SimCLR (Chen et al., 2020).

    We adopt a 2-3 layer MLP with GELU activations and dropout,
    which has empirically shown to work well for transferring
    pretrained representations to regression tasks while keeping
    the head expressive enough for per-species adaptation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: List[nn.Module] = []

        for idx in range(len(dims) - 1):
            in_dim, out_dim = dims[idx], dims[idx + 1]
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))

            is_last = idx == len(dims) - 2
            if not is_last:
                layers.append(_Activation(activation))
                if dropout and dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

        # LayerNorm prior to the head helps to stabilise fine-tuning when the
        # pooled representations change magnitude as LoRA adapters update the encoder.
        self.input_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        return self.net(x)


class SpeciesSpecificProjectionHead(nn.Module):
    """
    Species-specific projection layers.

    Each species receives its own projection MLP while sharing a common pooling strategy.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        species_to_idx: Dict[str, int],
        hidden_dims: Sequence[int],
        dropout: float = 0.1,
        activation: str = "gelu",
        pooling: str = "mean",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if len(species_to_idx) == 0:
            raise ValueError("species_to_idx must contain at least one entry.")

        self.pooling = pooling
        self.num_labels = num_labels
        self.species_to_idx = copy.deepcopy(species_to_idx)

        # Sort species by their assigned index to keep ModuleList ordering deterministic.
        ordered_species = sorted(species_to_idx.items(), key=lambda kv: kv[1])
        if ordered_species[0][1] != 0 or ordered_species[-1][1] != len(ordered_species) - 1:
            raise ValueError(
                "species_to_idx indices must be contiguous starting from 0. "
                f"Received indices: {[idx for _, idx in ordered_species]}"
            )

        self.species = [species for species, _ in ordered_species]
        hidden_dims = _ensure_list_of_ints(hidden_dims)

        self.heads = nn.ModuleList(
            [
                ProjectionMLP(
                    input_dim=hidden_size,
                    hidden_dims=hidden_dims,
                    output_dim=num_labels,
                    activation=activation,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in ordered_species
            ]
        )

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return hidden_states.mean(dim=1)
        if self.pooling in {"cls", "first"}:
            return hidden_states[:, 0, :]
        raise ValueError(f"Unsupported pooling strategy '{self.pooling}'")

    def forward(self, hidden_states: torch.Tensor, species_idx: torch.Tensor) -> torch.Tensor:
        if species_idx is None:
            raise ValueError("species_idx must be provided for species-specific projection.")

        if species_idx.dim() == 2 and species_idx.size(-1) == 1:
            species_idx = species_idx.squeeze(-1)

        if species_idx.dim() != 1:
            raise ValueError(
                f"species_idx must be a 1D tensor of shape (batch,), got shape {species_idx.shape}"
            )

        if species_idx.max().item() >= len(self.heads) or species_idx.min().item() < 0:
            raise ValueError(
                "species_idx contains an index outside the known range. "
                f"Expected values in [0, {len(self.heads) - 1}], got min={species_idx.min().item()} "
                f"max={species_idx.max().item()}"
            )

        pooled = self._pool(hidden_states)

        logits = pooled.new_empty((pooled.size(0), self.num_labels))
        unique_species = species_idx.unique(sorted=True)

        for idx in unique_species.tolist():
            mask = species_idx == idx
            logits[mask] = self.heads[idx](pooled[mask])

        return logits


class GPNForSpeciesExpression(GPNPreTrainedModel):
    """
    GPN model with LoRA-ready encoder and species-specific projection heads for expression regression.
    """

    def __init__(self, config) -> None:
        super().__init__(config)

        if not hasattr(config, "species_to_idx"):
            raise ValueError(
                "config.species_to_idx is required to build GPNForSpeciesExpression. "
                "Populate it before instantiating the model."
            )

        hidden_dims = getattr(config, "species_projection_hidden_dims", [])
        dropout = float(getattr(config, "species_projection_dropout", 0.1))
        activation = getattr(config, "species_projection_activation", "gelu")
        pooling = getattr(config, "species_projection_pooling", "mean")

        self.model = GPNModel(config)
        self.species_projection = SpeciesSpecificProjectionHead(
            hidden_size=config.hidden_size,
            num_labels=config.num_labels,
            species_to_idx=config.species_to_idx,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            pooling=pooling,
            bias=config.bias,
        )

        self.loss_regression = nn.MSELoss()
        self.regression_softplus = config.regression_softplus

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        aux_features=None,
        species_idx: torch.LongTensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        hidden_states = self.model(input_ids=input_ids, aux_features=aux_features).last_hidden_state
        logits = self.species_projection(hidden_states, species_idx=species_idx)

        if self.regression_softplus:
            logits = F.softplus(logits)

        loss = None
        if labels is not None:
            if logits.size() != labels.size():
                raise ValueError(
                    f"Shape mismatch between logits ({logits.size()}) and labels ({labels.size()}). "
                    "Ensure labels are shaped (batch, num_labels)."
                )
            loss = self.loss_regression(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

