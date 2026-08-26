"""Embedding layers shared by GPN-SS and GPN-MSA."""

from typing import Protocol

import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


class AuxEmbeddingConfig(Protocol):
    """Configuration fields consumed by :class:`OneHotAuxEmbedding`."""

    vocab_size: int
    aux_features_vocab_size: int | None
    n_aux_features: int
    hidden_size: int


class OneHotAuxEmbedding(nn.Module):
    """Embed nucleotide tokens and optional auxiliary features without weights."""

    def __init__(self, config: AuxEmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings: nn.Embedding | None = None
        assert config.vocab_size + config.n_aux_features <= config.hidden_size

    def forward(
        self,
        input_ids: Int[Tensor, "... position"] | None = None,
        input_probs: Float[Tensor, "... position nucleotide"] | None = None,
        aux_features: Int[Tensor, "... position auxiliary"] | None = None,
    ) -> Float[Tensor, "... position hidden"]:
        if input_ids is not None:
            result = F.one_hot(input_ids, num_classes=self.config.hidden_size).float()
        elif input_probs is not None:
            result = F.pad(
                input_probs, (0, self.config.hidden_size - self.config.vocab_size)
            )
        else:
            raise ValueError("Either input_ids or input_probs must be provided")

        if aux_features is not None:
            if self.config.aux_features_vocab_size is not None:
                aux_features = (
                    F.one_hot(
                        aux_features.long(),
                        num_classes=self.config.aux_features_vocab_size,
                    )
                    .reshape(result.shape[0], result.shape[1], -1)
                    .float()
                )
            result[
                :,
                :,
                self.config.vocab_size : self.config.vocab_size
                + self.config.n_aux_features,
            ] = aux_features

        return result
