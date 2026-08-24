"""Embedding layers shared by GPN-SS and GPN-MSA."""

import torch.nn as nn
import torch.nn.functional as F


class OneHotAuxEmbedding(nn.Module):
    """Embed nucleotide tokens and optional auxiliary features without weights."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = None
        assert config.vocab_size + config.n_aux_features <= config.hidden_size

    def forward(self, input_ids=None, input_probs=None, aux_features=None):
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
