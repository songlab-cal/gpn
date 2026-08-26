from types import SimpleNamespace

import torch

from gpn.embeddings import OneHotAuxEmbedding


def test_one_hot_embedding_accepts_continuous_auxiliary_features() -> None:
    config = SimpleNamespace(
        vocab_size=6,
        n_aux_features=2,
        hidden_size=8,
        aux_features_vocab_size=None,
    )
    embedding = OneHotAuxEmbedding(config)
    input_ids = torch.tensor([[1, 2]])
    auxiliary = torch.tensor([[[0.25, 0.75], [0.5, 0.5]]])

    actual = embedding(input_ids=input_ids, aux_features=auxiliary)

    assert actual.shape == (1, 2, 8)
    torch.testing.assert_close(actual[..., 6:8], auxiliary)
