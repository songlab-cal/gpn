from pathlib import Path

import torch

from gpn.legacy import GPNRoFormerConfig, GPNRoFormerForMaskedLM


def tiny_config() -> GPNRoFormerConfig:
    return GPNRoFormerConfig(
        vocab_size=6,
        hidden_size=16,
        embedding_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=16,
        n_aux_features=5,
    )


def test_gpn_roformer_ties_only_the_decoder_bias():
    config = tiny_config()
    model = GPNRoFormerForMaskedLM(config)
    predictions = model.cls.predictions

    assert model._tied_weights_keys == {
        "cls.predictions.decoder.bias": "cls.predictions.bias"
    }
    assert predictions.decoder.bias is predictions.bias
    assert predictions.decoder.weight.shape == (config.vocab_size, 16)


def test_published_style_checkpoint_restores_decoder_bias(tmp_path: Path):
    """Exercise the Transformers 5 alias-aware checkpoint loading path."""

    config = tiny_config()
    model = GPNRoFormerForMaskedLM(config)
    learned_bias = torch.tensor([-0.05, 0.01, 0.02, 0.03, 0.04, -0.06])
    state_dict = model.state_dict()
    state_dict["cls.predictions.bias"] = learned_bias
    # Published GPN-MSA checkpoints predate alias metadata and contain only the
    # canonical head bias, not its decoder alias.
    del state_dict["cls.predictions.decoder.bias"]
    config.save_pretrained(tmp_path)
    torch.save(state_dict, tmp_path / "pytorch_model.bin")

    restored = GPNRoFormerForMaskedLM.from_pretrained(
        tmp_path,
        use_safetensors=False,
    )
    predictions = restored.cls.predictions

    assert predictions.decoder.bias is predictions.bias
    torch.testing.assert_close(predictions.bias, learned_bias)
