from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gpn.star import logits as logits_module
from gpn.star import model as model_module


def make_phylo_dist(path: Path) -> Path:
    path.mkdir(parents=True)
    for filename in model_module._PHYLO_DIST_FILENAMES:
        (path / filename).touch()
    return path


def test_explicit_phylo_dist_override_wins(tmp_path):
    configured_path = make_phylo_dist(tmp_path / "configured")
    override_path = make_phylo_dist(tmp_path / "override")
    config = SimpleNamespace(phylo_dist_path=str(configured_path))

    result = model_module._resolve_phylo_dist_path(
        "model",
        config,
        explicit_path=override_path,
    )

    assert result == str(override_path)


def test_existing_configured_phylo_dist_path_is_preserved(tmp_path):
    configured_path = make_phylo_dist(tmp_path / "configured")
    config = SimpleNamespace(phylo_dist_path=str(configured_path))

    result = model_module._resolve_phylo_dist_path("model", config)

    assert result == str(configured_path)


@pytest.mark.parametrize("configured_path", [None, "missing"])
def test_local_model_uses_bundled_phylo_dist(tmp_path, configured_path):
    model_path = tmp_path / "model"
    fallback_path = make_phylo_dist(model_path / "phylo_dist")
    config = SimpleNamespace(phylo_dist_path=configured_path)

    result = model_module._resolve_phylo_dist_path(model_path, config)

    assert result == str(fallback_path)


def test_hub_model_downloads_only_bundled_phylo_dist(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "snapshot"
    fallback_path = make_phylo_dist(snapshot_path / "weights" / "phylo_dist")
    config = SimpleNamespace(phylo_dist_path="stale/training/path")
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(snapshot_path)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )

    result = model_module._resolve_phylo_dist_path(
        "songlab/model",
        config,
        hub_kwargs={
            "_commit_hash": "resolved-commit",
            "cache_dir": tmp_path / "cache",
            "local_files_only": True,
            "revision": "main",
            "subfolder": "weights",
            "token": "token",
        },
    )

    assert result == str(fallback_path)
    assert calls == {
        "repo_id": "songlab/model",
        "allow_patterns": ["weights/phylo_dist/*.npy"],
        "cache_dir": tmp_path / "cache",
        "local_files_only": True,
        "revision": "resolved-commit",
        "token": "token",
    }


def test_invalid_explicit_override_raises(tmp_path):
    override_path = tmp_path / "incomplete"
    override_path.mkdir()
    (override_path / "pairwise.npy").touch()

    with pytest.raises(
        FileNotFoundError,
        match="must contain pairwise.npy and in_clade.npy",
    ):
        model_module._resolve_phylo_dist_path(
            "songlab/model",
            SimpleNamespace(phylo_dist_path=None),
            explicit_path=override_path,
        )


def test_missing_local_phylo_dist_raises(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()

    with pytest.raises(FileNotFoundError, match="No valid phylogenetic-distance"):
        model_module._resolve_phylo_dist_path(
            model_path,
            SimpleNamespace(phylo_dist_path="missing"),
        )


def test_phylo_info_stays_on_cpu_during_meta_model_initialization(tmp_path):
    phylo_dist_path = tmp_path / "phylo_dist"
    phylo_dist_path.mkdir()
    np.save(
        phylo_dist_path / "pairwise.npy",
        np.array([[0.0, 0.2], [0.2, 0.0]]),
    )
    np.save(phylo_dist_path / "in_clade.npy", np.array([0.0, 0.2]))
    config = SimpleNamespace(
        phylo_dist_path=str(phylo_dist_path),
        clade_thres=0.1,
    )

    with torch.device("meta"):
        phylo_info = model_module.GPNStarPhyloInfo(config)

    assert phylo_info.phylo_dist_pairwise.device.type == "cpu"
    assert phylo_info.in_clade_phylo_dist.device.type == "cpu"
    assert phylo_info.clade_labels.device.type == "cpu"


def test_masked_lm_constructs_without_tying_decoder_to_input_embedding(tmp_path):
    phylo_dist_path = tmp_path / "phylo_dist"
    phylo_dist_path.mkdir()
    np.save(
        phylo_dist_path / "pairwise.npy",
        np.array([[0.0, 0.2], [0.2, 0.0]]),
    )
    np.save(phylo_dist_path / "in_clade.npy", np.array([0.0, 0.2]))
    config = model_module.GPNStarConfig(
        phylo_dist_path=str(phylo_dist_path),
        clade_thres=0.1,
        hidden_size=8,
        embedding_size=4,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=16,
    )

    model = model_module.GPNStarForMaskedLM(config)

    predictions = model.cls.predictions
    assert predictions.decoder.bias is predictions.bias
    assert (
        predictions.decoder.weight.data_ptr()
        != model.model.target_embedding.input_embed.weight.data_ptr()
    )
    assert predictions.decoder.weight.shape == (config.vocab_size, 4)
    assert model.model.target_embedding.input_embed.weight.shape == (
        config.vocab_size,
        8,
    )


def test_auto_model_load_resolves_phylo_dist_before_parent_load(monkeypatch, tmp_path):
    phylo_dist_path = make_phylo_dist(tmp_path / "override")
    config = SimpleNamespace(phylo_dist_path="stale/training/path")
    loaded_model = object()
    calls = {}

    def fake_parent_load(cls, model_path, *model_args, config=None, **kwargs):
        calls["parent"] = (cls, model_path, model_args, config, kwargs)
        return loaded_model

    monkeypatch.setattr(
        model_module.PreTrainedModel,
        "from_pretrained",
        classmethod(fake_parent_load),
    )

    result = model_module.GPNStarForMaskedLM.from_pretrained(
        "songlab/model",
        config=config,
        phylo_dist_path=phylo_dist_path,
        revision="commit-sha",
    )

    assert result is loaded_model
    assert config.phylo_dist_path == str(phylo_dist_path)
    assert calls["parent"][1:] == (
        "songlab/model",
        (),
        config,
        {"revision": "commit-sha"},
    )


def test_direct_model_load_preserves_config_override_semantics(monkeypatch, tmp_path):
    phylo_dist_path = make_phylo_dist(tmp_path / "override")
    config = SimpleNamespace(
        phylo_dist_path=None,
        clade_thres=0.05,
        output_attentions=True,
        _commit_hash="resolved-commit",
    )
    loaded_model = object()
    calls = {}

    def fake_config_load(cls, model_path, **kwargs):
        calls["config"] = (cls, model_path, kwargs)
        return config, {"torch_dtype": "auto"}

    def fake_parent_load(cls, model_path, *model_args, config=None, **kwargs):
        calls["parent"] = (cls, model_path, model_args, config, kwargs)
        return loaded_model

    monkeypatch.setattr(
        model_module.GPNStarConfig,
        "from_pretrained",
        classmethod(fake_config_load),
    )
    monkeypatch.setattr(
        model_module.PreTrainedModel,
        "from_pretrained",
        classmethod(fake_parent_load),
    )

    result = model_module.GPNStarForMaskedLM.from_pretrained(
        "songlab/model",
        clade_thres=0.05,
        output_attentions=True,
        torch_dtype="auto",
        cache_dir=tmp_path / "cache",
        phylo_dist_path=phylo_dist_path,
    )

    assert result is loaded_model
    assert config.phylo_dist_path == str(phylo_dist_path)
    assert calls["config"][1:] == (
        "songlab/model",
        {
            "return_unused_kwargs": True,
            "clade_thres": 0.05,
            "output_attentions": True,
            "cache_dir": tmp_path / "cache",
        },
    )
    assert calls["parent"][1:] == (
        "songlab/model",
        (),
        config,
        {
            "cache_dir": tmp_path / "cache",
            "torch_dtype": "auto",
            "_commit_hash": "resolved-commit",
        },
    )


def test_mlm_for_logits_uses_auto_model_with_phylo_override(monkeypatch):
    calls = {}

    class FakeModel(logits_module.torch.nn.Module):
        def eval(self):
            calls["eval"] = True
            return self

    class FakeTokenizer:
        vocab = ("A", "C", "G", "T")

    def fake_load_model(model_path, **kwargs):
        calls["load"] = (model_path, kwargs)
        return FakeModel()

    monkeypatch.setattr(
        logits_module.AutoModelForMaskedLM,
        "from_pretrained",
        fake_load_model,
    )
    monkeypatch.setattr(logits_module, "Tokenizer", FakeTokenizer)

    wrapped_model = logits_module.MLMforLogitsModel(
        "model",
        phylo_dist_path="override",
    )

    assert calls == {
        "load": ("model", {"phylo_dist_path": "override"}),
        "eval": True,
    }
    assert isinstance(wrapped_model.model, FakeModel)
    assert (
        wrapped_model.id_a,
        wrapped_model.id_c,
        wrapped_model.id_g,
        wrapped_model.id_t,
    ) == (0, 1, 2, 3)
