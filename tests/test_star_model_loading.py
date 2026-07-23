from types import SimpleNamespace

import pytest

from gpn.star import logits as logits_module
from gpn.star import model as model_module


def mock_huggingface_load(monkeypatch, configured_phylo_dist_path):
    config = SimpleNamespace(phylo_dist_path=configured_phylo_dist_path)
    loaded_model = object()
    calls = {}

    def fake_load_config(model_path):
        calls["config_model_path"] = model_path
        return config

    def fake_load_model(model_path, *, config):
        calls["model_path"] = model_path
        calls["config"] = config
        return loaded_model

    monkeypatch.setattr(
        model_module.AutoConfig,
        "from_pretrained",
        fake_load_config,
    )
    monkeypatch.setattr(
        model_module.AutoModelForMaskedLM,
        "from_pretrained",
        fake_load_model,
    )
    return config, loaded_model, calls


def test_explicit_phylo_dist_override_wins(monkeypatch, tmp_path):
    configured_path = tmp_path / "configured"
    configured_path.mkdir()
    override_path = tmp_path / "override"
    override_path.mkdir()
    config, loaded_model, calls = mock_huggingface_load(
        monkeypatch,
        str(configured_path),
    )

    result = model_module.load_model(
        "model",
        phylo_dist_path=str(override_path),
    )

    assert result is loaded_model
    assert config.phylo_dist_path == str(override_path)
    assert calls == {
        "config_model_path": "model",
        "model_path": "model",
        "config": config,
    }


def test_explicit_pathlike_override_is_normalized(monkeypatch, tmp_path):
    configured_path = tmp_path / "configured"
    configured_path.mkdir()
    override_path = tmp_path / "override"
    override_path.mkdir()
    config, _, _ = mock_huggingface_load(monkeypatch, str(configured_path))

    model_module.load_model("model", phylo_dist_path=override_path)

    assert config.phylo_dist_path == str(override_path)


def test_existing_configured_phylo_dist_path_is_preserved(monkeypatch, tmp_path):
    configured_path = tmp_path / "configured"
    configured_path.mkdir()
    config, loaded_model, calls = mock_huggingface_load(
        monkeypatch,
        str(configured_path),
    )

    result = model_module.load_model("model")

    assert result is loaded_model
    assert config.phylo_dist_path == str(configured_path)
    assert calls["config"] is config


@pytest.mark.parametrize("missing_attribute", [False, True])
def test_null_or_missing_configured_path_uses_model_fallback(
    monkeypatch,
    tmp_path,
    missing_attribute,
):
    model_path = tmp_path / "model"
    fallback_path = model_path / "phylo_dist"
    fallback_path.mkdir(parents=True)
    config, loaded_model, _ = mock_huggingface_load(monkeypatch, None)
    if missing_attribute:
        del config.phylo_dist_path

    result = model_module.load_model(model_path)

    assert result is loaded_model
    assert config.phylo_dist_path == str(fallback_path)


def test_missing_configured_path_uses_model_fallback(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    fallback_path = model_path / "phylo_dist"
    fallback_path.mkdir(parents=True)
    config, loaded_model, calls = mock_huggingface_load(
        monkeypatch,
        str(tmp_path / "missing-configured"),
    )

    result = model_module.load_model(str(model_path))

    assert result is loaded_model
    assert config.phylo_dist_path == str(fallback_path)
    assert calls["model_path"] == str(model_path)


def test_invalid_explicit_override_raises_even_when_fallback_exists(
    monkeypatch,
    tmp_path,
):
    model_path = tmp_path / "model"
    (model_path / "phylo_dist").mkdir(parents=True)
    mock_huggingface_load(monkeypatch, str(tmp_path / "configured"))
    missing_override = tmp_path / "missing-override"

    with pytest.raises(
        FileNotFoundError,
        match="Phylogenetic-distance directory does not exist",
    ):
        model_module.load_model(
            str(model_path),
            phylo_dist_path=str(missing_override),
        )


def test_missing_configured_and_fallback_paths_raise(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    configured_path = tmp_path / "missing-configured"
    mock_huggingface_load(monkeypatch, str(configured_path))

    with pytest.raises(
        FileNotFoundError,
        match="fallback directory",
    ) as error:
        model_module.load_model(str(model_path))

    assert str(configured_path) in str(error.value)
    assert str(model_path / "phylo_dist") in str(error.value)


def test_mlm_for_logits_forwards_phylo_dist_override(monkeypatch):
    calls = {}

    class FakeModel(logits_module.torch.nn.Module):
        def eval(self):
            calls["eval"] = True
            return self

    class FakeTokenizer:
        vocab = ["A", "C", "G", "T"]

    def fake_load_model(model_path, phylo_dist_path=None):
        calls["load"] = (model_path, phylo_dist_path)
        return FakeModel()

    monkeypatch.setattr(logits_module, "load_model", fake_load_model)
    monkeypatch.setattr(logits_module, "Tokenizer", FakeTokenizer)

    wrapped_model = logits_module.MLMforLogitsModel(
        "model",
        phylo_dist_path="override",
    )

    assert calls == {
        "load": ("model", "override"),
        "eval": True,
    }
    assert isinstance(wrapped_model.model, FakeModel)
    assert (
        wrapped_model.id_a,
        wrapped_model.id_c,
        wrapped_model.id_g,
        wrapped_model.id_t,
    ) == (0, 1, 2, 3)
