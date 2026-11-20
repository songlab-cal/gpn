import numpy as np
import torch

from gpn.star.model import GPNStarConfig, GPNStarForMaskedLM, GPNStarSourceModule


def _all_pooled_model(tmp_path, species=6):
    pairwise = np.full((species, species), 0.1, dtype=np.float32)
    np.fill_diagonal(pairwise, 0.0)
    np.save(tmp_path / "pairwise.npy", pairwise)
    rng = np.random.default_rng(0)
    np.save(tmp_path / "in_clade.npy", rng.random(species).astype(np.float32))
    torch.manual_seed(0)
    config = GPNStarConfig(
        phylo_dist_path=str(tmp_path),
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    return GPNStarForMaskedLM(config)


def _gradients(model, species):
    torch.manual_seed(1)
    source_ids = torch.randint(0, 5, (2, 8, species))
    target = torch.tensor([[0], [1]])
    input_ids = torch.gather(
        source_ids, 2, target[:, None, :].expand(-1, 8, -1)
    ).clone()
    labels = input_ids.clone()
    labels[:, ::2] = -100
    model.zero_grad(set_to_none=True)
    model(
        input_ids=input_ids,
        source_ids=source_ids,
        target_species=target,
        labels=labels,
    ).loss.backward()
    return {n: p.grad for n, p in model.named_parameters()}


def test_embedding_table_stays_out_of_the_graph_without_singleton_clades(
    tmp_path, monkeypatch
):
    species = 6
    model = _all_pooled_model(tmp_path, species)
    clade_dict = model.model.phylo_info.clade_dict
    assert [len(v) for v in clade_dict.values()] == [species]
    grads = _gradients(model, species)
    name = "model.source_embedding.embed.weight"
    assert name in grads
    assert grads[name] is None

    # the reference path puts the table in the graph with an empty index
    split_info = GPNStarSourceModule.split_info

    def with_table(self, source_ids, clade_dict):
        multi, single, ids, _ = split_info(self, source_ids, clade_dict)
        return multi, single, ids, self.embed.weight

    monkeypatch.setattr(GPNStarSourceModule, "split_info", with_table)
    reference = _gradients(model, species)
    assert reference[name] is not None
    for key, grad in grads.items():
        if key == name:
            continue
        if grad is None:
            assert reference[key] is None
            continue
        torch.testing.assert_close(grad, reference[key], rtol=0, atol=1e-5)
    assert sum(g is not None for g in grads.values()) > 50


def test_embedding_table_keeps_its_gradient_with_singleton_clades(tmp_path):
    species = 6
    pairwise = np.ones((species, species), dtype=np.float32)
    np.fill_diagonal(pairwise, 0.0)
    np.save(tmp_path / "pairwise.npy", pairwise)
    np.save(tmp_path / "in_clade.npy", np.zeros(species, dtype=np.float32))
    torch.manual_seed(0)
    model = GPNStarForMaskedLM(
        GPNStarConfig(
            phylo_dist_path=str(tmp_path),
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=16,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )
    grads = _gradients(model, species)
    grad = grads["model.source_embedding.embed.weight"]
    assert grad is not None
    assert grad.abs().sum() > 0
