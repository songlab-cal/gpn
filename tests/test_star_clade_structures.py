import numpy as np
import pytest
import torch

from gpn.star.model import GPNStarConfig, GPNStarForMaskedLM


def _model(tmp_path, species, structure, layers):
    rng = np.random.default_rng(0)
    if structure == "single":
        pairwise = np.ones((species, species))
    elif structure == "one":
        pairwise = np.full((species, species), 0.1)
    elif structure == "pooled":
        group = np.arange(species) // 2
        pairwise = np.where(group[:, None] == group[None, :], 0.1, 1.0)
    else:
        group = np.arange(species) // 3
        pairwise = np.where(group[:, None] == group[None, :], 0.1, 1.0)
        pairwise[species - 1, :] = 1.0
        pairwise[:, species - 1] = 1.0
    pairwise = pairwise.astype(np.float32)
    np.fill_diagonal(pairwise, 0.0)
    np.save(tmp_path / "pairwise.npy", pairwise)
    np.save(tmp_path / "in_clade.npy", rng.random(species).astype(np.float32))
    torch.manual_seed(0)
    config = GPNStarConfig(
        phylo_dist_path=str(tmp_path),
        num_hidden_layers=layers,
        num_attention_heads=2,
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    return GPNStarForMaskedLM(config).eval()


@pytest.mark.parametrize("structure", ["single", "pooled", "one", "mixed"])
@pytest.mark.parametrize("species", [6, 24])
@pytest.mark.parametrize("layers", [1, 3])
def test_eval_forward_matches_the_reference_path(tmp_path, structure, species, layers):
    model = _model(tmp_path, species, structure, layers)
    sizes = [len(v) for v in model.model.phylo_info.clade_dict.values()]
    if structure == "single":
        assert max(sizes) == 1
    elif structure == "one":
        assert sizes == [species]
    source_ids = torch.randint(0, 5, (2, 8, species))
    input_ids = source_ids[:, :, :1].clone()
    target = torch.zeros(2, 1, dtype=torch.long)

    with torch.no_grad():
        fast = model(
            input_ids=input_ids, source_ids=source_ids, target_species=target
        ).logits
    # grad enabled takes the unreassociated path that never builds pooled_pre
    with torch.enable_grad():
        reference = model(
            input_ids=input_ids, source_ids=source_ids, target_species=target
        ).logits.detach()
    torch.testing.assert_close(fast, reference, rtol=0, atol=1e-5)


def test_pooled_pre_without_pooled_clades_is_empty(tmp_path):
    model = _model(tmp_path, 6, "single", 1)
    source = model.model.source_embedding
    clade_dict = model.model.phylo_info.clade_dict
    pre = source.pooled_pre(
        torch.randint(0, 5, (2, 8, 6)),
        clade_dict,
        model.model.phylo_info.in_clade_phylo_dist,
    )
    assert pre.shape == (2, 8, 0, source.attn_pool.all_head_size)
