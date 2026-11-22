import copy

import numpy as np
import torch

from gpn.star.model import GPNStarConfig, GPNStarForMaskedLM


def _tiny_model(tmp_path, species=24, layers=2):
    rng = np.random.default_rng(0)
    # two groups of close species so some clades hold several species
    group = np.arange(species) // 6
    pairwise = np.where(group[:, None] == group[None, :], 0.1, 1.0)
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
    return GPNStarForMaskedLM(config).eval(), species


def _logits(model, source_ids, input_ids):
    with torch.no_grad():
        return model(
            input_ids=input_ids,
            source_ids=source_ids,
            target_species=torch.zeros(input_ids.shape[0], 1, dtype=torch.long),
        ).logits


def test_param_data_edit_after_caching_does_not_leave_stale_caches(tmp_path):
    model, species = _tiny_model(tmp_path)
    source_ids = torch.randint(0, 5, (2, 8, species))
    input_ids = source_ids[:, :, :1].clone()
    _logits(model, source_ids, input_ids)

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:
                parameter.data.add_(0.05)
    fresh = copy.deepcopy(model)
    for module in fresh.modules():
        module.__dict__.pop("_composed_cache", None)
        module.__dict__.pop("_split_cache", None)

    torch.testing.assert_close(
        _logits(model, source_ids, input_ids),
        _logits(fresh, source_ids, input_ids),
        rtol=0,
        atol=1e-5,
    )


def test_in_place_and_data_assignment_edits_are_seen(tmp_path):
    model, species = _tiny_model(tmp_path)
    source_ids = torch.randint(0, 5, (2, 8, species))
    input_ids = source_ids[:, :, :1].clone()
    first = _logits(model, source_ids, input_ids)

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:
                parameter.mul_(1.1)
    second = _logits(model, source_ids, input_ids)
    assert not torch.allclose(first, second, atol=1e-6)

    for parameter in model.parameters():
        if parameter.ndim == 2:
            parameter.data = parameter.data * 0.9
    third = _logits(model, source_ids, input_ids)
    assert not torch.allclose(second, third, atol=1e-6)


def test_fused_qkv_cache_is_dropped_after_param_data_edit(tmp_path):
    # three layers: layer 1 is neither first nor last, so row q/k/v share one GEMM on CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # an earlier test can leave the legacy and new TF32 flags disagreeing
    torch.set_float32_matmul_precision("highest")
    model, species = _tiny_model(tmp_path, layers=3)
    model = model.to(device)
    source_ids = torch.randint(0, 5, (2, 8, species), device=device)
    input_ids = source_ids[:, :, :1].clone()

    def logits(m):
        with torch.no_grad():
            return m(
                input_ids=input_ids,
                source_ids=source_ids,
                target_species=torch.zeros(2, 1, dtype=torch.long, device=device),
            ).logits

    logits(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim == 2 and (
                "query" in name or "key" in name or "value" in name
            ):
                parameter.data.add_(0.05)
    fresh = copy.deepcopy(model)
    for module in fresh.modules():
        module.__dict__.pop("_composed_cache", None)
        module.__dict__.pop("_split_cache", None)
    torch.testing.assert_close(logits(model), logits(fresh), rtol=0, atol=1e-5)


def _replica(module):
    """Mimic torch.nn.parallel.replicate on CPU: parameters become plain tensor attributes."""
    copy_ = module._replicate_for_data_parallel()
    for name, child in module._modules.items():
        if child is not None:
            copy_._modules[name] = _replica(child)
    for name, parameter in module._parameters.items():
        copy_._parameters.pop(name, None)
        copy_.__dict__[name] = parameter
    return copy_


def test_data_parallel_replica_of_a_model_that_never_ran_does_not_crash(tmp_path):
    model, species = _tiny_model(tmp_path)
    source_ids = torch.randint(0, 5, (2, 8, species))
    input_ids = source_ids[:, :, :1].clone()
    replica = _replica(model)
    assert len(list(replica.model.parameters())) == 0
    assert "_fp_params" not in replica.model.__dict__
    torch.testing.assert_close(
        _logits(replica, source_ids, input_ids),
        _logits(model, source_ids, input_ids),
        rtol=0,
        atol=1e-5,
    )


def test_data_parallel_replica_drops_caches_copied_from_the_original(tmp_path):
    model, species = _tiny_model(tmp_path)
    source_ids = torch.randint(0, 5, (2, 8, species))
    input_ids = source_ids[:, :, :1].clone()
    _logits(model, source_ids, input_ids)

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:
                parameter.data.add_(0.05)
    replica = _replica(model)
    replica.model.__dict__.pop("_fp_params", None)
    replica.model.__dict__.pop("_fp_value", None)
    fresh = copy.deepcopy(model)
    for module in fresh.modules():
        module.__dict__.pop("_composed_cache", None)
        module.__dict__.pop("_split_cache", None)

    torch.testing.assert_close(
        _logits(replica, source_ids, input_ids),
        _logits(fresh, source_ids, input_ids),
        rtol=0,
        atol=1e-5,
    )
