import copy

import numpy as np
import torch

from gpn.star.model import GPNStarConfig, GPNStarForMaskedLM


def _tiny_model(tmp_path, species=24):
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
		num_hidden_layers=2,
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
