from __future__ import annotations

import math
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Self, overload

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, RoFormerConfig, apply_chunking_to_forward
from transformers.modeling_outputs import (
    MaskedLMOutput,
)
from transformers.models.roformer.modeling_roformer import (
    RoFormerEncoder,
    RoFormerLayer,
    RoFormerOnlyMLMHead,
    RoFormerSinusoidalPositionalEmbedding,
)
from transformers.utils import ModelOutput


class GPNStarConfig(RoFormerConfig):
    model_type = "GPNStar"

    def __init__(
        self,
        vocab_size: int = 6,
        time_enc: str = "fire_1",
        phylo_dist_path: str | None = None,
        clade_thres: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.time_enc = time_enc
        self.max_evol_dist: float | None = None
        self.phylo_dist_path = phylo_dist_path
        self.clade_thres = clade_thres


_PHYLO_DIST_FILENAMES = ("pairwise.npy", "in_clade.npy")


def _parse_time_encoding(value: str) -> tuple[str, int]:
    encoding, scale = value.split("_")
    return encoding, int(scale)


def _contains_phylo_dist_files(path: str | os.PathLike[str] | None) -> bool:
    return path is not None and all(
        os.path.isfile(os.path.join(path, filename))
        for filename in _PHYLO_DIST_FILENAMES
    )


def _resolve_phylo_dist_path(
    pretrained_model_name_or_path: str | os.PathLike[str],
    config: GPNStarConfig,
    *,
    explicit_path: str | os.PathLike[str] | None = None,
    hub_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Resolve local or Hub-hosted phylogenetic-distance model assets."""
    if explicit_path is not None:
        explicit_path = os.fsdecode(os.fspath(explicit_path))
        if not _contains_phylo_dist_files(explicit_path):
            raise FileNotFoundError(
                "Phylogenetic-distance directory must contain pairwise.npy and "
                f"in_clade.npy: '{explicit_path}'"
            )
        return explicit_path

    configured_path = getattr(config, "phylo_dist_path", None)
    if configured_path is not None:
        configured_path = os.fsdecode(os.fspath(configured_path))
    if _contains_phylo_dist_files(configured_path):
        assert configured_path is not None
        return configured_path

    model_path = os.fsdecode(os.fspath(pretrained_model_name_or_path))
    subfolder = ((hub_kwargs or {}).get("subfolder") or "").strip("/")
    local_fallback = os.path.join(model_path, subfolder, "phylo_dist")
    if _contains_phylo_dist_files(local_fallback):
        return local_fallback
    if os.path.exists(model_path):
        raise FileNotFoundError(
            "No valid phylogenetic-distance assets found. Expected pairwise.npy "
            f"and in_clade.npy in '{local_fallback}'."
        )

    from huggingface_hub import snapshot_download

    asset_dir = "/".join(part for part in (subfolder, "phylo_dist") if part)
    download_kwargs: dict[str, Any] = {
        key: (hub_kwargs or {})[key]
        for key in (
            "cache_dir",
            "force_download",
            "local_files_only",
            "token",
        )
        if key in (hub_kwargs or {})
    }
    revision = (hub_kwargs or {}).get("_commit_hash") or (hub_kwargs or {}).get(
        "revision"
    )
    revision = revision or getattr(config, "_commit_hash", None)
    if revision is not None:
        download_kwargs["revision"] = revision
    snapshot_path = snapshot_download(
        repo_id=model_path,
        allow_patterns=[f"{asset_dir}/*.npy"],
        **download_kwargs,
    )
    hub_fallback = os.path.join(snapshot_path, asset_dir)
    if not _contains_phylo_dist_files(hub_fallback):
        raise FileNotFoundError(
            f"Hugging Face model '{model_path}' does not bundle both required "
            f"files under '{asset_dir}'."
        )
    return hub_fallback


_HI_MASK = -8192  # int32 0xFFFFE000: keeps sign, exponent and 10 mantissa bits


def _split_hi_lo(t: Tensor) -> tuple[Tensor, Tensor]:
	hi = (t.contiguous().view(torch.int32) & _HI_MASK).view(torch.float32)
	return hi, t - hi


def _split_weight(linear: nn.Linear) -> tuple[Tensor, Tensor]:
	"""hi/lo halves of linear.weight, cached while the weight is unchanged."""
	w = linear.weight
	stamp = (w._version, w.data_ptr(), w.device)
	hit = linear.__dict__.get("_split_cache")
	if hit is None or hit[0] != stamp:
		with torch.no_grad():
			hit = (stamp, _split_hi_lo(w))
		linear.__dict__["_split_cache"] = hit
	return hit[1]


def _compensated_ok(linear: nn.Linear, x: Tensor) -> bool:
	return not (
		x.dtype != torch.float32
		or not x.is_cuda
		or torch.is_grad_enabled()
		or linear.weight.dtype != torch.float32
		or torch.cuda.get_device_capability(x.device)[0] < 8
	)


def compensated_linear_cat(owner: nn.Module, linears: tuple, x: Tensor):
	"""Outputs of several same-input Linears from one compensated GEMM.

	The concatenated hi/lo weight is cached in owner's _composed_cache, which
	GPNStarModel._drop_stale_weight_caches pops. Returns None when the plain
	path applies.
	"""
	if not all(_compensated_ok(l, x) for l in linears):
		return None
	weights = [l.weight for l in linears]
	stamp = tuple((w._version, w.data_ptr(), w.device) for w in weights)
	cache = owner.__dict__.setdefault("_composed_cache", {})
	hit = cache.get("qkv")
	if hit is None or hit[0] != stamp:
		with torch.no_grad():
			hi, lo = _split_hi_lo(torch.cat(weights, dim=0))
			bias = torch.cat([l.bias for l in linears]) if linears[0].bias is not None else None
		hit = cache["qkv"] = (stamp, hi, lo, bias)
	_, w_hi, w_lo, bias = hit
	x_hi, x_lo = _split_hi_lo(x)
	prev = torch.backends.cuda.matmul.allow_tf32
	torch.backends.cuda.matmul.allow_tf32 = True
	try:
		out = F.linear(x_hi, w_hi, bias)
		out = out + F.linear(x_hi, w_lo)
		out = out + F.linear(x_lo, w_hi)
	finally:
		torch.backends.cuda.matmul.allow_tf32 = prev
	return out.split([l.out_features for l in linears], dim=-1)


def compensated_linear(linear: nn.Linear, x: Tensor) -> Tensor:
	"""linear(x) as three TF32 tensor-core GEMMs (hi*hi + hi*lo + lo*hi).

	Each operand is split into a 10-bit-mantissa part and the remainder, so the
	result is within about 1e-6 relative of fp32. Plain fp32 on CPU, below
	compute capability 8, with grad, or for non-fp32 inputs.
	"""
	if (
		x.dtype != torch.float32
		or not x.is_cuda
		or torch.is_grad_enabled()
		or linear.weight.dtype != torch.float32
		or torch.cuda.get_device_capability(x.device)[0] < 8
	):
		return linear(x)
	w_hi, w_lo = _split_weight(linear)
	x_hi, x_lo = _split_hi_lo(x)
	prev = torch.backends.cuda.matmul.allow_tf32
	torch.backends.cuda.matmul.allow_tf32 = True
	try:
		out = F.linear(x_hi, w_hi, linear.bias)
		out = out + F.linear(x_hi, w_lo)
		out = out + F.linear(x_lo, w_hi)
	finally:
		torch.backends.cuda.matmul.allow_tf32 = prev
	return out


class FIRETimeBias(nn.Module):
    def __init__(self, max_dist: float, fire_hidden_size: int = 32) -> None:
        super().__init__()
        self.c = nn.Parameter(torch.tensor(100, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(1, fire_hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(fire_hidden_size, 1, bias=False),
        )
        self.max_dist = max_dist

    def forward(
        self, rel_pos: Float[Tensor, "batch target source"]
    ) -> Float[Tensor, "batch 1 1 target source"]:
        rel_pos = rel_pos[:, None, None, :, :, None].to(
            torch.float32
        )  # (B, 1 (L), 1 (A), T, C, 1)
        c = self.c.clamp(min=0)
        rel_pos = torch.log(c * rel_pos + 1) / torch.log(c * self.max_dist + 1)
        return self.mlp(rel_pos).squeeze(-1)


class GPNStarEmbedding(nn.Module):
    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.input_embed = nn.Embedding(self.vocab_size, self.hidden_size)

    def forward(
        self, input_ids: Int[Tensor, "batch position target"] | None = None
    ) -> Float[Tensor, "batch position target hidden"]:
        if input_ids is None:
            raise ValueError("input_ids are required")
        target_embeddings = self.input_embed(input_ids.to(torch.int))
        return target_embeddings  # (B, L, T, H)


class GPNStarAttentionPool(nn.Module):
    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__()

        self.num_attention_heads = config.num_attention_heads // 2
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.attention_weights = nn.Embedding(
            config.vocab_size, self.num_attention_heads
        )
        self.value = nn.Embedding(config.vocab_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.ffn = nn.Linear(self.all_head_size, config.hidden_size)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def pool(
        self,
        source_ids: Int[Tensor, "batch position species"],
        in_clade_time_bias: Float[Tensor, "batch 1 head 1 species"],
    ) -> Float[Tensor, "batch position half_hidden"]:
        attention_scores = (
            self.attention_weights(source_ids).transpose(-1, -2).unsqueeze(-2)
        )  # (B, L, A, 1, N_c)
        value_layer = self.transpose_for_scores(
            self.value(source_ids)
        )  # (B, L, A, N_c, D)

        attention_scores = attention_scores + in_clade_time_bias
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        pooled_layer = torch.matmul(attention_probs, value_layer)  # (B, L, A, 1, D)
        pooled_layer = pooled_layer.transpose(-2, -3).contiguous()  # (B, L, 1, A, D)
        new_pooled_layer_shape = pooled_layer.size()[:-3] + (
            self.all_head_size,
        )  # (B, L, A*D) A*D = H // 2
        pooled_layer = pooled_layer.view(*new_pooled_layer_shape)
        return pooled_layer

    def forward(
        self,
        source_ids: Int[Tensor, "batch position species"],
        in_clade_time_bias: Float[Tensor, "batch 1 head 1 species"],
    ) -> Float[Tensor, "batch position hidden"]:
        return self.ffn(self.pool(source_ids, in_clade_time_bias))


class GPNStarSourceModule(nn.Module):
    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__()

        self.attn_pool = GPNStarAttentionPool(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.time_enc, self.time_scale = _parse_time_encoding(config.time_enc)

        # time embedding modules. Move to higher level?
        if self.time_enc == "fire":
            if config.max_evol_dist is None:
                raise ValueError("max_evol_dist must be set before model construction")
            self.embed_positions = FIRETimeBias(config.max_evol_dist)
        else:
            raise ValueError("time encoding not implemented.")

    def forward(
        self,
        source_ids: Int[Tensor, "batch position species"],
        clade_dict: Mapping[int, set[int]],
        in_clade_phylo_dist: Float[Tensor, "... species"],
    ) -> Float[Tensor, "batch position clade hidden"]:
        in_clade_time_bias = self.embed_positions(
            in_clade_phylo_dist[None, None, :] * self.time_scale
        )

        # Attention pooling: per-species token -> per-clade token
        clade_embeddings: list[Tensor] = []
        for species in clade_dict.values():
            species_indices = list(species)
            if len(species_indices) > 1:
                clade_embeddings.append(
                    self.attn_pool(
                        source_ids[..., species_indices].to(torch.int),  # (B, L, N_c)
                        in_clade_time_bias[..., species_indices],
                    )
                )  # (B, L, A, 1, N_c)
            else:
                clade_embeddings.append(
                    self.embed(source_ids[..., species_indices[0]].to(torch.int))
                )
        clade_pooled_source_ids = torch.stack(clade_embeddings, dim=-2)

        return clade_pooled_source_ids

    def pooled_pre(
        self,
        source_ids: Int[Tensor, "batch position species"],
        clade_dict: Mapping[int, set[int]],
        in_clade_phylo_dist: Float[Tensor, "... species"],
    ) -> Tensor:
        """Pooled clades before attn_pool.ffn, stacked: (B, L, n_pooled, H // 2)."""
        in_clade_time_bias = self.embed_positions(
            in_clade_phylo_dist[None, None, :] * self.time_scale
        )
        out = []
        for species in clade_dict.values():
            if len(species) > 1:
                species_indices = list(species)
                out.append(
                    self.attn_pool.pool(
                        source_ids[..., species_indices].to(torch.int),
                        in_clade_time_bias[..., species_indices],
                    )
                )
        if not out:
            # every clade is a singleton: no pooled clades, zero columns
            return self.attn_pool.value.weight.new_empty(
                source_ids.shape[:2] + (0, self.attn_pool.all_head_size)
            )
        return torch.stack(out, dim=-2)

    def split_info(
        self,
        source_ids: Int[Tensor, "batch position species"],
        clade_dict: Mapping[int, set[int]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Singleton clades embed to embed[token], one of vocab_size rows.

        Returns (multi_idx, single_idx, single_ids, embed_weight): the clade
        positions that are pooled, the clade positions that are singletons,
        the token ids of the singletons (B, L, n_single) and the table.
        """
        device = source_ids.device
        key = (device.type, device.index)
        cache = self.__dict__.setdefault("_split_cache", {})
        sizes = [len(v) for v in clade_dict.values()]
        hit = cache.get(key)
        if hit is None or hit[0] != sizes:
            multi = [i for i, n in enumerate(sizes) if n > 1]
            single = [i for i, n in enumerate(sizes) if n == 1]
            species = [list(v)[0] for v in clade_dict.values() if len(v) == 1]
            hit = (
                sizes,
                torch.tensor(multi, dtype=torch.long, device=device),
                torch.tensor(single, dtype=torch.long, device=device),
                torch.tensor(species, dtype=torch.long, device=device),
            )
            cache[key] = hit
        _, multi_idx, single_idx, species_idx = hit
        single_ids = source_ids.index_select(-1, species_idx).to(torch.long)
        return multi_idx, single_idx, single_ids, self.embed.weight

    def deferred_split(
        self,
        source_ids: Int[Tensor, "batch position species"],
        clade_dict: Mapping[int, set[int]],
        in_clade_phylo_dist: Float[Tensor, "... species"],
    ) -> tuple[Any, ...]:
        """split_info plus the pre-ffn pooled clades and the ffn module."""
        split = self.split_info(source_ids, clade_dict)
        pre = self.pooled_pre(source_ids, clade_dict, in_clade_phylo_dist)
        return split + (pre, self.attn_pool.ffn)

    @staticmethod
    def materialize(split: tuple[Any, ...], positions: Tensor) -> Tensor:
        """Full source embeddings (B, 1, C, H) at one position per batch row."""
        multi_idx, single_idx, single_ids, embed_weight, pre, ffn = split
        pos = positions.view(-1, 1, 1, 1)
        pooled = ffn(pre.gather(1, pos.expand(-1, 1, pre.shape[-2], pre.shape[-1])))
        sids = single_ids.gather(1, pos[..., 0].expand(-1, 1, single_ids.shape[-1]))
        singles = F.embedding(sids, embed_weight)
        out = pooled.new_empty(
            pooled.shape[:2] + (multi_idx.numel() + single_idx.numel(), pooled.shape[-1])
        )
        out.index_copy_(2, multi_idx, pooled)
        out.index_copy_(2, single_idx, singles.to(out.dtype))
        return out


class GPNStarColCrossAttention(nn.Module):
    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads // 2
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Time encoding
        self.time_enc, self.time_scale = _parse_time_encoding(config.time_enc)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def _composed(self, linear: nn.Linear, ffn: nn.Linear) -> tuple[Tensor, Tensor]:
        """Weight and bias of linear(ffn(x)); cached while no weight changes."""
        if torch.is_grad_enabled():
            w = linear.weight @ ffn.weight
            return w, linear(ffn.bias[None]).squeeze(0)
        cache = self.__dict__.setdefault("_composed_cache", {})
        tensors = (linear.weight, linear.bias, ffn.weight, ffn.bias)
        stamp = tuple((id(t), t._version, t.device, t.dtype) for t in tensors)
        hit = cache.get(id(linear))
        if hit is None or hit[0] != stamp:
            with torch.no_grad():
                w = linear.weight @ ffn.weight
                b = linear(ffn.bias[None]).squeeze(0)
            hit = (stamp, (w, b))
            cache[id(linear)] = hit
        return hit[1]

    def _project(
        self,
        linear: nn.Linear,
        source_embeddings: Tensor | None,
        source_split: tuple[Any, ...] | None,
    ) -> Tensor:
        """linear(source_embeddings), running the GEMM on pooled clades only.

        Singleton-clade rows are one of vocab_size vectors, so their output is
        a gather from linear(embedding table).
        """
        if source_split is None:
            return linear(source_embeddings)
        multi_idx, single_idx, single_ids, embed_weight = source_split[:4]
        if len(source_split) > 4:
            pre, ffn = source_split[4:]
            pooled = F.linear(pre, *self._composed(linear, ffn))
        else:
            pooled = linear(source_embeddings.index_select(2, multi_idx))
        singles = F.embedding(single_ids, linear(embed_weight))
        out = pooled.new_empty(
            single_ids.shape[:2] + (multi_idx.numel() + single_idx.numel(), pooled.shape[-1])
        )
        out.index_copy_(2, multi_idx, pooled)
        out.index_copy_(2, single_idx, singles.to(out.dtype))
        return out

    def _forward_split(
        self,
        query_layer: Tensor,
        attention_mask: Tensor,
        evol_time_bias: Tensor,
        source_split: tuple[Any, ...],
    ) -> tuple[Tensor, ...]:
        """Eval attention without K and V rows for the singleton clades.

        Singleton keys and values are rows of a vocab_size table, so their
        scores are q.K_tab gathered by token and their context is the softmax
        mass per token times V_tab.
        """
        multi_idx, single_idx, single_ids, embed_weight, pre, ffn = source_split
        heads, size = self.num_attention_heads, self.attention_head_size
        reassociate = not torch.is_grad_enabled()
        if reassociate:
            wk, bk = self._composed(self.key, ffn)
            wv, bv = self._composed(self.value, ffn)
            n_multi = pre.shape[2]
            wk = wk.view(heads, size, -1)
            wv = wv.view(heads, size, -1)
            bk = bk.view(heads, size)
            bv = bv.view(heads, size)
        else:
            pooled_k = F.linear(pre, *self._composed(self.key, ffn))
            pooled_v = F.linear(pre, *self._composed(self.value, ffn))
            batch, length, n_multi = pooled_k.shape[:3]
            pooled_k = pooled_k.view(batch, length, n_multi, heads, size)
            pooled_v = pooled_v.view(batch, length, n_multi, heads, size)
        k_tab = self.key(embed_weight).view(-1, heads, size)
        v_tab = self.value(embed_weight).view(-1, heads, size)
        vocab = k_tab.shape[0]

        if reassociate:
            q_pre = torch.einsum("blatd,adk->blatk", query_layer, wk)
            score_multi = torch.einsum("blatk,blck->blatc", q_pre, pre)
            score_multi = score_multi + torch.einsum("blatd,ad->blat", query_layer, bk)[..., None]
        else:
            score_multi = torch.einsum("blatd,blcad->blatc", query_layer, pooled_k)
        score_tab = torch.einsum("blatd,vad->blatv", query_layer, k_tab)
        index = single_ids[:, :, None, None, :].expand(-1, -1, heads, score_tab.shape[3], -1)
        score_single = score_tab.gather(-1, index)
        scores = score_multi.new_empty(score_multi.shape[:-1] + (n_multi + single_idx.numel(),))
        scores.index_copy_(-1, multi_idx, score_multi)
        scores.index_copy_(-1, single_idx, score_single)
        attention_mask = attention_mask + evol_time_bias.to(query_layer.dtype)
        scores = scores * (1 / math.sqrt(size)) + attention_mask
        probs = torch.softmax(scores, dim=-1)

        probs_multi = probs.index_select(-1, multi_idx)
        if reassociate:
            pooled = torch.einsum("blatc,blck->blatk", probs_multi, pre)
            context = torch.einsum("blatk,adk->blatd", pooled, wv)
            context = context + probs_multi.sum(-1, keepdim=True) * bv[:, None, :]
        else:
            context = torch.einsum("blatc,blcad->blatd", probs_multi, pooled_v)
        onehot = F.one_hot(single_ids, vocab).to(probs.dtype)
        mass = torch.einsum("blatc,blcv->blatv", probs.index_select(-1, single_idx), onehot)
        context = context + torch.einsum("blatv,vad->blatd", mass, v_tab)
        context = context.transpose(-2, -3).contiguous()
        return (context.view(context.size()[:-2] + (self.all_head_size,)),)

    def forward(
        self,
        hidden_states: Float[Tensor, "batch position target hidden"],
        source_embeddings: Float[Tensor, "batch position clade hidden"],
        attention_mask: Float[Tensor, "batch 1 1 target clade"] | None = None,
        evol_time_bias: Float[Tensor, "batch 1 head target clade"] | None = None,
        output_attentions: bool = False,
        source_split: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, ...]:
        if attention_mask is None or evol_time_bias is None:
            raise ValueError("attention_mask and evol_time_bias are required")
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        if (
            source_split is not None
            and len(source_split) > 4
            and not output_attentions
            and not self.training
        ):
            return self._forward_split(query_layer, attention_mask, evol_time_bias, source_split)
        key_proj = self._project(self.key, source_embeddings, source_split)
        value_proj = self._project(self.value, source_embeddings, source_split)
        fused = not output_attentions and not self.training
        if fused:
            # (B, L, C, A, D): contracted in place, no transpose copy
            kv_shape = key_proj.shape[:-1] + (
                self.num_attention_heads,
                self.attention_head_size,
            )
            key_layer = key_proj.view(kv_shape)
            value_layer = value_proj.view(kv_shape)
        else:
            key_layer = self.transpose_for_scores(key_proj)
            value_layer = self.transpose_for_scores(value_proj)

        attention_mask = attention_mask + evol_time_bias.to(query_layer.dtype)
        # shape (B, L, A, T, D) x (B, L, A, D, C) -> (B, L, A, T, C)

        if output_attentions:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            context_layer = torch.matmul(attention_probs, value_layer)

            context_layer = context_layer.transpose(-2, -3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            outputs: tuple[Tensor, ...] = (
                (context_layer, attention_probs)
                if output_attentions
                else (context_layer,)
            )

        elif fused:
            scores = torch.einsum("blatd,blcad->blatc", query_layer, key_layer)
            scores = scores * (1 / math.sqrt(self.attention_head_size)) + attention_mask
            probs = torch.softmax(scores, dim=-1)
            context_layer = torch.einsum("blatc,blcad->blatd", probs, value_layer)
            context_layer = context_layer.transpose(-2, -3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            outputs = (context_layer,)

        else:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                context_layer = F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask if attention_mask is not None else None,
                    dropout_p=self.attention_probs_dropout_prob
                    if self.training
                    else 0.0,
                    scale=1 / math.sqrt(self.attention_head_size),
                )

            context_layer = context_layer.transpose(-2, -3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            outputs = (context_layer,)

        return outputs


class GPNStarColCrossOutput(nn.Module):
    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size // 2, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: Float[Tensor, "batch position target half_hidden"],
        input_tensor: Float[Tensor, "batch position target hidden"] | None = None,
    ) -> Float[Tensor, "batch position target hidden"]:
        if input_tensor is None:
            raise ValueError("input_tensor is required")
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GPNStarColAttention(nn.Module):
    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__()
        self.self = GPNStarColCrossAttention(config)
        self.out = GPNStarColCrossOutput(config)

        self.time_enc, self.time_scale = _parse_time_encoding(config.time_enc)

        if self.time_enc == "fire":
            if config.max_evol_dist is None:
                raise ValueError("max_evol_dist must be set before model construction")
            self.embed_positions = FIRETimeBias(config.max_evol_dist)
        else:
            raise ValueError(f"Time encoding {self.time_enc} not implemented!")

    def forward(
        self,
        hidden_states: Float[Tensor, "batch position target hidden"],
        source_embeddings: Float[Tensor, "batch position clade hidden"],
        attention_mask: Float[Tensor, "batch 1 1 target clade"] | None = None,
        phylo_dist: Float[Tensor, "batch target clade"] | None = None,
        output_attentions: bool = False,
        source_split: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, ...]:
        if phylo_dist is None:
            raise ValueError("phylo_dist is required")
        evol_time_bias = self.embed_positions(phylo_dist)

        self_outputs = self.self(
            hidden_states=hidden_states,
            source_embeddings=source_embeddings,
            attention_mask=attention_mask,
            evol_time_bias=evol_time_bias,
            output_attentions=output_attentions,
            source_split=source_split,
        )
        output: tuple[Tensor, ...] = (self.out(self_outputs[0], hidden_states),)
        if output_attentions:
            output = output + (self_outputs[1],)
        return output


class GPNStarRowSelfAttention(nn.Module):
    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads // 2
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.time_enc, self.time_scale = _parse_time_encoding(config.time_enc)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

        self.rotary_value = config.rotary_value

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def forward(
        self,
        hidden_states: Float[Tensor, "batch target position hidden"],
        attention_mask: Float[Tensor, "batch 1 head position position_key"]
        | None = None,
        sinusoidal_pos: Float[Tensor, "1 1 position head_dimension"] | None = None,
        output_attentions: bool = False,
        query_positions: Tensor | None = None,
        embed_lookup: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, ...]:
        if embed_lookup is not None and query_positions is None:
            # hidden_states are rows of the embedding table: project the table once
            ids, table = embed_lookup  # (B, T, L), (V, H)
            ids = ids.to(torch.int)
            query_layer = self.transpose_for_scores(
                F.embedding(ids[:, :1], self.query(table))
            )
            key_layer = self.transpose_for_scores(
                F.embedding(ids[:, :1], self.key(table))
            )
            value_layer = self.transpose_for_scores(
                F.embedding(ids, self.value(table))
            )
            hidden_states = None
        # attention scores are only calculated wrt the human genome
        elif query_positions is not None:
            # queries only at one position per batch element; keys and values
            # still cover every position
            idx = query_positions.view(-1, 1, 1, 1).expand(
                -1, 1, 1, hidden_states.shape[-1]
            )
            query_input = hidden_states[:, :1, ...].gather(2, idx)
        else:
            query_input = hidden_states[:, :1, ...]
        qkv = None
        if hidden_states is not None and query_positions is None and hidden_states.shape[1] == 1:
            qkv = compensated_linear_cat(
                self, (self.query, self.key, self.value), hidden_states
            )
        if qkv is not None:
            query_layer, key_layer, value_layer = (
                self.transpose_for_scores(t) for t in qkv
            )
        elif hidden_states is not None:
            query_layer = self.transpose_for_scores(self.query(query_input))
            key_layer = self.transpose_for_scores(self.key(hidden_states[:, :1, ...]))
            value_layer = self.transpose_for_scores(compensated_linear(self.value, hidden_states))

        if sinusoidal_pos is not None and query_positions is not None:
            sin_q = sinusoidal_pos[0, 0][query_positions][:, None, None, None, :]
            query_layer = self.apply_rotary_position_embeddings(
                sin_q, query_layer, query_layer
            )[0]
            if self.rotary_value:
                _, key_layer, value_layer = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, key_layer, key_layer, value_layer
                )
            else:
                _, key_layer = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, key_layer, key_layer
                )
        elif sinusoidal_pos is not None:
            if self.rotary_value:
                query_layer, key_layer, value_layer = (
                    self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer, value_layer
                    )
                )
            else:
                query_layer, key_layer = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_layer, key_layer
                )

        if output_attentions:
            attention_scores = torch.matmul(
                query_layer, key_layer.transpose(-1, -2)
            )  # (B, 1, A, L, L)
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in RoFormerModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(
                attention_scores, dim=-1
            )  # (B, 1, A, L, L)
            attention_probs = self.dropout(attention_probs)

            context_layer = torch.matmul(
                attention_probs, value_layer
            )  # (B, 1, A, L, L) x (B, T, A, L, D) -> (B, T, A, L, D)

            context_layer = context_layer.transpose(
                -2, -3
            ).contiguous()  # (B, T, L, A, D)
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(
                *new_context_layer_shape
            )  # (B, T, L, H/2)
            outputs: tuple[Tensor, ...] = (
                (context_layer, attention_probs)
                if output_attentions
                else (context_layer,)
            )

        else:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                context_layer = F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask if attention_mask is not None else None,
                    dropout_p=self.attention_probs_dropout_prob
                    if self.training
                    else 0.0,
                    scale=1 / math.sqrt(self.attention_head_size),
                )

            context_layer = context_layer.transpose(-2, -3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            outputs = (context_layer,)

        return outputs

    @staticmethod
    @overload
    def apply_rotary_position_embeddings(
        sinusoidal_pos: Tensor,
        query_layer: Tensor,
        key_layer: Tensor,
        value_layer: None = None,
    ) -> tuple[Tensor, Tensor]: ...

    @staticmethod
    @overload
    def apply_rotary_position_embeddings(
        sinusoidal_pos: Tensor,
        query_layer: Tensor,
        key_layer: Tensor,
        value_layer: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    @staticmethod
    def apply_rotary_position_embeddings(
        sinusoidal_pos: Tensor,
        query_layer: Tensor,
        key_layer: Tensor,
        value_layer: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack(
            [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
        ).reshape_as(query_layer)
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack(
            [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
        ).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack(
                [-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1
            ).reshape_as(value_layer)
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer


class GPNStarRowSelfOutput(nn.Module):
    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size // 2, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: Float[Tensor, "batch target position half_hidden"],
        input_tensor: Float[Tensor, "batch target position hidden"] | None = None,
    ) -> Float[Tensor, "batch target position hidden"]:
        if input_tensor is None:
            raise ValueError("input_tensor is required")
        hidden_states = compensated_linear(self.dense, hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GPNStarRowAttention(nn.Module):
    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__()
        self.self = GPNStarRowSelfAttention(config)
        self.out = GPNStarRowSelfOutput(config)

    def forward(
        self,
        hidden_states: Float[Tensor, "batch target position hidden"],
        attention_mask: Tensor | None = None,
        sinusoidal_pos: Tensor | None = None,
        output_attentions: bool = False,
        query_positions: Tensor | None = None,
        embed_lookup: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, ...]:
        self_outputs = self.self(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            sinusoidal_pos=sinusoidal_pos,
            output_attentions=output_attentions,
            query_positions=query_positions,
            embed_lookup=embed_lookup,
        )
        residual = hidden_states
        if query_positions is not None:
            idx = query_positions.view(-1, 1, 1, 1).expand(
                -1, hidden_states.shape[1], 1, hidden_states.shape[-1]
            )
            residual = hidden_states.gather(2, idx)
        output: tuple[Tensor, ...] = (self.out(self_outputs[0], residual),)
        if output_attentions:
            output = output + (self_outputs[1],)
        return output


class GPNStarAttention(nn.Module):
    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__()
        self.row_attention = GPNStarRowAttention(config)
        self.col_attention = GPNStarColAttention(config)

    def forward(
        self,
        hidden_states: Float[Tensor, "batch position target hidden"],
        source_embeddings: Float[Tensor, "batch position clade hidden"],
        attention_mask: Tensor | None = None,
        sinusoidal_pos: Tensor | None = None,
        phylo_dist: Float[Tensor, "batch target clade"] | None = None,
        head_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_value: tuple[Tensor, ...] | None = None,
        output_attentions: bool = False,
        query_positions: Tensor | None = None,
        source_split: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
        embed_lookup: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, ...]:
        row_output = self.row_attention(
            hidden_states=hidden_states.transpose(-2, -3),
            attention_mask=None,
            sinusoidal_pos=sinusoidal_pos,
            output_attentions=output_attentions,
            query_positions=query_positions,
            embed_lookup=embed_lookup,
        )
        if query_positions is not None:
            if source_embeddings is None:
                source_embeddings = GPNStarSourceModule.materialize(
                    source_split, query_positions
                )
            else:
                idx = query_positions.view(-1, 1, 1, 1).expand(
                    -1, 1, source_embeddings.shape[-2], source_embeddings.shape[-1]
                )
                source_embeddings = source_embeddings.gather(1, idx)
            source_split = None
        col_output = self.col_attention(
            hidden_states=row_output[0].transpose(-2, -3),
            source_embeddings=source_embeddings,
            attention_mask=attention_mask,
            phylo_dist=phylo_dist,
            output_attentions=output_attentions,
            source_split=source_split,
        )
        out: tuple[Tensor, ...] = (col_output[0],)
        if output_attentions:
            out = out + (
                row_output[1],
                col_output[1],
            )
        return out


class GPNStarLayer(RoFormerLayer):
    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__(config)
        self.attention = GPNStarAttention(config)

    def feed_forward_chunk(self, attention_output: Tensor) -> Tensor:
        inter = self.intermediate
        h = inter.intermediate_act_fn(compensated_linear(inter.dense, attention_output))
        out = self.output
        h = out.dropout(compensated_linear(out.dense, h))
        return out.LayerNorm(h + attention_output)

    def forward(
        self,
        hidden_states: Tensor,
        source_embeddings: Tensor,
        attention_mask: Tensor | None = None,
        sinusoidal_pos: Tensor | None = None,
        phylo_dist: Tensor | None = None,
        head_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_value: tuple[Tensor, ...] | None = None,
        output_attentions: bool = False,
        query_positions: Tensor | None = None,
        source_split: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
        embed_lookup: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Any, ...]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            source_embeddings,
            attention_mask,
            sinusoidal_pos,
            phylo_dist,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            query_positions=query_positions,
            source_split=source_split,
            embed_lookup=embed_lookup,
        )
        attention_output = self_attention_outputs[0]

        present_key_value: Any = None

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention "
                    "layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                sinusoidal_pos,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs


class GPNStarEncoder(RoFormerEncoder):
    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__(config)
        self.layer = nn.ModuleList(
            [GPNStarLayer(config) for _ in range(config.num_hidden_layers)]
        )  #

    def forward(
        self,
        hidden_states: Tensor,
        source_embeddings: Tensor,
        attention_mask: Tensor | None = None,
        head_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: tuple[tuple[Tensor, ...], ...] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        phylo_dist: Tensor | None = None,
        query_positions: Tensor | None = None,
        source_split: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
        embed_lookup: tuple[Tensor, Tensor] | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithRowAndColAttentions | tuple[Any, ...]:
        if output_attentions:
            query_positions = None
        if self.gradient_checkpointing and self.training:
            use_cache = False
        all_hidden_states: tuple[Tensor, ...] | None = (
            () if output_hidden_states else None
        )
        all_row_attentions: tuple[Tensor, ...] | None = (
            () if output_attentions else None
        )
        all_col_attentions: tuple[Tensor, ...] | None = (
            () if output_attentions else None
        )

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        # [sequence_length, embed_size_per_head] -> [batch_size, num_heads, sequence_length, embed_size_per_head]
        sinusoidal_pos = self.embed_positions(
            hidden_states.shape[:-1], past_key_values_length
        )[None, None, :, :]

        next_decoder_cache: tuple[Any, ...] | None = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if query_positions is not None and i == len(self.layer) - 1:
                # last layer: only the requested position is computed
                layer_outputs = layer_module(
                    hidden_states,
                    source_embeddings,
                    attention_mask,
                    sinusoidal_pos,
                    phylo_dist,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_positions=query_positions,
                    source_split=source_split,
                )
            elif self.gradient_checkpointing and self.training:

                def create_custom_forward(module: nn.Module) -> Callable[..., Any]:
                    def custom_forward(*inputs: Any) -> Any:
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    source_embeddings,
                    attention_mask,
                    sinusoidal_pos,
                    phylo_dist,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    source_embeddings,
                    attention_mask,
                    sinusoidal_pos,
                    phylo_dist,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    source_split=source_split,
                    embed_lookup=embed_lookup if i == 0 else None,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                assert next_decoder_cache is not None
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                assert all_row_attentions is not None
                assert all_col_attentions is not None
                all_row_attentions = all_row_attentions + (layer_outputs[1],)
                all_col_attentions = all_col_attentions + (layer_outputs[2],)

        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_row_attentions,
                    all_col_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithRowAndColAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            row_attentions=all_row_attentions,
            col_attentions=all_col_attentions,
        )


@dataclass
class BaseModelOutputWithRowAndColAttentions(ModelOutput):
    last_hidden_state: Tensor | None = None
    past_key_values: tuple[Any, ...] | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    row_attentions: tuple[Tensor, ...] | None = None
    col_attentions: tuple[Tensor, ...] | None = None


class GPNStarPreTrainedModel(PreTrainedModel):
    config_class = GPNStarConfig
    base_model_prefix = "model"

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[str],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Save a portable checkpoint with its required phylogenetic assets."""
        source_dir = self.config.phylo_dist_path
        if not _contains_phylo_dist_files(source_dir):
            raise FileNotFoundError(
                "Cannot save GPN-Star without pairwise.npy and in_clade.npy in "
                f"the configured phylo_dist_path: '{source_dir}'"
            )

        asset_dir = os.path.join(os.fspath(save_directory), "phylo_dist")
        os.makedirs(asset_dir, exist_ok=True)
        for filename in _PHYLO_DIST_FILENAMES:
            source = os.path.join(source_dir, filename)
            destination = os.path.join(asset_dir, filename)
            if os.path.realpath(source) != os.path.realpath(destination):
                shutil.copy2(source, destination)

        original_path = self.config.phylo_dist_path
        self.config.phylo_dist_path = None
        try:
            return super().save_pretrained(save_directory, *args, **kwargs)
        finally:
            self.config.phylo_dist_path = original_path

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args: Any,
        config: GPNStarConfig | None = None,
        **kwargs: Any,
    ) -> Self:
        """Load a checkpoint and its bundled phylogenetic-distance assets."""
        explicit_path = kwargs.pop("phylo_dist_path", None)
        if config is None:
            hub_kwargs: dict[str, Any] = {
                key: kwargs[key]
                for key in (
                    "cache_dir",
                    "force_download",
                    "local_files_only",
                    "proxies",
                    "revision",
                    "subfolder",
                    "token",
                )
                if key in kwargs
            }
            config_kwargs: dict[str, Any] = kwargs.copy()
            for key in ("torch_dtype", "dtype"):
                if config_kwargs.get(key) == "auto":
                    config_kwargs.pop(key)
            if config_kwargs.get("quantization_config") is not None:
                config_kwargs.pop("quantization_config")
            config, unused_kwargs = cls.config_class.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                **config_kwargs,
            )
            for key in ("torch_dtype", "dtype", "quantization_config"):
                if kwargs.get(key) is not None:
                    unused_kwargs[key] = kwargs[key]
            kwargs = {**hub_kwargs, **dict(unused_kwargs)}
            commit_hash = getattr(config, "_commit_hash", None)
            if commit_hash is not None:
                kwargs["_commit_hash"] = commit_hash

        config.phylo_dist_path = _resolve_phylo_dist_path(
            pretrained_model_name_or_path,
            config,
            explicit_path=explicit_path,
            hub_kwargs=kwargs,
        )
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            **kwargs,
        )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RoFormerSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(
        self, module: nn.Module, value: bool = False
    ) -> None:
        if isinstance(module, RoFormerEncoder):
            module.gradient_checkpointing = value


class GPNStarPhyloInfo:
    def __init__(self, config: GPNStarConfig) -> None:
        # Four variables extracted from the phylogenetic tree
        # Pairwise distance (N, N)
        # Distance to MRCA of the clade (N,)
        # Clades dict {clade: set(species)}
        # Clade labels (N,)

        # Read
        if config.phylo_dist_path is None:
            raise ValueError("phylo_dist_path is required")
        phylo_dist_path = config.phylo_dist_path
        self.phylo_dist_pairwise = torch.tensor(
            np.load(phylo_dist_path + "/pairwise.npy"),
            device="cpu",
        )
        self.in_clade_phylo_dist = torch.tensor(
            np.load(phylo_dist_path + "/in_clade.npy"),
            device="cpu",
        )

        self.clade_dict = self.cluster_clades(
            self.phylo_dist_pairwise, config.clade_thres
        )

        self.clade_labels = torch.zeros(
            self.phylo_dist_pairwise.shape[0],
            dtype=torch.int64,
            device="cpu",
        )
        for clade_id, species in self.clade_dict.items():
            for s in species:
                self.clade_labels[s] = clade_id

        self.max_evol_dist = self.phylo_dist_pairwise.max().item()

    @staticmethod
    def cluster_clades(
        phylo_dist_pairwise: Float[Tensor, "species other_species"],
        threshold: float,
    ) -> dict[int, set[int]]:
        N = phylo_dist_pairwise.shape[0]
        G: nx.Graph[int] = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i + 1, N):
                if phylo_dist_pairwise[i, j] <= threshold:
                    G.add_edge(i, j)
        clade_dict = {
            i: nodes for i, nodes in enumerate(list(nx.connected_components(G)))
        }
        return clade_dict


class GPNStarModel(GPNStarPreTrainedModel):
    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__(config)
        self.config = config
        self.phylo_info = GPNStarPhyloInfo(config)
        self.config.max_evol_dist = self.phylo_info.max_evol_dist
        self.target_embedding = GPNStarEmbedding(self.config)
        self.source_embedding = GPNStarSourceModule(self.config)
        self.encoder = GPNStarEncoder(self.config)

        # Initialize weights and apply final processing
        self.post_init()

    def _drop_stale_weight_caches(self) -> None:
        """Drop weight-derived caches when any parameter's values changed.

        The caches key on tensor _version, which param.data edits do not bump;
        a per-tensor L1 fingerprint of all parameters catches those edits.
        """
        params = self.__dict__.get("_fp_params")
        if params is None:
            params = list(self.parameters())
            if params:
                self.__dict__["_fp_params"] = params
        if params:
            with torch.no_grad():
                fp = torch.stack(torch._foreach_norm(params, 1)).cpu()
            old = self.__dict__.get("_fp_value")
            self.__dict__["_fp_value"] = fp
            if old is None or torch.equal(old, fp):
                return
        # A DataParallel replica holds its weights as plain tensor attributes, so it has
        # nothing to fingerprint; it is rebuilt every forward, and the caches it copied
        # from the original module are dropped instead.
        for module in self.modules():
            module.__dict__.pop("_composed_cache", None)
            if isinstance(module, nn.Linear):
                module.__dict__.pop("_split_cache", None)

    def forward(
        self,
        input_ids: Int[Tensor, "batch position target"] | None = None,
        source_ids: Int[Tensor, "batch position species"] | None = None,
        target_species: Int[Tensor, "batch target"] | None = None,
        output_attentions: bool = False,
        query_positions: Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithRowAndColAttentions | tuple[Any, ...]:
        if input_ids is None or source_ids is None or target_species is None:
            raise ValueError("input_ids, source_ids, and target_species are required")
        if not self.training and not torch.is_grad_enabled():
            self._drop_stale_weight_caches()
        hidden_states = self.target_embedding(
            input_ids=input_ids,
        )  # (B, L, T, H)

        clade_dict = self.phylo_info.clade_dict
        in_clade_phylo_dist = self.phylo_info.in_clade_phylo_dist.to(
            hidden_states.device
        )
        phylo_dist_pairwise = self.phylo_info.phylo_dist_pairwise.to(
            hidden_states.device
        )
        clade_labels = self.phylo_info.clade_labels.to(hidden_states.device)

        # Embed source sequences
        source_split = None
        source_embeddings = None
        if not (self.training and getattr(self.encoder, "gradient_checkpointing", False)):
            if self.training or torch.is_grad_enabled() or output_attentions:
                source_split = self.source_embedding.split_info(source_ids, clade_dict)
            else:
                source_split = self.source_embedding.deferred_split(
                    source_ids, clade_dict, in_clade_phylo_dist
                )
        if source_split is None or len(source_split) == 4:
            source_embeddings = self.source_embedding(
                source_ids, clade_dict, in_clade_phylo_dist
            )  # (B, L, C, H)

        # Source-target phylo distance
        phylo_dist = phylo_dist_pairwise[target_species]  # (B, T, N)
        phylo_dist = self.compute_clade_means(
            phylo_dist, clade_labels, len(clade_dict)
        )  # (B, T, C)
        # Each target species do not query the clade it comes from
        target_clades = clade_labels[target_species]  # (B, T)
        attention_mask = F.one_hot(target_clades, num_classes=phylo_dist.size(-1)).to(
            hidden_states.dtype
        )  # (B, T, C)
        attention_mask = (
            attention_mask * torch.finfo(hidden_states.dtype).min
        )  # (B, T, C)
        attention_mask = attention_mask[:, None, None, :, :]

        embed_lookup = None
        embed = self.target_embedding.input_embed
        if (
            not self.training
            and not torch.is_grad_enabled()
            and not output_attentions
            and input_ids.shape[-1] == hidden_states.shape[-2]
        ):
            # layer 0's row Q/K/V read rows of the embedding table
            embed_lookup = (input_ids.transpose(-1, -2), embed.weight)

        x = self.encoder(
            hidden_states,
            source_embeddings,
            phylo_dist=phylo_dist,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            query_positions=query_positions,
            source_split=source_split,
            embed_lookup=embed_lookup,
            **kwargs,
        )
        return x

    @staticmethod
    def compute_clade_means(
        A: Float[Tensor, "batch target species"],
        indices: Int[Tensor, "... species"],
        C: int,
    ) -> Float[Tensor, "batch target clade"]:
        B, T, N = A.shape
        A_flat = A.reshape(-1, N)  # Shape: (B*T, N)
        indices_expanded = indices.unsqueeze(0).expand(
            A_flat.shape[0], N
        )  # Shape: (B*T, N)
        sums = torch.zeros(A_flat.shape[0], C, device=A.device, dtype=A.dtype)
        counts = torch.zeros(A_flat.shape[0], C, device=A.device, dtype=A.dtype)
        sums.scatter_add_(dim=1, index=indices_expanded, src=A_flat)
        ones = torch.ones_like(A_flat)
        counts.scatter_add_(dim=1, index=indices_expanded, src=ones)
        counts = counts.clamp(min=1)
        means_flat = sums / counts
        means = means_flat.view(B, T, C)
        return means


def compute_loss(
    logits: Float[Tensor, "... nucleotide"],
    labels: Int[Tensor, "..."] | None,
    output_probs: Float[Tensor, "... nucleotide"] | None,
    loss_weight: Float[Tensor, "..."] | None,
    vocab_size: int,
) -> Float[Tensor, ""] | None:
    loss = None
    if labels is not None and loss_weight is None:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))

    elif labels is not None and loss_weight is not None:
        loss_fct = CrossEntropyLoss(reduction="none")
        labels = labels.view(-1)
        exclude = labels == -100
        labels = labels[~exclude]
        logits = logits.view(-1, vocab_size)[~exclude]
        loss_weight = loss_weight.view(-1)[~exclude]
        loss = loss_fct(logits, labels)
        loss = (loss * loss_weight).sum() / loss_weight.sum()

    elif output_probs is not None:
        if loss_weight is None:
            raise ValueError("loss_weight is required with soft output probabilities")
        loss_fct = CrossEntropyLoss(reduction="none")
        output_probs = output_probs.view(-1, vocab_size)
        exclude = (output_probs == 0.0).all(dim=-1)
        output_probs = output_probs[~exclude]
        logits = logits.view(-1, vocab_size)[~exclude]
        loss_weight = loss_weight.view(-1)[~exclude]
        loss = loss_fct(logits, output_probs)
        loss = (loss * loss_weight / loss_weight.sum()).sum()
    return loss


class GPNStarForMaskedLM(GPNStarPreTrainedModel):
    # The decoder bias aliases ``cls.predictions.bias``. The decoder weight is
    # deliberately independent of the target embedding (and has a different
    # width in the published model), so it must not be tied here.
    _tied_weights_keys = {
        "cls.predictions.decoder.bias": "cls.predictions.bias",
    }

    def __init__(self, config: GPNStarConfig) -> None:
        super().__init__(config)

        self.model = GPNStarModel(config)
        self.cls = RoFormerOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        labels: Int[Tensor, "..."] | None = None,
        output_probs: Float[Tensor, "... nucleotide"] | None = None,
        loss_weight: Float[Tensor, "..."] | None = None,
        **kwargs: Any,
    ) -> MaskedLMOutput:
        hidden_state = self.model(**kwargs).last_hidden_state
        logits = self.cls(hidden_state)
        loss = compute_loss(
            logits, labels, output_probs, loss_weight, self.config.vocab_size
        )
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )
