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

    def forward(
        self,
        source_ids: Int[Tensor, "batch position species"],
        in_clade_time_bias: Float[Tensor, "batch 1 head 1 species"],
    ) -> Float[Tensor, "batch position hidden"]:
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
        pooled_layer = self.ffn(pooled_layer)

        return pooled_layer


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

    def forward(
        self,
        hidden_states: Float[Tensor, "batch position target hidden"],
        source_embeddings: Float[Tensor, "batch position clade hidden"],
        attention_mask: Float[Tensor, "batch 1 1 target clade"] | None = None,
        evol_time_bias: Float[Tensor, "batch 1 head target clade"] | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, ...]:
        if attention_mask is None or evol_time_bias is None:
            raise ValueError("attention_mask and evol_time_bias are required")
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(source_embeddings))
        value_layer = self.transpose_for_scores(self.value(source_embeddings))

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
    ) -> tuple[Tensor, ...]:
        # attention scores are only calculated wrt the human genome
        query_layer = self.transpose_for_scores(self.query(hidden_states[:, :1, ...]))
        key_layer = self.transpose_for_scores(self.key(hidden_states[:, :1, ...]))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if sinusoidal_pos is not None:
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
        hidden_states = self.dense(hidden_states)
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
    ) -> tuple[Tensor, ...]:
        self_outputs = self.self(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            sinusoidal_pos=sinusoidal_pos,
            output_attentions=output_attentions,
        )
        output: tuple[Tensor, ...] = (self.out(self_outputs[0], hidden_states),)
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
    ) -> tuple[Tensor, ...]:
        row_output = self.row_attention(
            hidden_states=hidden_states.transpose(-2, -3),
            attention_mask=None,
            sinusoidal_pos=sinusoidal_pos,
            output_attentions=output_attentions,
        )
        col_output = self.col_attention(
            hidden_states=row_output[0].transpose(-2, -3),
            source_embeddings=source_embeddings,
            attention_mask=attention_mask,
            phylo_dist=phylo_dist,
            output_attentions=output_attentions,
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
        **kwargs: Any,
    ) -> BaseModelOutputWithRowAndColAttentions | tuple[Any, ...]:
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

            if self.gradient_checkpointing and self.training:

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

    def forward(
        self,
        input_ids: Int[Tensor, "batch position target"] | None = None,
        source_ids: Int[Tensor, "batch position species"] | None = None,
        target_species: Int[Tensor, "batch target"] | None = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutputWithRowAndColAttentions | tuple[Any, ...]:
        if input_ids is None or source_ids is None or target_species is None:
            raise ValueError("input_ids, source_ids, and target_species are required")
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

        x = self.encoder(
            hidden_states,
            source_embeddings,
            phylo_dist=phylo_dist,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
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
