# Copyright 2026 The MiniMax AI Team, HuggingFace Team, and the VeOmni Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Patch configuration for MiniMax M3 VL transformers>=5.16.0 code generation.

The VeOmni integration keeps the upstream MiniMax M3 VL modeling body from
transformers, then adds VeOmni hooks for parallel plans, VLM collator metadata,
and FSDP-symmetric dummy vision execution. The public MiniMaxAI/MiniMax-M3
checkpoint uses an older language/MoE key layout, so runtime checkpoint
conversion is registered separately in checkpoint_tensor_converter.py. The
public index maps the spatial merge projector through `patch_merge_mlp` into
the generated `multi_modal_projector.merge_linear_{1,2}` parameters; full
public checkpoint loading still requires running the 59-shard load gate.

Regen command:
patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_gpu_patch_gen_config -o veomni/models/transformers/minimax_m3_vl/generated --diff
"""

from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import initialization as init
from transformers.modeling_utils import PreTrainedModel
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
    MiniMaxM3VLConfig,
    MiniMaxM3VLTextConfig,
)
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLAttention,
    MiniMaxM3VLCausalLMOutputWithPast,
    MiniMaxM3VLDecoderLayer,
    MiniMaxM3VLExperts,
    MiniMaxM3VLRMSNorm,
    MiniMaxM3VLTopKRouter,
)

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad_and_slice
from veomni.models.loss_utils import ForCausalLMLoss
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.device import IS_NPU_AVAILABLE
from veomni.utils.model_outputs import FusedLinearAuxOutputMixin


config = PatchConfig(
    source_module="transformers.models.minimax_m3_vl.modeling_minimax_m3_vl",
    target_file="patched_modeling_minimax_m3_vl_gpu.py",
    description="MiniMax M3 VL with VeOmni parallel-plan hooks",
)
config.add_import("torch.distributed", alias="dist", is_from_import=False)
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("functools", names=["partial"])
config.add_import("veomni.ops", names=["VeomniOp"])
config.add_import("veomni.ops.config", names=["resolve_op_impl"])
config.add_import("veomni.models.loss_utils", names=["ForCausalLMLoss"])
config.add_import(
    "veomni.utils.model_outputs",
    names=["CausalLMOutputWithLogProbs", "FusedLinearAuxOutput", "FusedLinearAuxOutputMixin"],
)
config.drop_import_names("MoeCausalLMOutputWithPast")
# Only the multimodal model is supported; the text-only causal LM is not generated.
config.exclude_from_output("load_balancing_loss_func", "MiniMaxM3VLForCausalLM")
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=["gather_outputs", "slice_input_tensor", "sp_pad_and_slice"],
)
config.add_import(
    "veomni.ops.kernels.attention.ulysses",
    names=["gather_seq_scatter_kv_heads", "prepare_ulysses_qkv", "restore_ulysses_output"],
)
config.add_import("veomni.utils.device", names=["IS_NPU_AVAILABLE"])


@config.replace_class(
    "MiniMaxM3VLPreTrainedModel",
    description="Allow VeOmni flash-attention implementations for the vision tower and dense language layers",
)
class PatchedMiniMaxM3VLPreTrainedModel(PreTrainedModel):
    config: MiniMaxM3VLConfig | MiniMaxM3VLTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxM3VLDecoderLayer", "MiniMaxM3VLVisionEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": MiniMaxM3VLDecoderLayer,
        "attentions": MiniMaxM3VLAttention,
    }
    input_modalities = ("image", "video", "text")
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
    _compatible_flash_implementations = [
        "MiniMaxAI/msa",
        "flash_attention_2",
        "flash_attention_3",
        "flash_attention_4",
        "veomni_flash_attention_2",
        "veomni_flash_attention_2_hub",
        "veomni_flash_attention_3",
        "veomni_flash_attention_3_hub",
        "veomni_flash_attention_4",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, MiniMaxM3VLExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, MiniMaxM3VLTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, MiniMaxM3VLRMSNorm):
            init.zeros_(module.weight)


@config.override_method(
    "MiniMaxM3VLRMSNorm.__init__",
    description="Construct a local rms_norm offset VeomniOp",
)
def minimax_m3_vl_rmsnorm_init_patched(self, dim: int, eps: float = 1e-6):
    nn.Module.__init__(self)
    self.eps = eps
    self.weight = nn.Parameter(torch.zeros(dim))
    self.veomni_rms_norm = VeomniOp("rms_norm", "offset", resolve_op_impl("rms_norm_implementation"))


@config.override_method(
    "MiniMaxM3VLRMSNorm.forward",
    description="Always call the local rms_norm offset VeomniOp",
)
def minimax_m3_vl_rmsnorm_forward_patched(self, x):
    # The zero-initialised weight stays near 0, so ``1 + weight`` lands near 1.0
    # where bf16 ticks are 2**-7 apart. Hand every row a fp32 weight so the
    # offset is added in fp32 as in eager; adding it in bf16 would snap most
    # channels onto a few gammas.
    return self.veomni_rms_norm(x, self.weight.float(), eps=self.eps)


@config.add_helper
def collate_multimodal_metadata(batch, sp_pad):
    """Derive MiniMax ViT metadata on CPU inside the VeOmni collator.

    Each ``grid_thw`` entry is one logical visual sample with ``t * h * w``
    tokens. An SP-padding tail, when present, is appended as a separate
    synthetic sequence so neither another sample nor padding can affect the
    real visual tokens through varlen attention.
    """
    md = {}
    for modality, grid_key, pad_key in (
        ("image", "image_grid_thw", "pixel_values"),
        ("video", "video_grid_thw", "pixel_values_videos"),
    ):
        grid = batch.get(grid_key)
        if grid is None:
            continue
        grid_list = grid.tolist() if torch.is_tensor(grid) else grid
        if not grid_list:
            continue

        md[f"{modality}_grid_thw_list"] = grid_list
        cu_seqlens = [0]
        max_seqlen = 0
        for t, h, w in grid_list:
            seq_len = t * h * w
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
            max_seqlen = max(max_seqlen, seq_len)

        pad = sp_pad.get(pad_key, 0)
        if pad > 0:
            cu_seqlens.append(cu_seqlens[-1] + pad)
            max_seqlen = max(max_seqlen, pad)

        md[f"vit_{modality}_cu_seqlens"] = torch.tensor(
            cu_seqlens,
            dtype=torch.int32,
            device="cpu",
        )
        md[f"vit_{modality}_max_seqlen"] = max_seqlen
    if md:
        batch["multimodal_metadata"] = md


@config.add_helper
def _grid_thw_to_list(grid_thw, grid_thw_list):
    if grid_thw_list is not None:
        return grid_thw_list
    return grid_thw.tolist()


@config.add_helper
def _drop_sp_padding_features(features, grid_thw, grid_thw_list, merge_size):
    """Keep the gathered features of real merged tokens, dropping the SP-padding tail.

    Under SP the vision tower pads its pixel rows to a multiple of
    ``sp_size * merge_size**2`` before slicing, so the gathered projector output
    ends with features of that synthetic tail. Placeholder masks only cover the
    real ``sum(t * h * w) // merge_size**2`` tokens.
    """
    grid_thw_list = _grid_thw_to_list(grid_thw, grid_thw_list)
    return features[: sum(t * h * w for t, h, w in grid_thw_list) // merge_size**2]


@config.override_method(
    "MiniMaxM3VLIndexer.__init__",
    description="Freeze the sparse-attention indexer, which receives no gradient",
)
def minimax_m3_vl_indexer_init_patched(self, config: MiniMaxM3VLTextConfig, layer_idx: int):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = config.index_head_dim
    self.num_heads = config.index_n_heads
    self.block_size = config.index_block_size
    self.topk_blocks = config.index_topk_blocks
    self.local_blocks = config.index_local_blocks
    self.q_proj = nn.Linear(config.hidden_size, config.index_n_heads * config.index_head_dim, bias=False)
    self.k_proj = nn.Linear(config.hidden_size, config.index_head_dim, bias=False)
    self.q_norm = MiniMaxM3VLRMSNorm(config.index_head_dim, eps=config.rms_norm_eps)
    self.k_norm = MiniMaxM3VLRMSNorm(config.index_head_dim, eps=config.rms_norm_eps)
    # --- Patch.1 ---
    # The indexer is not trained: block selection is a top-k over discrete block
    # scores and no indexer loss is computed, so no gradient can reach these
    # parameters and they keep their checkpoint values. Freezing them states that
    # explicitly, which DDP needs (without ``find_unused_parameters`` it raises on
    # the next iteration) and keeps them out of the optimizer. MiniMax describes the indexer as
    # distilled from full attention with a KL objective, but has not published how
    # it is handled in post-training; add that objective here once it is known.
    self.requires_grad_(False)
    # --- Patch.1 ---


@config.add_helper
def _minimax_m3_indexer_project(indexer, hidden_states, position_embeddings):
    """Return the indexer's normed, rotated ``[B, S, G, D]`` queries and ``[B, S, 1, D]`` keys.

    This is the projection half of ``MiniMaxM3VLIndexer.forward``. On the packed
    training path, block selection runs in the MiniMax sparse attention
    operator after the Ulysses exchange instead.

    The indexer is not trained (see ``MiniMaxM3VLIndexer.__init__``), and its
    input ``hidden_states`` still requires grad, so the projections run under
    ``no_grad`` to skip building an autograd graph that no loss ever reaches.
    """
    batch, q_len, _ = hidden_states.shape
    with torch.no_grad():
        index_query = indexer.q_norm(indexer.q_proj(hidden_states).view(batch, q_len, -1, indexer.head_dim))
        index_key = indexer.k_norm(indexer.k_proj(hidden_states).view(batch, q_len, 1, indexer.head_dim))
        cos, sin = position_embeddings
        return apply_rotary_pos_emb(
            index_query,
            index_key,
            cos[..., : indexer.head_dim],
            sin[..., : indexer.head_dim],
            unsqueeze_dim=2,
        )


@config.add_helper
def _require_varlen_attention(attention_implementation, *, num_segments, ulysses_enabled, attention_name, causal):
    """Reject a backend that would leak future tokens, mix packed sequences or skip the Ulysses exchange.

    MiniMax attention passes no mask, so only a varlen backend keeps packed
    sequences (media inputs or text samples) apart, and only the VeOmni
    ``veomni_*`` adapters run the Ulysses all-to-all; anything else silently
    attends across sequences or within the local shard only. ``causal``
    attention (the dense language layers) needs the varlen backend even for a
    single sequence: eager attention, for one, applies no causal mask itself.
    """
    # This mask-free model requires FlashAttention's varlen contract; other
    # attention implementations need their own mask preparation.
    varlen_backend = attention_implementation.removeprefix("veomni_") in (
        "flash_attention_2",
        "flash_attention_2_hub",
        "flash_attention_3",
        "flash_attention_3_hub",
        "flash_attention_4",
    ) and (not ulysses_enabled or attention_implementation.startswith("veomni_"))
    if not varlen_backend and (causal or num_segments > 1 or ulysses_enabled):
        requirement = "causal masking" if causal else "multiple packed sequences"
        raise ValueError(
            f"MiniMax M3 VL {attention_name} needs a varlen FlashAttention backend (flash_attention_2/3/4) for "
            f"{requirement}, and its VeOmni veomni_* adapter under Ulysses, got {attention_implementation!r}."
        )


@config.add_helper
def _minimax_m3_sparse_attention(
    attention, query_states, key_states, value_states, hidden_states, position_embeddings, *, dropout, **kwargs
):
    """Run one sparse layer's block selection and attention on packed ``[1, H, S, D]`` Q/K/V.

    Under Ulysses, the main Q/K/V use the sequence/head all-to-all, the
    per-KV-group index Q is split across ranks exactly like K, and the
    single-head index K is all-gathered. The MiniMax sparse attention operators
    consume TND tensors; any padded layout they need stays inside them. Returns
    ``[1, S, Hq, D]`` in the local sequence layout.
    """
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    indexer = attention.indexer
    index_query, index_key = _minimax_m3_indexer_project(indexer, hidden_states, position_embeddings)

    parallel_state = get_parallel_state()
    if parallel_state.ulysses_enabled:
        query_states, key_states, value_states, _ = prepare_ulysses_qkv(
            query_states,
            key_states,
            value_states,
            group=parallel_state.ulysses_group,
            ulysses_size=parallel_state.ulysses_size,
        )
        index_query = gather_seq_scatter_kv_heads(
            index_query,
            group=parallel_state.ulysses_group,
            ulysses_size=parallel_state.ulysses_size,
        )
        index_key = gather_outputs(index_key, gather_dim=1, group=parallel_state.ulysses_group)

    # The text model guarantees packed varlen metadata; training packs samples
    # with identical query/key boundaries.
    cu_seqlens = kwargs["cu_seq_lens_q"]
    max_seqlen = kwargs["max_length_q"]
    block_indices = attention.veomni_msa_indexer(
        index_query[0],
        index_key[0],
        cu_seqlens,
        max_seqlen=max_seqlen,
        block_size=indexer.block_size,
        topk_blocks=indexer.topk_blocks,
        local_blocks=indexer.local_blocks,
    )
    attn_output = attention.veomni_msa_attention(
        query_states[0],
        key_states[0],
        value_states[0],
        block_indices,
        cu_seqlens,
        max_seqlen=max_seqlen,
        block_size=indexer.block_size,
        scale=attention.scaling,
        dropout_p=dropout,
    ).unsqueeze(0)

    if parallel_state.ulysses_enabled:
        attn_output = restore_ulysses_output(attn_output, group=parallel_state.ulysses_group)
    return attn_output


@config.modify_init(
    "MiniMaxM3VLAttention",
    description="Bind instance-local attention VeomniOps: standard attention for dense layers, MSA for sparse ones",
)
def minimax_m3_vl_attention_bind_ops(original_init, self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if self.indexer is None:
        self.veomni_attn = VeomniOp("attention", "standard", self.config._attn_implementation)
    else:
        self.veomni_msa_indexer = VeomniOp("minimax_sparse_attention", "indexer")
        self.veomni_msa_attention = VeomniOp("minimax_sparse_attention", "attention")


# ================================================================
# Patch: MiniMaxM3VLAttention.forward
# 1. dispatch on the layer type after RoPE: dense layers call the standard
#    attention VeomniOp (a varlen FlashAttention backend, whose ``veomni_*``
#    adapter runs the Ulysses all-to-all), sparse layers run the TND MiniMax
#    sparse attention operators; both keep inputs/outputs packed. Attention
#    masks and KV cache are rejected upstream in the text model, so this is the
#    only path
# ================================================================
@config.override_method(
    "MiniMaxM3VLAttention.forward",
    description="Run dense layers through varlen attention and sparse layers through the TND sparse attention operators",
)
def minimax_m3_vl_attention_forward_patched(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # --- Patch.1 ---
    dropout = 0.0 if not self.training else self.attention_dropout
    if self.indexer is None:
        attn_output, _ = self.veomni_attn(
            self,
            query_states,
            key_states,
            value_states,
            None,
            dropout=dropout,
            scaling=self.scaling,
            **kwargs,
        )
    else:
        attn_output = _minimax_m3_sparse_attention(
            self,
            query_states,
            key_states,
            value_states,
            hidden_states,
            position_embeddings,
            dropout=dropout,
            **kwargs,
        )
    # --- Patch.1 ---

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), None


# ================================================================
# Patch: MiniMaxM3VL3DRotaryEmbedding.forward
# 1. accept CPU-precomputed grid_thw_list from the collator and hand a CPU grid
#    to `get_vision_position_ids`, so its `.tolist()` never syncs a CUDA tensor
#    in the hot path; external callers without the list keep the upstream path
# ================================================================
@config.override_method(
    "MiniMaxM3VL3DRotaryEmbedding.forward",
    description="Consume collator-precomputed MiniMax vision grid lists when available",
)
def minimax_m3_vl_3d_rotary_embedding_forward_patched(self, grid_thw, device, dtype, kwargs=None, grid_thw_list=None):
    # --- Patch.1 ---
    if grid_thw_list is not None:
        grid_thw = torch.tensor(grid_thw_list, dtype=torch.long)
    # --- Patch.1 ---
    coords = get_vision_position_ids(grid_thw, self.spatial_merge_size, include_temporal=True, kwargs=kwargs)
    coords = coords.to(device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        self.theta ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32, device=device) / self.axis_dim)
    )
    freqs = (coords.unsqueeze(-1) * inv_freq).reshape(coords.shape[0], -1)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


@config.modify_init(
    "MiniMaxM3VLVisionAttention",
    description="Bind instance-local attention VeomniOp",
)
def minimax_m3_vl_vision_attention_bind_ops(original_init, self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.veomni_attn = VeomniOp("attention", "standard", self.config._attn_implementation)


@config.override_method(
    "MiniMaxM3VLVisionAttention.forward",
    description="Always call the local attention VeomniOp",
)
def minimax_m3_vl_vision_attention_forward_patched(
    self, hidden_states, position_embeddings, attention_mask=None, **kwargs
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    queries = self.q_proj(hidden_states).view(hidden_shape)
    keys = self.k_proj(hidden_states).view(hidden_shape)
    values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    queries, keys = apply_rotary_pos_emb_vision(queries, keys, cos, sin)
    queries, keys = queries.transpose(1, 2), keys.transpose(1, 2)

    attn_output, attn_weights = self.veomni_attn(
        self,
        queries,
        keys,
        values,
        attention_mask,
        scaling=self.scale,
        dropout=0.0 if not self.training else self.dropout,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.out_proj(attn_output), attn_weights


# ================================================================
# Patch: MiniMaxM3VLVisionModel.forward
# 1. consume collator-precomputed grid / varlen-attention metadata
# 2. pad and slice 3D RoPE identically to the SP-sharded pixel rows
# 3. isolate the SP-padding tail and pass global sequence boundaries to every
#    vision attention layer; keep a runtime fallback for external callers
# 4. keep NPU cu_seqlens on CPU as required by its FA2 varlen interface
# ================================================================
@config.override_method(
    "MiniMaxM3VLVisionModel.forward",
    description="Add MiniMax vision SP RoPE and varlen-attention metadata handling",
)
def minimax_m3_vl_vision_model_forward_patched(self, pixel_values, grid_thw, **kwargs):
    r"""
    grid_thw (`torch.Tensor` of shape `(num_images, 3)`):
        The temporal, height and width of each image's feature grid, used to build the vision 3D RoPE.
    """
    vit_metadata = kwargs.pop("vit_metadata", None) or {}
    precomputed_grid_thw_list = vit_metadata.get("grid_thw_list")
    precomputed_cu_seqlens = vit_metadata.get("cu_seqlens")
    precomputed_max_seqlen = vit_metadata.get("max_seqlen")

    embeds = self.embeddings(pixel_values).to(self.pre_layrnorm.weight.dtype)
    grid_thw_list = _grid_thw_to_list(grid_thw, precomputed_grid_thw_list)
    total_seq_len = sum(t * h * w for t, h, w in grid_thw_list)

    cos, sin = self.rotary_emb(
        grid_thw,
        device=embeds.device,
        dtype=embeds.dtype,
        kwargs=kwargs,
        grid_thw_list=grid_thw_list,
    )

    parallel_state = get_parallel_state()
    _require_varlen_attention(
        self.config._attn_implementation,
        num_segments=len(grid_thw_list),
        ulysses_enabled=parallel_state.ulysses_enabled,
        attention_name="vision attention",
        causal=False,
    )

    sp_pad_seq_len = 0
    if parallel_state.sp_enabled:
        merge_unit = self.config.spatial_merge_size**2
        cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=merge_unit)
        sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=merge_unit)
        sp_pad_seq_len = embeds.shape[0] * parallel_state.sp_size - total_seq_len

    if precomputed_cu_seqlens is not None:
        cu_seqlens = precomputed_cu_seqlens.to(
            embeds.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            non_blocking=True,
        )
    else:
        cu_seqlens_list = [0]
        fallback_max_seqlen = 0
        for t, h, w in grid_thw_list:
            seq_len = t * h * w
            cu_seqlens_list.append(cu_seqlens_list[-1] + seq_len)
            fallback_max_seqlen = max(fallback_max_seqlen, seq_len)
        if sp_pad_seq_len > 0:
            cu_seqlens_list.append(cu_seqlens_list[-1] + sp_pad_seq_len)
        cu_seqlens = torch.tensor(
            cu_seqlens_list,
            device=embeds.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )

    if precomputed_max_seqlen is not None:
        max_seqlen = precomputed_max_seqlen
    else:
        max_seqlen = max(fallback_max_seqlen, sp_pad_seq_len)

    if IS_NPU_AVAILABLE:
        cu_seqlens = cu_seqlens.cpu()

    hidden_states = self.pre_layrnorm(embeds).unsqueeze(0)
    for layer in self.layers:
        hidden_states = layer(
            hidden_states,
            attention_mask=None,
            position_embeddings=(cos, sin),
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            **kwargs,
        )
    return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=hidden_states[:, 0])


# ================================================================
# Patch: MiniMaxM3VLVisionModel.dummy_forward (new)
# 1. add dummy_forward so ranks without pixel_values can still run the
#    shared vision/projector parameters under FSDP
# 2. mirror MainCollator's SP layout and pass host-built metadata so dummy
#    execution follows the same vision-forward contract as real inputs
# ================================================================
@config.override_method(
    "MiniMaxM3VLVisionModel.dummy_forward",
    description="Provide MiniMax dummy vision forward for asymmetric FSDP batches",
)
def minimax_m3_vl_vision_dummy_forward_patched(self):
    # --- Patch.1 ---
    patch_size = self.config.patch_size
    temporal_patch_size = self.config.temporal_patch_size
    in_channels = getattr(self.config, "num_channels", getattr(self.config, "in_channels", 3))
    merge_size = self.config.spatial_merge_size
    t = 1
    h_base = 2 * merge_size
    w = 2 * merge_size
    parallel_state = get_parallel_state()
    h = h_base * parallel_state.sp_size if parallel_state.sp_enabled else h_base
    num_patches = t * h * w
    pixel_row_size = in_channels * temporal_patch_size * patch_size * patch_size

    weight = self.embeddings.proj.weight
    pixel_values = torch.zeros((num_patches, pixel_row_size), dtype=weight.dtype, device=weight.device)
    if parallel_state.sp_enabled:
        pixel_values = sp_pad_and_slice(
            pixel_values,
            dim=0,
            pad_value=0,
            pad_scale=merge_size**2,
        )
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.long, device=weight.device)
    vit_metadata = {
        "grid_thw_list": [[t, h, w]],
        "cu_seqlens": torch.tensor([0, num_patches], dtype=torch.int32, device="cpu"),
        "max_seqlen": num_patches,
    }
    return self(pixel_values=pixel_values, grid_thw=grid_thw, vit_metadata=vit_metadata)
    # --- Patch.1 ---


@config.replace_class(
    "MiniMaxM3VLExperts",
    description="Remove the HF experts decorator and always call the moe_experts swiglu_oai VeomniOp",
)
class PatchedMiniMaxM3VLExperts(nn.Module):
    """MiniMax M3 routed experts on the ``moe_experts/swiglu_oai`` VeomniOp."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.limit = config.swiglu_limit
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit
        self.veomni_moe = VeomniOp("moe_experts", "swiglu_oai", resolve_op_impl("moe_implementation"))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        unused = self.gate_up_proj.new_empty(0)
        return self.veomni_moe(
            hidden_states,
            top_k_weights,
            top_k_index,
            unused,
            unused,
            self.down_proj,
            self.gate_up_proj,
            num_experts=self.num_experts,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_limit=self.swiglu_limit,
        )


# ================================================================
# Patch: MiniMaxM3VLModel.forward
# 1. consume MiniMax multimodal_metadata and pass per-modality grid lists
#    to the vision tower
# 2. pop VeOmni data-pipeline helper masks/ids before dispatching to the
#    language model
# 3. run a zero-valued dummy vision/projector path when FSDP has no visual
#    input for a modality on this rank, keeping image/video call counts equal
#    across ranks and preventing asymmetric collectives from hanging
# 4. when no precomputed image/video masks are passed under SP, gather the
#    sharded input_ids before deriving the placeholder masks
# 5. under SP, drop the gathered features of the vision SP-padding tail so they
#    match the placeholder tokens (and the fallback mask's count check)
# ================================================================
@config.override_method(
    "MiniMaxM3VLModel.forward",
    description="Add MiniMax VLM metadata fast path and FSDP dummy vision branch",
)
def minimax_m3_vl_model_forward_patched(
    self,
    input_ids=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    **kwargs,
):
    r"""
    image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of each image's feature grid.
    video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of each video's feature grid.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # --- Patch.4 ---
    parallel_state = get_parallel_state()
    # --- Patch.4 ---

    # --- Patch.1 ---
    multimodal_metadata = kwargs.pop("multimodal_metadata", None) or {}
    image_vit_kwargs = {
        "vit_metadata": {
            "grid_thw_list": multimodal_metadata.get("image_grid_thw_list"),
            "cu_seqlens": multimodal_metadata.get("vit_image_cu_seqlens"),
            "max_seqlen": multimodal_metadata.get("vit_image_max_seqlen"),
        }
    }
    video_vit_kwargs = {
        "vit_metadata": {
            "grid_thw_list": multimodal_metadata.get("video_grid_thw_list"),
            "cu_seqlens": multimodal_metadata.get("vit_video_cu_seqlens"),
            "max_seqlen": multimodal_metadata.get("vit_video_max_seqlen"),
        }
    }
    # --- Patch.1 ---
    if parallel_state.sp_enabled:
        inputs_embeds = gather_outputs(inputs_embeds, gather_dim=1, group=parallel_state.sp_group)
    # --- Patch.2 ---
    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)
    kwargs.pop("mm_token_type_ids", None)
    # --- Patch.2 ---

    # --- Patch.3 ---
    image_features = None
    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values=pixel_values, image_grid_thw=image_grid_thw, **image_vit_kwargs
        ).pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
        if parallel_state.sp_enabled:
            image_features = gather_outputs(image_features, gather_dim=0, group=parallel_state.sp_group)
            # --- Patch.5 ---
            image_features = _drop_sp_padding_features(
                image_features,
                image_grid_thw,
                image_vit_kwargs["vit_metadata"]["grid_thw_list"],
                self.vision_tower.config.spatial_merge_size,
            )
            # --- Patch.5 ---

    elif parallel_state.fsdp_enabled:
        fake_vision = self.vision_tower.dummy_forward()
        fake_features = self.multi_modal_projector(fake_vision.last_hidden_state.squeeze(0))
        inputs_embeds = inputs_embeds + fake_features.mean().to(inputs_embeds.device, inputs_embeds.dtype) * 0.0

    video_features = None
    if pixel_values_videos is not None:
        video_features = self.get_video_features(
            pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **video_vit_kwargs
        ).pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
        if parallel_state.sp_enabled:
            video_features = gather_outputs(video_features, gather_dim=0, group=parallel_state.sp_group)
            # --- Patch.5 ---
            video_features = _drop_sp_padding_features(
                video_features,
                video_grid_thw,
                video_vit_kwargs["vit_metadata"]["grid_thw_list"],
                self.vision_tower.config.spatial_merge_size,
            )
            # --- Patch.5 ---
    elif parallel_state.fsdp_enabled:
        fake_vision = self.vision_tower.dummy_forward()
        fake_features = self.multi_modal_projector(fake_vision.last_hidden_state.squeeze(0))
        inputs_embeds = inputs_embeds + fake_features.mean().to(inputs_embeds.device, inputs_embeds.dtype) * 0.0
    # --- Patch.3 ---

    if image_mask is None or video_mask is None:
        # Fallback for callers that pass no precomputed masks. Under SP,
        # ``inputs_embeds`` is already gathered to the full sequence while
        # ``input_ids`` is still this rank's slice; gather it like Qwen-VL does.
        if input_ids is not None and parallel_state.sp_enabled:
            input_ids_list = [torch.zeros_like(input_ids) for _ in range(parallel_state.sp_size)]
            dist.all_gather(input_ids_list, input_ids, group=parallel_state.sp_group)
            input_ids = torch.cat(input_ids_list, dim=1)
        image_mask, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds, image_features=image_features, video_features=video_features
        )
    else:
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    if image_features is not None:
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
    if video_features is not None:
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

    if parallel_state.sp_enabled:
        inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, group=parallel_state.sp_group)

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )

    return MiniMaxM3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
        image_hidden_states=image_features,
        video_hidden_states=video_features,
    )


# ================================================================
# Patch: MiniMaxM3VLTextModel.forward
# 1. reject router-logit capture because MiniMax's routing semantics are not
#    compatible with the upstream Switch-style auxiliary loss
# 2. accept only packed inputs so language attention has a single path: the
#    collator's varlen kwargs flow unchanged to the attention layers, and a
#    lone unpadded sequence without them is described as a one-segment pack;
#    padded batches and KV cache are rejected (context parallelism is refused
#    at model build by ``check_context_parallel_supported``); dense layers
#    need a varlen backend to keep packed samples apart
# 3. generate RoPE once from the local packed position IDs; attention and the
#    indexer apply it before their respective Ulysses communications
# ================================================================
@config.override_method(
    "MiniMaxM3VLTextModel.forward",
    description="Accept only packed inputs and reject padded batches and KV cache",
)
def minimax_m3_vl_text_model_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    **kwargs,
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # --- Patch.1 ---
    output_router_logits = kwargs.pop("output_router_logits", None)
    if output_router_logits or self.config.output_router_logits:
        raise ValueError(
            "MiniMax M3 router auxiliary loss is disabled: the upstream Switch-style loss does not match "
            "MiniMax M3's sigmoid routing with e_score_correction_bias."
        )
    # --- Patch.1 ---

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # --- Patch.2 ---
    if past_key_values is not None:
        raise ValueError("MiniMax M3 VL language model in VeOmni is training-only and does not support KV cache.")
    parallel_state = get_parallel_state()
    if kwargs.get("cu_seq_lens_q") is None:
        if parallel_state.sp_enabled or inputs_embeds.shape[0] != 1:
            raise ValueError(
                "MiniMax M3 VL language attention requires packed inputs with cu_seq_lens_q/max_length_q, as "
                "produced by VeOmni's collator. Without them only a single unpadded sequence (batch size 1, no "
                "sequence parallelism) is accepted; padded batches are not supported."
            )
        # A lone sequence is a one-segment pack, so its mask must be all ones.
        # The check syncs, but only on this path: the collator always provides
        # cu_seq_lens_q. Metadata is built from shapes, on the device for CUDA
        # FlashAttention and on the host for NPU's host-side varlen interface.
        if attention_mask is not None and (attention_mask.dim() != 2 or not bool(attention_mask.all())):
            raise ValueError(
                "MiniMax M3 VL language attention without cu_seq_lens_q/max_length_q only accepts an unpadded "
                "sequence: attention_mask must be None or a [1, seq_len] all-ones mask."
            )
        sequence_length = inputs_embeds.shape[1]
        cu_seq_lens = torch.tensor(
            [0, sequence_length],
            dtype=torch.int32,
            device=torch.device("cpu") if IS_NPU_AVAILABLE else inputs_embeds.device,
        )
        kwargs.update(
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=sequence_length,
            max_length_k=sequence_length,
        )
    elif inputs_embeds.shape[0] != 1:
        # The sparse attention operators consume one packed [T, H, D] row.
        raise ValueError(
            "MiniMax M3 VL packed inputs (cu_seq_lens_q/max_length_q) must have batch size 1, "
            f"got {inputs_embeds.shape[0]}."
        )
    if any(layer.self_attn.indexer is None for layer in self.layers[: self.config.num_hidden_layers]):
        _require_varlen_attention(
            self.config._attn_implementation,
            num_segments=kwargs["cu_seq_lens_q"].numel() - 1,
            ulysses_enabled=parallel_state.ulysses_enabled,
            attention_name="dense language attention",
            causal=True,
        )
    # --- Patch.2 ---

    hidden_states = inputs_embeds

    if position_ids is None:
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

    # --- Patch.3 ---
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
    # --- Patch.3 ---

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)

    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


@config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.__init__",
    description="Bind the ForCausalLMLoss cross-entropy VeomniOp",
)
def minimax_m3_vl_sparse_for_conditional_generation_init_patched(self, config):
    super().__init__(config)
    self.model = MiniMaxM3VLModel(config)
    self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
    self.veomni_ce = VeomniOp("cross_entropy_loss", "standard", resolve_op_impl("cross_entropy_loss_implementation"))
    self.loss_function = partial(ForCausalLMLoss, op=self.veomni_ce)
    self.post_init()


@config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.forward",
    description="Always call the bound ForCausalLMLoss and unpack its fused-loss tuple",
)
def minimax_m3_vl_sparse_for_conditional_generation_forward_patched(
    self,
    input_ids=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    logits_to_keep=0,
    **kwargs,
):
    r"""
    image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of each image's feature grid.
    video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of each video's feature grid.
    """
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    # --- Patch.1 ---
    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        loss, logits, fused_linear_aux = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.text_config.vocab_size,
            hidden_states=hidden_states,
            weights=self.lm_head.weight,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states)
    # --- Patch.1 ---

    return MiniMaxM3VLCausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        fused_linear_aux=fused_linear_aux,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        video_hidden_states=outputs.video_hidden_states,
    )


# Keep MiniMax's multimodal output fields while exposing VeOmni's fused-loss
# payload as a constructor field, so ModelOutput pytree flattening remains
# FSDP2-safe. This helper is emitted immediately after the upstream output
# class by patchgen and therefore works for both GPU and NPU artifacts.
@config.add_helper_after("MiniMaxM3VLCausalLMOutputWithPast")
@dataclass
class MiniMaxM3VLCausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, MiniMaxM3VLCausalLMOutputWithPast):
    """MiniMaxM3VLCausalLMOutputWithPast plus VeOmni fused-loss payload."""


@config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_parallel_plan",
    description="Register MiniMax M3 VL expert parallel plan for the multimodal training path",
)
def minimax_m3_vl_get_parallel_plan_patched(self):
    from ..parallel_plan import get_vlm_parallel_plan

    return get_vlm_parallel_plan()


@config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_position_id_func",
    description="Use VeOmni's default 1-D packed-sequence position IDs for MiniMax M3 VL SFT data",
)
def minimax_m3_vl_get_position_id_func_patched(self):
    return None


@config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_metadata_collate_func",
    description="Expose MiniMax CPU-side vision grid metadata derivation to the VeOmni collator",
)
def minimax_m3_vl_get_metadata_collate_func_patched(self):
    return collate_multimodal_metadata  # noqa: F821 defined via add_helper
