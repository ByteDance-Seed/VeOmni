# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Patch configuration for Qwen3-VL transformers>=5.2.0 code generation.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3_vl.qwen3_vl_gpu_patch_gen_config -o veomni/models/transformers/qwen3_vl/generated --diff
"""

import copy
from functools import lru_cache, partial
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import (
    ALL_ATTENTION_FUNCTIONS,
    is_flash_attention_requested,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    sp_pad_and_slice,
)
from veomni.distributed.sequence_parallel.async_ulysses import (
    async_ulysses_output_projection,
    async_ulysses_qkv_projection,
)
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.device import IS_NPU_AVAILABLE


config = PatchConfig(
    source_module="transformers.models.qwen3_vl.modeling_qwen3_vl",
    target_file="patched_modeling_qwen3_vl_gpu.py",
    description="Qwen3-VL with VeOmni v5 compatibility (SP + async Ulysses + deepstack + fused-loss)",
)

# Imports consumed by the helpers below + the patched methods need to be
# emitted into the generated file. We use `add_post_import_block` (not
# `add_import`) to keep this dense import set compact and ruff-clean in the
# generated output.
config.add_post_import_block("""
import copy

import numpy as np
from functools import lru_cache, partial
from types import SimpleNamespace

import torch.distributed as dist

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    sp_pad_and_slice,
)
from veomni.distributed.sequence_parallel.async_ulysses import (
    async_ulysses_output_projection,
    async_ulysses_qkv_projection,
)
from veomni.utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.device import IS_NPU_AVAILABLE
""")


# ================================================================
# Module-level helpers injected after the import block
# 1. `rot_pos_ids` — vllm-adapted lru_cached pos-id builder used by
#    the patched `Qwen3VLVisionModel.rot_pos_emb` below
# 2. `_qwen3_vl_async_ulysses_attention_forward` — async Ulysses
#    attention path used by the patched `Qwen3VLTextAttention.forward`
# 3. `get_position_id` — picklable wrapper that the patched
#    `Qwen3VLForConditionalGeneration.get_position_id_func` returns via
#    `partial(...)` so the VeOmni data pipeline can precompute
#    multimodal position_ids on CPU worker processes
# ================================================================


# Copied and adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_vl.py#L431
@config.add_helper
@lru_cache(maxsize=1024)
def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
    if isinstance(h, torch.Tensor):
        h = int(h.item())
    if isinstance(w, torch.Tensor):
        w = int(w.item())
    if isinstance(spatial_merge_size, torch.Tensor):
        spatial_merge_size = int(spatial_merge_size.item())
    hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
    h_div = h // spatial_merge_size
    w_div = w // spatial_merge_size
    hpos_ids = hpos_ids.reshape(h_div, spatial_merge_size, w_div, spatial_merge_size)
    hpos_ids = hpos_ids.transpose(0, 2, 1, 3).flatten()

    wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
    wpos_ids = wpos_ids.reshape(h_div, spatial_merge_size, w_div, spatial_merge_size)
    wpos_ids = wpos_ids.transpose(0, 2, 1, 3).flatten()

    return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))


@config.add_helper
def _qwen3_vl_async_ulysses_attention_forward(
    self,
    hidden_states,
    attention_mask,
    position_embeddings,
    **kwargs,
):
    """Async Ulysses attention forward path for Qwen3VLTextAttention.

    Fuses QKV projection + q_norm/k_norm + RoPE + ulysses all-to-all, and
    the output projection + reverse all-to-all. Requires a flash-attention
    implementation because of the packed-varlen contract.
    """
    if not is_flash_attention_requested(self.config):
        raise ValueError(
            "Async Ulysses attention only supports flash attention implementations. "
            f"Current implementation: '{self.config._attn_implementation}'. "
            "Please set attn_implementation to a flash attention variant or disable async Ulysses."
        )

    unpadded_seq_len = hidden_states.size(1)

    q, k, v = async_ulysses_qkv_projection(
        hidden_states=hidden_states,
        seq_dimension=1,
        head_dimension=2,
        q_weight=self.q_proj.weight,
        q_bias=self.q_proj.bias,
        k_weight=self.k_proj.weight,
        k_bias=self.k_proj.bias,
        v_weight=self.v_proj.weight,
        v_bias=self.v_proj.bias,
        norm_type="rmsnorm",
        norm_q_weight=self.q_norm.weight,
        norm_q_bias=None,
        norm_k_weight=self.k_norm.weight,
        norm_k_bias=None,
        normalized_shape=self.head_dim,
        eps=self.config.rms_norm_eps,
        unpadded_dim_size=unpadded_seq_len * get_ulysses_sequence_parallel_world_size(),
        head_dim=self.head_dim,
    )

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    cos, sin = position_embeddings
    # cos/sin are per-rank sharded along the sequence dim; async path operates on
    # full-seq QKV, so gather them back across the SP group. In v5,
    # `apply_interleaved_mrope` collapses the leading 3-axis from mrope, so the
    # incoming shape here is (bs, seq_len, head_dim) and the seq dim is 1.
    cos = gather_outputs(cos, gather_dim=1, group=get_parallel_state().sp_group)
    sin = gather_outputs(sin, gather_dim=1, group=get_parallel_state().sp_group)

    query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        v,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        skip_ulysses=True,
        **kwargs,
    )

    attn_output = async_ulysses_output_projection(
        hidden_states=attn_output,
        seq_dimension=1,
        head_dimension=2,
        proj_weight=self.o_proj.weight,
        proj_bias=self.o_proj.bias,
        unpadded_dim_size=attn_output.shape[1],
    )
    return attn_output, attn_weights


@config.add_helper
def get_position_id(main_func, self, **kwargs):
    # Must be a module-level function for multiprocessing pickle
    position_ids, rope_deltas = main_func(self, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}


# ================================================================
# Patch: Qwen3VLVisionAttention.forward
# 1. accept precomputed max_seqlen from outer forward so the
#    `(cu_seqlens[1:] - cu_seqlens[:-1]).max()` CPU-GPU sync happens once
#    (hoisted to the outer visual forward) instead of once per layer
# ================================================================
@config.override_method(
    "Qwen3VLVisionAttention.forward",
    description="Use precomputed max_seqlen passed from outer forward to hoist CPU-GPU sync out of the layer loop",
)
def qwen3_vl_vision_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    # --- Patch.1 ---
    max_seqlen: int,
    # --- Patch.1 ---
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    if is_flash_attention_requested(self.config):
        # --- Patch.1 ---
        # max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        # --- Patch.1 ---
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
    else:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)]
        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


# ================================================================
# Patch: Qwen3VLVisionBlock.forward
# 1. thread the precomputed max_seqlen down into the attention call
#    (paired with the Qwen3VLVisionAttention.forward patch above)
# ================================================================
@config.override_method(
    "Qwen3VLVisionBlock.forward",
    description="Propagate precomputed max_seqlen to attention to avoid per-layer CPU-GPU sync",
)
def qwen3_vl_vision_block_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    # --- Patch.1 ---
    max_seqlen: int,
    # --- Patch.1 ---
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        # --- Patch.1 ---
        max_seqlen=max_seqlen,
        # --- Patch.1 ---
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states


# ================================================================
# Patch: Qwen3VLVisionModel.rot_pos_emb
# 1. swap the upstream Python-list-indexed grid builder for a vllm-style
#    lru_cached `rot_pos_ids` helper keyed on `(h, w, merge_size)` so
#    repeated (h, w) tiles across a batch hit the cache
# ================================================================
@config.override_method(
    "Qwen3VLVisionModel.rot_pos_emb",
    description="Use lru_cached rot_pos_ids helper (vllm-style) to avoid per-image Python loops",
)
def qwen3_vl_vision_rot_pos_emb_patched(self, grid_thw: torch.Tensor) -> torch.Tensor:
    # --- Patch.1 ---
    merge_size = self.spatial_merge_size

    max_hw = int(grid_thw[:, 1:].max().item())
    freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
    device = freq_table.device

    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in grid_thw:
        coords = rot_pos_ids(height, width, merge_size).to(device)  # noqa: F821 defined via add_post_import_block
        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)
        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    embeddings = freq_table[pos_ids]
    embeddings = embeddings.flatten(1)
    return embeddings
    # --- Patch.1 ---


# ================================================================
# Patch: Qwen3VLVisionModel.fast_pos_embed_interpolate
# 1. a fully-tensorized vllm-adapted implementation — replaces the
#    upstream per-image Python list extend / tolist path with
#    meshgrid-based bilinear weight synthesis so the whole batch stays
#    on device (also avoids the final torch.split + per-image permute)
# ================================================================
@config.override_method(
    "Qwen3VLVisionModel.fast_pos_embed_interpolate",
    description="Tensorized meshgrid implementation of fast_pos_embed_interpolate",
)
def qwen3_vl_vision_fast_pos_embed_interpolate_patched(self, grid_thw):
    # --- Patch.1 ---
    num_grid_per_side = self.num_grid_per_side
    m_size = self.spatial_merge_size
    hidden_dim = self.pos_embed.embedding_dim

    outputs = []
    dtype = self.pos_embed.weight.dtype
    for t, h, w in grid_thw:
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h, device=self.device, dtype=torch.float64)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w, device=self.device, dtype=torch.float64)

        h_floor = h_idxs.to(torch.long)
        w_floor = w_idxs.to(torch.long)
        h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
        w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
        h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
        h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

        # reuse dh*dw to avoid duplicate multiplies
        w11 = dh_grid * dw_grid
        w10 = dh_grid - w11
        w01 = dw_grid - w11
        w00 = 1 - dh_grid - w01

        h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
        w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
        h_grid_idx = h_grid * num_grid_per_side

        indices = (h_grid_idx + w_grid).reshape(4, -1)
        weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
        weights = weights.to(dtype=dtype)

        embeds = self.pos_embed(indices) * weights
        combined = embeds[0] + embeds[1] + embeds[2] + embeds[3]
        combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)

        combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
        repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)

        outputs.append(repeated)

    return torch.cat(outputs, dim=0)
    # --- Patch.1 ---


# ================================================================
# Patch: Qwen3VLVisionModel.forward
# 1. SP pad pos_embeds / cos / sin so sharded hidden_states get the
#    matching pos embedding (pad_scale=4 matches the upstream SP
#    sharding scale used for ViT inputs)
# 2. use pre-padding cu_seqlens[-1] as the un-sliced total seq length
# 3. when SP is enabled, extend cu_seqlens with the SP-padded sentinel
#    so the padded tokens belong to a valid attention window
# 4. precompute max_seqlen once here (single CPU-GPU sync hoisted out
#    of the attention layer loop) and thread it into each block
# 5. on NPU, move cu_seqlens to CPU — NPU FA2 varlen path requires it
# 6. return BaseModelOutputWithDeepstackFeatures (v5 contract: the v4
#    path returned `(merged, deepstack_list)` as a bare tuple)
# ================================================================
@config.override_method(
    "Qwen3VLVisionModel.forward",
    description="VeOmni SP + deepstack + precomputed max_seqlen; return BaseModelOutputWithDeepstackFeatures",
)
def qwen3_vl_vision_forward_patched(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithDeepstackFeatures:
    hidden_states = self.patch_embed(hidden_states)

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

    # --- Patch.1 ---
    if get_parallel_state().sp_enabled:
        pos_embeds = sp_pad_and_slice(pos_embeds, dim=0, pad_value=0, pad_scale=4)
    # --- Patch.1 ---

    hidden_states = hidden_states + pos_embeds

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)

    # --- Patch.2 ---
    total_seq_len = cu_seqlens[-1]
    rotary_pos_emb = rotary_pos_emb.reshape(total_seq_len, -1)
    # --- Patch.2 ---

    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    if get_parallel_state().sp_enabled:
        # --- Patch.1 ---
        cos, sin = position_embeddings
        cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=4)
        sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=4)
        position_embeddings = (cos, sin)
        # --- Patch.1 ---

        # --- Patch.3 ---
        sp_size = get_parallel_state().sp_size
        pad_seq_len = seq_len * sp_size - total_seq_len.item()
        if pad_seq_len > 0:
            new_cumsum = cu_seqlens[-1] + pad_seq_len
            cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
        # --- Patch.3 ---

    deepstack_feature_lists = []

    # --- Patch.4 ---
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()
    # --- Patch.4 ---

    # --- Patch.5 ---
    if IS_NPU_AVAILABLE:
        cu_seqlens = cu_seqlens.cpu()
    # --- Patch.5 ---

    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            # --- Patch.4 ---
            max_seqlen=max_seqlen,
            # --- Patch.4 ---
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    merged_hidden_states = self.merger(hidden_states)

    # --- Patch.6 ---
    return BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
        deepstack_features=deepstack_feature_lists,
    )
    # --- Patch.6 ---


# ================================================================
# Patch: Qwen3VLVisionModel.dummy_forward (new)
# 1. add dummy_forward so ranks without pixel_values can still run the
#    visual encoder under FSDP — otherwise reduce-scatter hangs when
#    some ranks get None pixel_values while others have real images
# 2. SP-aware shape: grid_thw height is scaled by sp_size so the dummy
#    batch stays a clean multiple after sequence slicing. Shapes are
#    derived from the vision config (patch_size / temporal_patch_size /
#    in_channels / spatial_merge_size) so model variants don't break
# ================================================================
@config.override_method(
    "Qwen3VLVisionModel.dummy_forward",
    description="Provide dummy vision forward for FSDP path with SP-aware shape",
)
def qwen3_vl_vision_dummy_forward_patched(self):
    # --- Patch.1 ---
    # Derive dummy shape from config so variants with different patch sizes /
    # channel counts work without silent breakage:
    #   - pixel row:  in_channels * temporal_patch_size * patch_size**2
    #   - grid_thw:   one (t=1) frame with a (h x w) grid of patches where
    #                 h and w are multiples of spatial_merge_size (required
    #                 by the ViT merger) — we use 2*merge_size for both.
    patch_size = self.config.patch_size
    temporal_patch_size = self.config.temporal_patch_size
    in_channels = self.config.in_channels
    merge_size = self.spatial_merge_size

    t = 1
    h_base = 2 * merge_size
    w = 2 * merge_size

    # --- Patch.2 ---
    if get_parallel_state().sp_enabled:
        h = h_base * get_parallel_state().sp_size
    else:
        h = h_base
    # --- Patch.2 ---

    num_patches = t * h * w
    pixel_row_size = in_channels * temporal_patch_size * patch_size * patch_size
    pixel_values = torch.zeros((num_patches, pixel_row_size), dtype=self.dtype, device=self.device)
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.int32, device=self.device)
    return self(hidden_states=pixel_values, grid_thw=grid_thw)
    # --- Patch.1 ---


# ================================================================
# Patch: Qwen3VLTextAttention.forward
# 1. route through the async Ulysses fused QKV/Output projection path
#    (defined via add_post_import_block as a module-level helper) when
#    `get_parallel_state().async_enabled` is True; otherwise fall
#    through to the upstream logic unchanged
# ================================================================
@config.override_method(
    "Qwen3VLTextAttention.forward",
    description="Route through async Ulysses fused QKV/Output projection when async_enabled",
)
def qwen3_vl_text_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # --- Patch.1 ---
    if get_parallel_state().async_enabled:
        return _qwen3_vl_async_ulysses_attention_forward(  # noqa: F821 defined via add_post_import_block
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    # --- Patch.1 ---

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# ================================================================
# Patch: Qwen3VLTextModel._deepstack_process
# 1. handle the case where `visual_pos_masks is None` (both image and
#    video pixel_values were None on this rank) — still touch
#    visual_embeds so FSDP sees the params; add 0.0 so gradients flow
#    without mutating hidden_states
# ================================================================
@config.override_method(
    "Qwen3VLTextModel._deepstack_process",
    description="Handle visual_pos_masks=None by adding 0.0 so FSDP sees the visual params",
)
def qwen3_vl_text_deepstack_process_patched(
    self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
):
    # --- Patch.1 ---
    if visual_pos_masks is None:
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states + visual_embeds.mean() * 0.0
        return hidden_states
    # --- Patch.1 ---

    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    hidden_states = hidden_states.clone()
    local_this = hidden_states[visual_pos_masks, :] + visual_embeds
    hidden_states[visual_pos_masks, :] = local_this
    return hidden_states


# ================================================================
# Patch: Qwen3VLModel.get_image_features
# 1. skip the upstream `torch.split(image_embeds, split_sizes)` — VeOmni
#    needs the flat tensor for SP all-to-all, and the downstream
#    masked_scatter is indexed by a single n_image_tokens slice rather
#    than a per-image tuple
# ================================================================
@config.override_method(
    "Qwen3VLModel.get_image_features",
    description="Return flat image_embeds tensor (skip per-image torch.split)",
)
def qwen3_vl_model_get_image_features_patched(
    self,
    pixel_values: torch.FloatTensor,
    image_grid_thw: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithDeepstackFeatures:
    pixel_values = pixel_values.type(self.visual.dtype)
    vision_output: BaseModelOutputWithDeepstackFeatures = self.visual(
        pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs
    )
    # --- Patch.1 ---
    # image_embeds = vision_output.pooler_output
    # split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    # vision_output.pooler_output = torch.split(image_embeds, split_sizes)
    # --- Patch.1 ---
    return vision_output


# ================================================================
# Patch: Qwen3VLModel.get_placeholder_mask
# 1. return raw (image_mask, video_mask) bool tensors without the
#    HF v5 `inputs_embeds` / `*_features` shape-validation branch —
#    VeOmni needs the masks on the *full* all-gathered input_ids,
#    which have a different seq-len than the SP-sliced inputs_embeds
# ================================================================
@config.override_method(
    "Qwen3VLModel.get_placeholder_mask",
    description="Return raw image/video placeholder bool masks for VeOmni SP-aware masked_scatter",
)
def qwen3_vl_model_get_placeholder_mask_patched(
    self,
    input_ids: torch.LongTensor,
    inputs_embeds: torch.FloatTensor | None = None,
    image_features: torch.FloatTensor | None = None,
    video_features: torch.FloatTensor | None = None,
):
    # --- Patch.1 ---
    special_image_mask = input_ids == self.config.image_token_id
    special_video_mask = input_ids == self.config.video_token_id
    # --- Patch.1 ---
    return special_image_mask, special_video_mask


# ================================================================
# Patch: Qwen3VLModel.forward
# 1. Ulysses SP scatter/gather around the visual-embed masked_scatter
#    (gather_seq_scatter_heads on inputs_embeds + image/video embeds;
#    gather_heads_scatter_seq after fill-back); slice image/video mask
#    + deepstack embeds to the per-rank range
# 2. precomputed image/video masks: use kwargs["image_mask"/"video_mask"]
#    from the VeOmni data pipeline; otherwise all-gather input_ids
#    across SP group and recompute on the full sequence
# 3. pop ViT-incompatible flash-attn kwargs (cu_seq_lens_q/k,
#    max_length_q/k) before calling the visual encoder; restore onto
#    the language-model kwargs afterwards
# 4. FSDP dummy_forward branch when pixel_values / pixel_values_videos
#    are None on this rank — keeps visual params on the FSDP all-reduce
#    graph via fake_embeds * 0, and threads a fake deepstack tensor
#    down into `_deepstack_process` so those params participate too
# 5. honor precomputed 3D position_ids: (bs, 3, L) -> (3, bs, L)
# ================================================================
@config.override_method(
    "Qwen3VLModel.forward",
    description="VeOmni SP + precomputed position-id + dummy-forward + deepstack multimodal patches",
)
def qwen3_vl_model_forward_patched(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen3VLModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # --- Patch.2 ---
    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)
    if video_mask is None and image_mask is None:
        if get_parallel_state().sp_enabled:
            input_ids_list = [torch.zeros_like(input_ids) for _ in range(get_parallel_state().sp_size)]
            dist.all_gather(input_ids_list, input_ids, group=get_parallel_state().sp_group)
            input_ids_full = torch.cat(input_ids_list, dim=1)
        else:
            input_ids_full = input_ids
        image_mask, video_mask = self.get_placeholder_mask(input_ids_full)
    # --- Patch.2 ---

    # --- Patch.3 ---
    flash_attn_kwargs = {}
    for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
        if key in kwargs:
            flash_attn_kwargs[key] = kwargs.pop(key)
    # --- Patch.3 ---

    # --- Patch.1 ---
    if get_parallel_state().sp_enabled:
        inputs_embeds = gather_seq_scatter_heads(
            inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
        )
    # --- Patch.1 ---

    fake_deepstack = None

    if pixel_values is not None:
        image_outputs: BaseModelOutputWithDeepstackFeatures = self.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        )
        image_embeds = image_outputs.pooler_output
        deepstack_image_embeds = image_outputs.deepstack_features

        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            image_embeds = gather_seq_scatter_heads(
                image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )
            deepstack_image_embeds = [
                gather_outputs(embed, gather_dim=0, group=get_parallel_state().sp_group)
                for embed in deepstack_image_embeds
            ]
        # --- Patch.1 ---

        n_image_tokens = image_mask.sum().long().item()
        embeds_image_mask = (
            image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )
        image_embeds = image_embeds[:n_image_tokens]
        deepstack_image_embeds = [embed[:n_image_tokens] for embed in deepstack_image_embeds]
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_image_mask, image_embeds)

        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            seq_len = image_mask.shape[1]
            seq_per_rank = seq_len // get_parallel_state().sp_size
            rank_start = get_parallel_state().sp_rank * seq_per_rank
            rank_end = rank_start + seq_per_rank

            deepstack_offset = image_mask[:, :rank_start].sum().item()
            image_mask = image_mask[:, rank_start:rank_end]
            deepstack_len = image_mask.sum().item()

            deepstack_image_embeds = [
                embed[deepstack_offset : deepstack_offset + deepstack_len] for embed in deepstack_image_embeds
            ]
        # --- Patch.1 ---

    elif get_parallel_state().fsdp_enabled:
        # --- Patch.4 ---
        fake_vision = self.visual.dummy_forward()
        fake_embeds = fake_vision.pooler_output.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        fake_deepstack = fake_vision.deepstack_features
        # --- Patch.4 ---

    if pixel_values_videos is not None:
        video_outputs: BaseModelOutputWithDeepstackFeatures = self.get_video_features(
            pixel_values_videos, video_grid_thw, return_dict=True
        )
        video_embeds = video_outputs.pooler_output
        deepstack_video_embeds = video_outputs.deepstack_features

        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            video_embeds = gather_seq_scatter_heads(
                video_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )
            deepstack_video_embeds = [
                gather_outputs(embed, gather_dim=0, group=get_parallel_state().sp_group)
                for embed in deepstack_video_embeds
            ]
        # --- Patch.1 ---

        n_video_tokens = video_mask.sum().long().item()
        embeds_video_mask = (
            video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )
        video_embeds = video_embeds[:n_video_tokens]
        deepstack_video_embeds = [embed[:n_video_tokens] for embed in deepstack_video_embeds]
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_video_mask, video_embeds)

        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            seq_len = video_mask.shape[1]
            seq_per_rank = seq_len // get_parallel_state().sp_size
            rank_start = get_parallel_state().sp_rank * seq_per_rank
            rank_end = rank_start + seq_per_rank

            deepstack_offset = video_mask[:, :rank_start].sum().item()
            video_mask = video_mask[:, rank_start:rank_end]
            deepstack_len = video_mask.sum().item()

            deepstack_video_embeds = [
                embed[deepstack_offset : deepstack_offset + deepstack_len] for embed in deepstack_video_embeds
            ]
        # --- Patch.1 ---

    elif get_parallel_state().fsdp_enabled:
        # --- Patch.4 ---
        fake_vision = self.visual.dummy_forward()
        fake_embeds = fake_vision.pooler_output.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        fake_deepstack = fake_vision.deepstack_features
        # --- Patch.4 ---

    # --- Patch.1 ---
    if get_parallel_state().sp_enabled:
        inputs_embeds = gather_heads_scatter_seq(
            inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
        )
    # --- Patch.1 ---

    visual_pos_masks = None
    deepstack_visual_embeds = None

    if pixel_values is not None and pixel_values_videos is not None:
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif pixel_values is not None:
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif pixel_values_videos is not None:
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds
    else:
        # --- Patch.4 ---
        # Pass fake deepstack so _deepstack_process still touches the
        # visual params; visual_pos_masks=None triggers the add-0.0 branch
        if fake_deepstack is not None:
            deepstack_visual_embeds = fake_deepstack
        # --- Patch.4 ---

    if position_ids is None:
        # --- Patch.5 ---
        # HF v5 may pass attention_mask as a dict (keyed by attention type); the
        # rope-index helpers below still expect a plain tensor, so unwrap it.
        if isinstance(attention_mask, dict):
            attention_mask_tensor = attention_mask.get("full_attention", None)
        else:
            attention_mask_tensor = attention_mask
        # Under Ulysses SP, input_ids/inputs_embeds here are per-rank slices, so
        # computing mrope positions on the fly would drift. The training path is
        # expected to go through the VeOmni `get_position_id_func` precompute
        # (via data pipeline); raise loudly if we somehow land here with SP on.
        if get_parallel_state().sp_enabled:
            raise RuntimeError(
                "Qwen3VLModel.forward: position_ids is None while sequence parallel "
                "is enabled; multimodal position_ids must be precomputed via "
                "`get_position_id_func` in the VeOmni data pipeline."
            )
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_tensor,
            past_key_values=past_key_values,
        )
        # --- Patch.5 ---
    else:
        # --- Patch.5 ---
        if position_ids.dim() == 3 and position_ids.shape[1] == 3:
            position_ids = position_ids.transpose(0, 1).contiguous()
        # --- Patch.5 ---

    # --- Patch.3 ---
    kwargs.update(flash_attn_kwargs)
    # --- Patch.3 ---

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )


# ================================================================
# Patch: Qwen3VLForConditionalGeneration.get_position_id_func (new)
# 1. wrap Qwen3VLModel.get_rope_index so the VeOmni data pipeline can
#    precompute multimodal position_ids on CPU worker processes
# 2. overwrite multimodal token ids with VeOmni constants
#    (IMAGE_INPUT_INDEX / VIDEO_INPUT_INDEX) so input_ids produced by
#    our data pipeline match the fake config here
# ================================================================
@config.override_method(
    "Qwen3VLForConditionalGeneration.get_position_id_func",
    description="Use VeOmni precomputed position-id function and unified multimodal token ids",
)
def qwen3_vl_get_position_id_func_patched(self):
    # --- Patch.1 ---
    fake_config = copy.copy(self.config)
    # --- Patch.2 ---
    fake_config.image_token_id = IMAGE_INPUT_INDEX
    fake_config.video_token_id = VIDEO_INPUT_INDEX
    # --- Patch.2 ---
    fake_model = SimpleNamespace(config=fake_config)
    return partial(get_position_id, Qwen3VLModel.get_rope_index, fake_model)  # noqa: F821 defined via add_post_import_block
    # --- Patch.1 ---


# ================================================================
# Patch: Qwen3VLForConditionalGeneration.forward
# 1. use the unified VeOmni fused loss_function (handles Ulysses
#    internally; takes hidden_states + lm_head weights instead of
#    pre-computed logits) — avoids materializing full-vocab logits
#    when labels are provided
# 2. drop the HF v5 logits-first path: only compute logits when
#    labels is None (inference), otherwise let loss_function fuse
#    matmul + cross-entropy
# ================================================================
@config.override_method(
    "Qwen3VLForConditionalGeneration.forward",
    description="Use VeOmni unified fused loss_function path",
)
def qwen3_vl_for_conditional_generation_forward_patched(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen3VLCausalLMOutputWithPast:
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    # --- Patch.1 ---
    loss = None
    logits = None
    if labels is not None:
        # TODO(PR#678): wrap with OpSlot guard for cross_entropy_loss dispatch
        # (see veomni/models/transformers/qwen3/qwen3_gpu_patch_gen_config.py).
        loss, logits = self.loss_function(
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

    return Qwen3VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )
