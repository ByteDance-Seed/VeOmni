# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
Patch configuration for Qwen3-VL (Transformers v5.x) to mirror VeOmni runtime patches.

Regen (recommended):
  python -m veomni.patchgen.run_codegen \
    veomni.models.transformers.qwen3_vl.qwen3_vl_patch_gen_config \
    --diff -v \
    -o veomni/models/transformers/qwen3_vl/generated

Source module:
  transformers.models.qwen3_vl.modeling_qwen3_vl
"""

from __future__ import annotations

import copy
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
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
from veomni.models.transformers.attention_utils import VARLEN_ATTENTION_TYPES
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.device import IS_NPU_AVAILABLE


# NOTE:
# - Names like ALL_ATTENTION_FUNCTIONS, eager_attention_forward, apply_rotary_pos_emb,
#   apply_rotary_pos_emb_vision, is_flash_attention_requested, DynamicCache, create_causal_mask,
#   BaseModelOutputWithPast, BaseModelOutputWithDeepstackFeatures, Qwen3VLModelOutputWithPast,
#   Qwen3VLCausalLMOutputWithPast, Qwen3VLModel, Qwen3VLForConditionalGeneration are expected to
#   exist in the source module and thus in the generated file.


config = PatchConfig(
    source_module="transformers.models.qwen3_vl.modeling_qwen3_vl",
    target_file="patched_modeling_qwen3_vl.py",
    description="Qwen3-VL patches for VeOmni (GPU/SP/Async-Ulysses), ported to Transformers v5.x",
)

# Ensure generated file has all external imports used by patches.
config.add_import("numpy", alias="np", is_from_import=False)
config.add_import("torch.distributed", alias="dist", is_from_import=False)
config.add_import("copy", is_from_import=False)
config.add_import("functools", names=["partial"])
config.add_import("types", names=["SimpleNamespace"])
config.add_import("veomni.utils.constants", names=["IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"])
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=[
        "gather_heads_scatter_seq",
        "gather_outputs",
        "gather_seq_scatter_heads",
        "get_ulysses_sequence_parallel_world_size",
        "sp_pad_and_slice",
    ],
)
config.add_import(
    "veomni.distributed.sequence_parallel.async_ulysses",
    names=["async_ulysses_output_projection", "async_ulysses_qkv_projection"],
)
config.add_import("veomni.models.transformers.attention_utils", names=["VARLEN_ATTENTION_TYPES"])
config.add_import("veomni.utils.device", names=["IS_NPU_AVAILABLE"])


# ================================================================
# Patch: Qwen3VLVisionAttention.forward
# 1) add flash_attention_3/4 & VeOmni varlen impl support
# 2) accept precomputed max_seqlen to avoid per-layer CPU-GPU sync
# ================================================================
@config.override_method(
    "Qwen3VLVisionAttention.forward",
    description="Support FA3/FA4/VeOmni varlen and pass precomputed max_seqlen.",
)
def qwen3vl_vision_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    max_seqlen: int | None = None,
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

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    if (self.config._attn_implementation in VARLEN_ATTENTION_TYPES) or is_flash_attention_requested(self.config):
        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
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
    return self.proj(attn_output)


# ================================================================
# Patch: Qwen3VLTextAttention
# 1) add async Ulysses attention forward method
# ================================================================
@config.override_method(
    "Qwen3VLTextAttention._async_ulysses_attention_forward",
    description="Inject async-Ulysses attention forward for VeOmni sequence parallelism.",
)
def qwen3vl_text_attention_async_ulysses_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if self.config._attn_implementation not in VARLEN_ATTENTION_TYPES:
        raise ValueError(
            "Async Ulysses attention only supports flash attention implementations. "
            f"Current implementation: '{self.config._attn_implementation}'."
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

    # TODO: check this, wehther we need to unpad sp padding?
    cos = gather_outputs(cos, dim=0, group=get_parallel_state().sp_group)
    sin = gather_outputs(sin, dim=1, group=get_parallel_state().sp_group)

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


@config.override_method(
    "Qwen3VLTextAttention.forward",
    description="Enable async-Ulysses attention when VeOmni async is on; otherwise keep HF v5 logic.",
)
def qwen3vl_text_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if get_parallel_state().async_enabled:
        return self._async_ulysses_attention_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )

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

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
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
    return self.o_proj(attn_output), attn_weights


# ================================================================
# Patch: Qwen3VLTextModel.forward
# Restore v4 behavior for position_ids/text_position_ids (loss parity).
# ================================================================
@config.override_method(
    "Qwen3VLTextModel.forward",
    description="Restore v4 position-id handling for loss parity: text_position_ids = position_ids[0] for 3D positions.",
)
def qwen3vl_text_model_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    visual_pos_masks: torch.Tensor | None = None,
    deepstack_visual_embeds: list[torch.Tensor] | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    attention_mask = create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for layer_idx, decoder_layer in enumerate(self.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
            )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


# ================================================================
# Patch: Qwen3VLVisionModel.rot_pos_emb (+ cache)
# ================================================================
@config.override_method(
    "Qwen3VLVisionModel.rot_pos_emb",
    description="Cache rot_pos_ids(h,w,merge) for performance; numpy-based implementation mirrors VeOmni runtime patch.",
)
def qwen3vl_vision_rot_pos_emb_patched(self, grid_thw: torch.Tensor) -> torch.Tensor:
    merge_size = self.spatial_merge_size

    cache = getattr(self, "_veomni_rot_pos_ids_cache", None)
    if cache is None:
        cache = {}
        setattr(self, "_veomni_rot_pos_ids_cache", cache)

    max_hw = int(grid_thw[:, 1:].max().item())
    freq_table = self.rotary_pos_emb(max_hw)
    device = freq_table.device

    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in grid_thw:
        h = int(height.item())
        w = int(width.item())
        key = (h, w, int(merge_size))
        coords = cache.get(key)
        if coords is None:
            hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
            h_div = h // merge_size
            w_div = w // merge_size
            hpos_ids = hpos_ids.reshape(h_div, merge_size, w_div, merge_size).transpose(0, 2, 1, 3).flatten()

            wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
            wpos_ids = wpos_ids.reshape(h_div, merge_size, w_div, merge_size).transpose(0, 2, 1, 3).flatten()

            coords = torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1)).to(torch.long)
            cache[key] = coords

        coords = coords.to(device)
        if num_frames > 1:
            coords = coords.repeat(int(num_frames.item()), 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    return freq_table[pos_ids].flatten(1)


# ================================================================
# Patch: Qwen3VLVisionModel.fast_pos_embed_interpolate
# ================================================================
@config.override_method(
    "Qwen3VLVisionModel.fast_pos_embed_interpolate",
    description="Efficient fast_pos_embed_interpolate implementation (VeOmni/vLLM-style).",
)
def qwen3vl_vision_fast_pos_embed_interpolate_patched(self, grid_thw: torch.Tensor) -> torch.Tensor:
    num_grid_per_side = self.num_grid_per_side
    m_size = self.spatial_merge_size
    hidden_dim = self.pos_embed.embedding_dim

    outputs = []
    dtype = self.pos_embed.weight.dtype
    for t, h, w in grid_thw:
        h = int(h.item())
        w = int(w.item())
        t_int = int(t.item())

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

        w11 = dh_grid * dw_grid
        w10 = dh_grid - w11
        w01 = dw_grid - w11
        w00 = 1 - dh_grid - w01

        h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
        w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
        h_grid_idx = h_grid * num_grid_per_side

        indices = (h_grid_idx + w_grid).reshape(4, -1)
        weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1).to(dtype=dtype)

        embeds = self.pos_embed(indices) * weights
        combined = embeds[0] + embeds[1] + embeds[2] + embeds[3]
        combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)

        combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
        outputs.append(combined.expand(t_int, -1, -1).reshape(-1, hidden_dim))

    return torch.cat(outputs, dim=0)


# ================================================================
# Patch: Qwen3VLVisionModel.forward
# ================================================================
@config.override_method(
    "Qwen3VLVisionModel.forward",
    description="SP-aware vision forward + precompute max_seqlen; matches VeOmni runtime patch behavior on v5 output type.",
)
def qwen3vl_vision_model_forward_patched(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs: Unpack[TransformersKwargs],
):
    hidden_states = self.patch_embed(hidden_states)

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    if get_parallel_state().sp_enabled:
        pos_embeds = sp_pad_and_slice(pos_embeds, dim=0, pad_value=0, pad_scale=4)
    hidden_states = hidden_states + pos_embeds

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)

    total_seq_len = int(cu_seqlens[-1].item())
    rotary_pos_emb = rotary_pos_emb.reshape(total_seq_len, -1)

    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    if get_parallel_state().sp_enabled:
        cos, sin = position_embeddings
        cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=4)
        sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=4)
        position_embeddings = (cos, sin)

        sp_size = get_parallel_state().sp_size
        pad_seq_len = seq_len * sp_size - total_seq_len
        if pad_seq_len > 0:
            cu_seqlens = torch.cat([cu_seqlens, (cu_seqlens[-1] + pad_seq_len).unsqueeze(0)], dim=0)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()
    if IS_NPU_AVAILABLE:
        cu_seqlens = cu_seqlens.cpu()

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature_lists.append(
                self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](hidden_states)
            )

    merged_hidden_states = self.merger(hidden_states)
    return BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
        deepstack_features=deepstack_feature_lists,
    )


# ================================================================
# Patch: Qwen3VLVisionModel.dummy_forward
# ================================================================
@config.override_method(
    "Qwen3VLVisionModel.dummy_forward",
    description="Dummy vision forward to avoid FSDP hang when some ranks have no pixel_values.",
)
def qwen3vl_vision_model_dummy_forward_patched(self):
    if get_parallel_state().sp_enabled:
        sp_size = get_parallel_state().sp_size
        pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
        grid_thw = torch.tensor([[1, 4 * sp_size, 4]], dtype=torch.int32, device=self.device)
    else:
        pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32, device=self.device)
    return self(hidden_states=pixel_values, grid_thw=grid_thw, return_dict=True)


# ================================================================
# Patch: Qwen3VLTextModel._deepstack_process
# ================================================================
@config.override_method(
    "Qwen3VLTextModel._deepstack_process",
    description="If visual_pos_masks is None, still trigger GPU ops but add 0.0 (VeOmni).",
)
def qwen3vl_text_model_deepstack_process_patched(
    self,
    hidden_states: torch.Tensor,
    visual_pos_masks: torch.Tensor | None,
    visual_embeds: torch.Tensor,
):
    if visual_pos_masks is None:
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        return hidden_states + visual_embeds.mean() * 0.0

    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
    hidden_states[visual_pos_masks, :] = local_this
    return hidden_states


# ================================================================
# Patch: Qwen3VLModel.get_image_features
# ================================================================
@config.override_method(
    "Qwen3VLModel.get_image_features",
    description="Skip torch.split in get_image_features: keep pooler_output as one tensor (VeOmni).",
)
def qwen3vl_model_get_image_features_patched(
    self,
    pixel_values: torch.FloatTensor,
    image_grid_thw: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
):
    pixel_values = pixel_values.type(self.visual.dtype)
    return self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)


# ================================================================
# Patch: Qwen3VLModel.get_placeholder_mask
# ================================================================
@config.override_method(
    "Qwen3VLModel.get_placeholder_mask",
    description="Relax placeholder mask checks for VeOmni (no strict token/feature length assertion).",
)
def qwen3vl_model_get_placeholder_mask_patched(
    self,
    input_ids: torch.LongTensor,
    inputs_embeds: torch.FloatTensor,
    image_features: torch.FloatTensor | None = None,
    video_features: torch.FloatTensor | None = None,
):
    if input_ids is None:
        special_image_mask = inputs_embeds == self.get_input_embeddings()(
            torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        special_image_mask = special_image_mask.all(-1)
        special_video_mask = inputs_embeds == self.get_input_embeddings()(
            torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        special_video_mask = special_video_mask.all(-1)
    else:
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id

    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    return special_image_mask, special_video_mask


# ================================================================
# Patch: Qwen3VLModel.get_rope_index
# (恢复 v4.57.3 行为以保证 loss parity)
# ================================================================
@config.override_method(
    "Qwen3VLModel.get_rope_index",
    description="Restore v4.57.3 get_rope_index behavior for loss parity (signature adjusted for v5).",
)
def qwen3vl_model_get_rope_index_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, current_input_ids in enumerate(total_input_ids):
            current_input_ids = current_input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(current_input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = current_input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()

            input_tokens = current_input_ids.tolist()
            llm_pos_ids_list: list[torch.Tensor] = []
            st = 0
            remain_images, remain_videos = int(image_nums.item()), int(video_nums.item())

            for _ in range(remain_images + remain_videos):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size

                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                )

                t_index = (
                    torch.arange(llm_grid_t, device=input_ids.device)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h, device=input_ids.device)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w, device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)

                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)

    return position_ids, mrope_position_deltas


# ================================================================
# Patch: Qwen3VLModel.forward
# ================================================================
@config.override_method(
    "Qwen3VLModel.forward",
    description="VeOmni SP/async/FSDP-dummy + flash-attn kwargs handling; keep v4 position-id semantics for loss parity.",
)
def qwen3vl_model_forward_patched(
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
    mm_token_type_ids: torch.IntTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # Pop VeOmni-only kwargs early to avoid leaking into HF attention backends.
    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)

    # if None, calculate mask
    if video_mask is None and image_mask is None:
        if get_parallel_state().sp_enabled:
            input_ids_list = [torch.zeros_like(input_ids) for i in range(get_parallel_state().sp_size)]
            dist.all_gather(input_ids_list, input_ids, group=get_parallel_state().sp_group)
            input_ids = torch.cat(input_ids_list, dim=0)
        image_mask, video_mask = self.get_placeholder_mask(input_ids)

    # Pop flash-attn kwargs for ViT; they should go only to the language model.
    flash_attn_kwargs = {}
    for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
        if key in kwargs:
            flash_attn_kwargs[key] = kwargs.pop(key)

    if get_parallel_state().sp_enabled:
        inputs_embeds = gather_seq_scatter_heads(
            inputs_embeds,
            seq_dim=1,
            head_dim=2,
            group=get_parallel_state().sp_group,
        )

    fake_deepstack = None

    # Image branch
    if pixel_values is not None:
        image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        image_embeds = image_outputs.pooler_output
        deepstack_image_embeds = image_outputs.deepstack_features

        if get_parallel_state().sp_enabled:
            image_embeds = gather_seq_scatter_heads(
                image_embeds,
                seq_dim=0,
                head_dim=-1,
                group=get_parallel_state().sp_group,
            )
            deepstack_image_embeds = [
                gather_outputs(embed, gather_dim=0, group=get_parallel_state().sp_group)
                for embed in deepstack_image_embeds
            ]

        n_image_tokens = int(image_mask.sum().long().item())
        embeds_image_mask = (
            image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )

        image_embeds = image_embeds[:n_image_tokens]
        deepstack_image_embeds = [embed[:n_image_tokens] for embed in deepstack_image_embeds]
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_image_mask, image_embeds)

        if get_parallel_state().sp_enabled:
            seq_len = image_mask.shape[1]
            seq_per_rank = seq_len // get_parallel_state().sp_size
            rank_start = get_parallel_state().sp_rank * seq_per_rank
            rank_end = rank_start + seq_per_rank

            deepstack_offset = int(image_mask[:, :rank_start].sum().item())
            image_mask = image_mask[:, rank_start:rank_end]
            deepstack_len = int(image_mask.sum().item())
            deepstack_image_embeds = [
                embed[deepstack_offset : deepstack_offset + deepstack_len] for embed in deepstack_image_embeds
            ]

    elif get_parallel_state().fsdp_enabled:
        fake_out = self.visual.dummy_forward()
        fake_embeds = fake_out.pooler_output.mean() * 0.0
        inputs_embeds = inputs_embeds + fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        fake_deepstack = fake_out.deepstack_features

    # Video branch
    if pixel_values_videos is not None:
        video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
        video_embeds = video_outputs.pooler_output
        deepstack_video_embeds = video_outputs.deepstack_features

        if get_parallel_state().sp_enabled:
            video_embeds = gather_seq_scatter_heads(
                video_embeds,
                seq_dim=0,
                head_dim=-1,
                group=get_parallel_state().sp_group,
            )
            deepstack_video_embeds = [
                gather_outputs(embed, gather_dim=0, group=get_parallel_state().sp_group)
                for embed in deepstack_video_embeds
            ]

        n_video_tokens = int(video_mask.sum().long().item())
        embeds_video_mask = (
            video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )

        video_embeds = video_embeds[:n_video_tokens]
        deepstack_video_embeds = [embed[:n_video_tokens] for embed in deepstack_video_embeds]
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_video_mask, video_embeds)

        if get_parallel_state().sp_enabled:
            seq_len = video_mask.shape[1]
            seq_per_rank = seq_len // get_parallel_state().sp_size
            rank_start = get_parallel_state().sp_rank * seq_per_rank
            rank_end = rank_start + seq_per_rank

            deepstack_offset = int(video_mask[:, :rank_start].sum().item())
            video_mask = video_mask[:, rank_start:rank_end]
            deepstack_len = int(video_mask.sum().item())
            deepstack_video_embeds = [
                embed[deepstack_offset : deepstack_offset + deepstack_len] for embed in deepstack_video_embeds
            ]

    elif get_parallel_state().fsdp_enabled:
        fake_out = self.visual.dummy_forward()
        fake_embeds = fake_out.pooler_output.mean() * 0.0
        inputs_embeds = inputs_embeds + fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        fake_deepstack = fake_out.deepstack_features

    if get_parallel_state().sp_enabled:
        inputs_embeds = gather_heads_scatter_seq(
            inputs_embeds,
            head_dim=2,
            seq_dim=1,
            group=get_parallel_state().sp_group,
        )

    # Build deepstack inputs
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
        deepstack_visual_embeds = fake_deepstack

    # Position ids (v4 semantics, non-compiled path)
    if position_ids is None and input_ids is not None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask.get("full_attention", None)
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_stage = (cache_position is not None and cache_position[0] == 0) or (
            past_key_values is None or past_key_values.get_seq_length() == 0
        )
        if prefill_stage or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
            )
            pos_1d = torch.arange(seq_length, device=inputs_embeds.device).view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            pos_1d = pos_1d.add(delta)
            position_ids = pos_1d.unsqueeze(0).expand(3, -1, -1)
    elif position_ids is not None:
        # handle (bs, 3, seq) precomputed position_ids
        if position_ids.dim() == 3 and position_ids.shape[1] == 3:
            position_ids = position_ids.transpose(0, 1).contiguous()

    # Restore flash-attn kwargs to LM
    kwargs.update(flash_attn_kwargs)

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
        rope_deltas=self.rope_deltas,
    )


# ================================================================
# Patch: Qwen3VLForConditionalGeneration.get_position_id_func
# ================================================================
@config.override_method(
    "Qwen3VLForConditionalGeneration._get_position_id",
    description="Pickle-friendly helper used by get_position_id_func (class function style).",
)
def qwen3vl_get_position_id_static(main_func, fake_model, **kwargs):
    position_ids, rope_deltas = main_func(fake_model, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}


@config.override_method(
    "Qwen3VLForConditionalGeneration.get_position_id_func",
    description="Expose multiprocessing-safe function to compute position_ids/rope_deltas using VeOmni placeholder token ids.",
)
def qwen3vl_get_position_id_func_patched(self):
    fake_config = copy.copy(self.config)
    fake_config.image_token_id = IMAGE_INPUT_INDEX
    fake_config.video_token_id = VIDEO_INPUT_INDEX
    fake_model = SimpleNamespace(config=fake_config)
    return partial(Qwen3VLForConditionalGeneration._get_position_id, Qwen3VLModel.get_rope_index, fake_model)


# ================================================================
# Patch: Qwen3VLForConditionalGeneration.forward (fused/unified loss)
# ================================================================
@config.override_method(
    "Qwen3VLForConditionalGeneration.forward",
    description="Use VeOmni fused/unified loss path (pass hidden_states + weights) for loss parity.",
)
def qwen3vl_for_conditional_generation_forward_patched(
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
    mm_token_type_ids: torch.IntTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
):
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
        mm_token_type_ids=mm_token_type_ids,
        **kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None
    if labels is not None:
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

    return Qwen3VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        rope_deltas=outputs.rope_deltas,
    )
