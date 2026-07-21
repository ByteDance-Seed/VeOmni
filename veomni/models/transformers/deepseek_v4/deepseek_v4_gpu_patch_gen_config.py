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
Patch configuration for DeepseekV4 GPU patched modeling generation.

Regen command:
patchgen veomni.models.transformers.deepseek_v4.deepseek_v4_gpu_patch_gen_config -o veomni/models/transformers/deepseek_v4/generated --diff

Patches:
1. ``DeepseekV4Indexer.forward`` — optional TileLang Lightning Indexer for
   canonical CUDA prefill/training positions, selected by
   ``dsa_indexer_implementation=tilelang`` with eager cache/decode fallback.
2. ``eager_attention_forward`` — optional TileLang sparse MQA attention,
   selected by ``dsa_attention_implementation=tilelang``. Converts the
   upstream additive sliding/compressor mask into compact indices.
3. ``DeepseekV4Experts`` — drops upstream ``@use_experts_implementation``
   (which would otherwise dispatch to ``grouped_mm`` and bypass VeOmni's
   fused MoE kernel). Keeps the v5 stacked ``gate_up_proj [E, 2*I, H]`` /
   ``down_proj [E, H, I]`` layout and the gpt-oss-style ``swiglu_limit``
   clamp. Dispatch is OpSlot-guarded (``veomni_moe_experts_forward``):
   non-eager -> ``fused_moe_forward``; eager -> per-expert loop.
4. ``DeepseekV4RMSNorm`` / ``DeepseekV4UnweightedRMSNorm`` — functional
   OpSlot dispatch to Liger RMSNorm while preserving the two distinct class
   layouts. The unweighted form passes ``weight=None`` to Liger's supported
   non-affine path; eager keeps the official weighted FP32 multiply order.
5. ``DeepseekV4MLP.forward`` — shared experts always apply the official
   ``swiglu_limit`` clamp, then optionally fuse silu*mul via Liger when the
   SwiGLU OpSlot is non-eager. Routed experts remain on the fused-MoE path
   above so their clamp is kept there too.
6. ``DeepseekV4ForCausalLM.forward`` — OpSlot guard for fused
   cross-entropy (``veomni_causal_lm_loss``) + ``MoeCausalLMOutputWithLogProbs``
   so callers can read per-token log-probs / entropy alongside the loss.
7. ``DeepseekV4RotaryEmbedding.forward`` — retains FP32 cos/sin tables for
   official-compatible inference and casts them to the activation dtype during
   training so FSDP activation-checkpoint recomputation sees stable metadata.
8. ``DeepseekV4HyperConnections.pre`` / ``DeepseekV4HyperConnections.post`` /
   ``DeepseekV4HyperConnections.head`` — optional TileKernels mHC dispatch
   selected by ``mhc_implementation=tilelang``.
9. ``DeepseekV4Attention.forward`` — matches the official BF16 per-head Q
   normalization before RoPE, and adds Ulysses SP (Q head all-to-all + MQA
   sequence all-gather around compressors / sparse attention).
10. ``DeepseekV4TopKRouter.forward`` / ``DeepseekV4HashRouter.forward`` —
   always perform the official FP32 router projection.
11. Register ``get_parallel_plan`` on ``DeepseekV4ForCausalLM``.

Intentionally NOT patched:

- Class-level ``LigerSwiGLUMLP`` / ``LigerRMSNorm`` replacement — keep the
  native V4 constructors (unweighted norm layout and shared-expert
  ``moe_intermediate_size`` mapping) and only swap arithmetic through OpSlots.
- ``apply_rotary_pos_emb`` — DeepSeek-V4 uses a *partial* RoPE (the
  trailing ``qk_rope_head_dim`` slice only, with the leading nope channels
  untouched) plus an interleaved ``repeat_interleave(2)`` cos/sin layout
  that ``liger_rotary_pos_emb`` does not implement. SKILL.md flags this
  exact case (partial_rotary -> liger NaN).
- ``DeepseekV4Attention.forward`` remains eager/TileLang-only
  (``_supports_flash_attn = False`` / ``_supports_sdpa = False`` /
  ``_supports_flex_attn = False``: ``head_dim=512`` exceeds FlashAttention's
  256 cap, SDPA lacks the per-head learnable sink, and FlexAttention can't
  resize BlockMask after the in-block compressor concatenation). MoE does not
  share this limitation: the experts patch above binds VeOmni fused MoE by
  default on GPU. Ulysses SP is handled inside the patched attention forward
  (head all-to-all on Q + sequence all-gather on MQA KV / compressor inputs)
  rather than via FA's ``veomni_flash_attention_*_with_sp`` path.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4HCACache,
    apply_rotary_pos_emb,
    load_balancing_loss_func,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.models.transformers.deepseek_v4.packed_utils import (
    build_packed_compression_metadata,
    compress_packed_windows,
    isolate_packed_causal_mask_,
    packed_compressed_block_bias,
    packed_compressed_causal_ranges,
)
from veomni.ops import fused_moe_forward
from veomni.ops.dispatch import OpsConfigSlot, OpSlot
from veomni.ops.kernels.deepseek_v4 import sparse_attn_tilelang, v4_lighting_indexer
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import MoeCausalLMOutputWithLogProbs


# OpSlot declarations — mirrored into the generated module via
# ``add_post_import_block`` below. The duplicate at module scope here is
# only for IDE/type-check friendliness while authoring this file; the
# runtime slots used by the generated modeling are bound at model-build
# time by ``_bind_veomni_ops()`` in ``veomni/models/auto.py``.
veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
veomni_rms_norm = OpSlot("rms_norm", "standard")
veomni_unweighted_rms_norm = OpSlot("rms_norm", "unweighted")
veomni_swiglu_mlp = OpSlot("swiglu_mlp", "standard")
veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
veomni_load_balancing_loss = OpSlot("load_balancing_loss", "standard")
veomni_mhc_pre = OpSlot("mhc", "pre")
veomni_mhc_post = OpSlot("mhc", "post")
veomni_mhc_head = OpSlot("mhc", "head")
veomni_dsa_indexer_implementation = OpsConfigSlot("dsa_indexer_implementation")
veomni_dsa_attention_implementation = OpsConfigSlot("dsa_attention_implementation")

# Names resolved at codegen time from generated imports.
get_parallel_state = None
gather_seq_scatter_heads = None
gather_heads_scatter_seq = None
gather_outputs = None


config = PatchConfig(
    source_module="transformers.models.deepseek_v4.modeling_deepseek_v4",
    target_file="patched_modeling_deepseek_v4_gpu.py",
    description="DeepseekV4 with VeOmni fused-MoE + OpSlot-guarded fused-CE patches",
)

config.add_import("veomni.ops", names=["fused_moe_forward"])
config.add_import(
    "veomni.ops.kernels.deepseek_v4",
    names=["sparse_attn_tilelang", "v4_lighting_indexer"],
)
config.add_import(
    "veomni.distributed.parallel_state",
    names=["get_parallel_state"],
)
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=["gather_heads_scatter_seq", "gather_outputs", "gather_seq_scatter_heads"],
)
config.add_import(
    "veomni.models.transformers.deepseek_v4.packed_utils",
    names=[
        "build_packed_compression_metadata",
        "compress_packed_windows",
        "isolate_packed_causal_mask_",
        "packed_compressed_block_bias",
        "packed_compressed_causal_ranges",
    ],
)

# Surface ``MoeCausalLMOutputWithLogProbs`` so the patched ``forward`` can
# return per-token log-probs / entropy as constructor fields. Mutating
# ``output.log_probs`` / ``output.entropy`` after constructing
# ``MoeCausalLMOutputWithPast`` would bypass ModelOutput pytree flattening,
# breaking FSDP2's pre-backward unshard hook on ``lm_head`` (parallels
# the qwen3_5_moe / qwen3_moe fix).
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "MoeCausalLMOutputWithLogProbs"],
)
config.drop_import_names("MoeCausalLMOutputWithPast")

config.add_post_import_block(
    """
    from veomni.ops.dispatch import OpSlot, OpsConfigSlot
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_rms_norm = OpSlot("rms_norm", "standard")
    veomni_unweighted_rms_norm = OpSlot("rms_norm", "unweighted")
    veomni_swiglu_mlp = OpSlot("swiglu_mlp", "standard")
    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    veomni_load_balancing_loss = OpSlot("load_balancing_loss", "standard")
    veomni_mhc_pre = OpSlot("mhc", "pre")
    veomni_mhc_post = OpSlot("mhc", "post")
    veomni_mhc_head = OpSlot("mhc", "head")
    veomni_dsa_indexer_implementation = OpsConfigSlot("dsa_indexer_implementation")
    veomni_dsa_attention_implementation = OpsConfigSlot("dsa_attention_implementation")
    """
)


# ================================================================
# Patch: DeepSeek V4 RMSNorm dispatch
# ================================================================
@config.override_method(
    "DeepseekV4RMSNorm.forward",
    description="OpSlot guard for Liger fused weighted RMSNorm with official eager FP32 fallback",
)
def deepseek_v4_rms_norm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if veomni_rms_norm.use_non_eager_impl:
        return veomni_rms_norm(hidden_states, self.weight, self.variance_epsilon)

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return (self.weight.float() * hidden_states).to(input_dtype)


@config.override_method(
    "DeepseekV4UnweightedRMSNorm.forward",
    description="OpSlot guard for Liger fused unweighted RMSNorm",
)
def deepseek_v4_unweighted_rmsnorm_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    if veomni_unweighted_rms_norm.use_non_eager_impl:
        return veomni_unweighted_rms_norm(x, None, self.eps)

    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


# ================================================================
# Patch: official RoPE table precision and checkpoint-stable training dtype
# ================================================================
@config.override_method(
    "DeepseekV4RotaryEmbedding.forward",
    description="Retain FP32 cos/sin for inference and use activation dtype for checkpoint-stable training",
)
def deepseek_v4_rotary_embedding_forward_patched(self, x, position_ids, layer_type=None):
    inv_freq = getattr(self, f"{layer_type}_inv_freq")
    attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with maybe_autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        cos = freqs.cos() * attention_scaling
        sin = freqs.sin() * attention_scaling
    if self.training:
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    return cos, sin


# ================================================================
# Patch: TileKernels mHC dispatch
# ================================================================
@config.override_method(
    "DeepseekV4HyperConnection.forward",
    description="Dispatch DeepSeek V4 mHC pre/Sinkhorn/collapse through an OpSlot",
)
def deepseek_v4_hyper_connection_forward_patched(
    self,
    hidden_streams: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if veomni_mhc_pre.use_non_eager_impl:
        return veomni_mhc_pre(
            hidden_streams,
            self.fn,
            self.scale,
            self.base,
            self.input_norm.eps,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )

    hc = self.hc_mult
    flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
    pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = self.scale.unbind(0)
    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
    for _ in range(self.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
    collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
    return post, comb, collapsed


@config.override_method(
    "DeepseekV4HyperHead.forward",
    description="Dispatch the final DeepSeek V4 mHC collapse through an OpSlot",
)
def deepseek_v4_hyper_head_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    if veomni_mhc_head.use_non_eager_impl:
        return veomni_mhc_head(
            x,
            self.hc_fn,
            self.hc_scale,
            self.hc_base,
            self.input_norm.eps,
            self.hc_mult,
            self.eps,
        )

    flat = self.input_norm(x.flatten(2).float())
    mixes = F.linear(flat, self.hc_fn.float())
    pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
    return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


@config.override_method(
    "DeepseekV4DecoderLayer.forward",
    description="Dispatch DeepSeek V4 mHC residual post-mixing through an OpSlot",
)
def deepseek_v4_decoder_layer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> torch.Tensor:
    dtype = hidden_states.dtype
    post, comb, collapsed = self.attn_hc(hidden_states)
    attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
    if veomni_mhc_post.use_non_eager_impl:
        hidden_states = veomni_mhc_post(attn_output, hidden_states, post, comb)
    else:
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

    post, comb, collapsed = self.ffn_hc(hidden_states)
    mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
    if veomni_mhc_post.use_non_eager_impl:
        return veomni_mhc_post(mlp_output, hidden_states, post, comb)
    return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), hidden_states
    )


# ================================================================
# Patch: packed compressed-attention windows
# 1. Keep every HCA/CSA compression window within one packed sequence.
# 2. Reset compressed RoPE positions and causal ranges at each boundary.
# ================================================================
@config.override_method(
    "DeepseekV4HCACompressor.forward",
    description="Keep HCA compression local to packed sequences",
)
def deepseek_v4_hca_compressor_forward_patched(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: Cache | None,
    layer_idx: int,
    packed_sequence_slices: tuple[tuple[int, int], ...] | None = None,
    packed_compression_metadata: dict[int, dict[str, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (packed_sequence_slices is None) != (packed_compression_metadata is None):
        raise ValueError("Packed sequence slices and compression metadata must be provided together")
    batch, _, _ = hidden_states.shape
    cache_layer: DeepseekV4HCACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if cache_layer is None and packed_sequence_slices is not None and packed_compression_metadata is not None:
        rate_metadata = packed_compression_metadata[self.compress_rate]
        compressed = compress_packed_windows(
            kv,
            gate,
            self.position_bias,
            self.head_dim,
            self.compress_rate,
            self.kv_norm,
            self.rotary_emb,
            self.rope_layer_type,
            position_ids,
            rate_metadata,
            overlap=False,
        )
        block_bias = packed_compressed_block_bias(rate_metadata)
        return compressed.unsqueeze(1), block_bias

    if cache_layer is None:
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
        chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)

    if chunk_kv.shape[1] > 0:
        n_windows = chunk_kv.shape[1] // self.compress_rate
        chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
        chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(
            chunk_gate.dtype
        )
        compressed = self.kv_norm(
            (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2)
        )
        positions = torch.arange(n_windows, device=compressed.device)
        positions = (positions * self.compress_rate + first_window_position).unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
        compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

    if cache_layer is not None:
        compressed = cache_layer.update_compressor_states("compressor", compressed)
    compressed_kv = compressed.unsqueeze(1)

    compressed_len = compressed_kv.shape[2]
    seq_len = position_ids.shape[1]
    if seq_len == 1 or compressed_len == 0:
        return compressed_kv, None

    entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
    causal_threshold = (position_ids + 1) // self.compress_rate
    block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
    block_bias = block_bias.masked_fill(
        entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
        float("-inf"),
    )
    return compressed_kv, block_bias


@config.override_method(
    "DeepseekV4CSACompressor.forward",
    description="Keep CSA compression and indexing local to packed sequences",
)
def deepseek_v4_csa_compressor_forward_patched(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: Cache | None,
    layer_idx: int,
    packed_sequence_slices: tuple[tuple[int, int], ...] | None = None,
    packed_compression_metadata: dict[int, dict[str, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (packed_sequence_slices is None) != (packed_compression_metadata is None):
        raise ValueError("Packed sequence slices and compression metadata must be provided together")
    batch, seq_len, _ = hidden_states.shape
    cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if cache_layer is None and packed_sequence_slices is not None and packed_compression_metadata is not None:
        rate_metadata = packed_compression_metadata[self.compress_rate]
        compressed = compress_packed_windows(
            kv,
            gate,
            self.position_bias,
            self.head_dim,
            self.compress_rate,
            self.kv_norm,
            self.rotary_emb,
            self.rope_layer_type,
            position_ids,
            rate_metadata,
            overlap=True,
        )
        compressed_kv = compressed.unsqueeze(1)
        top_k_indices = self.indexer(
            hidden_states,
            q_residual,
            position_ids,
            past_key_values,
            layer_idx,
            packed_sequence_slices=packed_sequence_slices,
            packed_compression_metadata=packed_compression_metadata,
        )
        compressed_len = compressed_kv.shape[2]
        valid = top_k_indices >= 0
        safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
        block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
        block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
        return compressed_kv, block_bias[..., :compressed_len]

    if cache_layer is None:
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
        chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)

    if chunk_kv.shape[1] > 0:
        n_windows = chunk_kv.shape[1] // self.compress_rate
        ratio = self.compress_rate
        chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
        chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)
        new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
        new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
        new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
        new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
        if n_windows > 1:
            new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
            new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
        if cache_layer is not None:
            prior_kv, prior_gate = cache_layer.update_overlap_state("compressor", chunk_kv, chunk_gate, self.head_dim)
            if prior_kv is not None:
                new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)
        compressed = self.kv_norm((new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2))
        positions = torch.arange(n_windows, device=compressed.device)
        positions = positions * self.compress_rate + first_window_position
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
        compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

    if cache_layer is not None:
        compressed = cache_layer.update_compressor_states("compressor", compressed)
    compressed_kv = compressed.unsqueeze(1)
    top_k_indices = self.indexer(hidden_states, q_residual, position_ids, past_key_values, layer_idx)
    compressed_len = compressed_kv.shape[2]
    valid = top_k_indices >= 0
    safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
    block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
    block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
    return compressed_kv, block_bias[..., :compressed_len]


# ================================================================
# Patch: DeepseekV4Indexer.forward
# 1. Dispatch CUDA prefill/training index scoring to the TileLang Lightning
#    Indexer when ``dsa_indexer_implementation=tilelang``. Cache/decode and unusual
#    position layouts retain the upstream eager implementation.
# ================================================================
@config.override_method("DeepseekV4Indexer.forward", description="Optional TileLang Lightning Indexer dispatch")
def deepseek_v4_indexer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: Cache | None,
    layer_idx: int,
    packed_sequence_slices: tuple[tuple[int, int], ...] | None = None,
    packed_compression_metadata: dict[int, dict[str, torch.Tensor]] | None = None,
) -> torch.LongTensor:
    if (packed_sequence_slices is None) != (packed_compression_metadata is None):
        raise ValueError("Packed sequence slices and compression metadata must be provided together")
    batch, seq_len, _ = hidden_states.shape
    cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if cache_layer is None and packed_sequence_slices is not None and packed_compression_metadata is not None:
        rate_metadata = packed_compression_metadata[self.compress_rate]
        compressed = compress_packed_windows(
            kv,
            gate,
            self.position_bias,
            self.head_dim,
            self.compress_rate,
            self.kv_norm,
            self.rotary_emb,
            self.rope_layer_type,
            position_ids,
            rate_metadata,
            overlap=True,
        )
        chunk_kv = chunk_gate = None
        first_window_position = 0
    elif cache_layer is None:
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
        chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("indexer", kv, gate)

    if packed_compression_metadata is not None and cache_layer is None:
        pass
    elif chunk_kv.shape[1] > 0:
        n_windows = chunk_kv.shape[1] // self.compress_rate
        ratio = self.compress_rate
        chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
        chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

        new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
        new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
        new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
        new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
        if n_windows > 1:
            new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
            new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
        if cache_layer is not None:
            prior_kv, prior_gate = cache_layer.update_overlap_state("indexer", chunk_kv, chunk_gate, self.head_dim)
            if prior_kv is not None:
                new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

        compressed = self.kv_norm((new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2))
        positions = torch.arange(n_windows, device=compressed.device)
        positions = positions * self.compress_rate + first_window_position
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
        compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

    compressed_kv = compressed if cache_layer is None else cache_layer.update_compressor_states("indexer", compressed)

    cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
    q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
    q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)
    weights = self.weights_proj(hidden_states).float() * (self.weights_scaling * self.softmax_scale)
    compressed_len = compressed_kv.shape[1]
    top_k = min(self.index_topk, compressed_len)

    # --- Patch.1 ---
    indexer_implementation = veomni_dsa_indexer_implementation.value
    if indexer_implementation not in {"eager", "tilelang"}:
        raise ValueError(
            "DeepSeek-V4 does not support "
            f"dsa_indexer_implementation={indexer_implementation!r}; expected 'eager' or 'tilelang'"
        )
    canonical_positions = torch.arange(seq_len, device=position_ids.device).unsqueeze(0).expand_as(position_ids)
    packed_ranges = None
    if packed_compression_metadata is not None and cache_layer is None:
        packed_ranges = packed_compressed_causal_ranges(packed_compression_metadata[self.compress_rate])
    use_tilelang = (
        indexer_implementation == "tilelang"
        and hidden_states.is_cuda
        and q.dtype == torch.bfloat16
        and compressed_kv.dtype == torch.bfloat16
        and self.num_heads <= 64
        and self.num_heads % 8 == 0
        and self.head_dim >= 32
        and self.head_dim == 1 << (self.head_dim - 1).bit_length()
        and cache_layer is None
        and compressed_len > 0
        and (packed_ranges is not None or torch.equal(position_ids, canonical_positions))
    )
    if use_tilelang:
        _, top_k_indices = v4_lighting_indexer(
            q.transpose(0, 1).contiguous(),
            compressed_kv.transpose(0, 1).contiguous(),
            weights.transpose(0, 1).contiguous(),
            self.compress_rate,
            top_k,
            cu_seqlen_ks=None if packed_ranges is None else packed_ranges[0],
            cu_seqlen_ke=None if packed_ranges is None else packed_ranges[1],
        )
        return top_k_indices.to(torch.long)
    # --- Patch.1 ---

    scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores) * self.softmax_scale
    eager_weights = self.weights_proj(hidden_states).float() * self.weights_scaling
    index_scores = (scores * eager_weights.unsqueeze(-1)).sum(dim=2)
    if compressed_len > 0:
        entry_indices = torch.arange(compressed_len, device=index_scores.device)
        if packed_ranges is None:
            causal_starts = torch.zeros_like(position_ids)
            causal_ends = (position_ids + 1) // self.compress_rate
        else:
            causal_starts, causal_ends = (value.unsqueeze(0) for value in packed_ranges)
        future_mask = (entry_indices.view(1, 1, -1) < causal_starts.unsqueeze(-1)) | (
            entry_indices.view(1, 1, -1) >= causal_ends.unsqueeze(-1)
        )
        index_scores = index_scores.masked_fill(future_mask, float("-inf"))
        top_k_indices = index_scores.topk(top_k, dim=-1).indices
        invalid = (top_k_indices < causal_starts.unsqueeze(-1)) | (top_k_indices >= causal_ends.unsqueeze(-1))
        return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)
    return index_scores.topk(top_k, dim=-1).indices


# ================================================================
# Patch: DeepseekV4Attention.forward
# 1. Pass the collator-provided packed sequence slices into compressors.
# 2. Ulysses SP: all-to-all Q heads, sequence all-gather for MQA KV and
#    compressor inputs (windows/indexers need the full sequence), then
#    scatter attention outputs back to the local sequence shard.
# ================================================================
@config.override_method(
    "DeepseekV4Attention.forward",
    description="Packed compressor path + Ulysses SP for DeepSeek-V4 eager/TileLang attention",
)
def deepseek_v4_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings[self.rope_layer_type]

    q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_residual).view(*hidden_shape)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.q_b_norm.eps)
    q = q.transpose(1, 2)
    q = apply_rotary_pos_emb(q, cos, sin)

    kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
    kv = apply_rotary_pos_emb(kv, cos, sin)

    if past_key_values is not None:
        kv = past_key_values.update(kv, kv, self.layer_idx)[0]

    ulysses_enabled = get_parallel_state().ulysses_enabled
    compressor_hidden = hidden_states
    compressor_q_residual = q_residual
    compressor_position_ids = position_ids
    s_aux = self.sinks
    if ulysses_enabled:
        if past_key_values is not None:
            raise RuntimeError("DeepSeek-V4 Ulysses SP does not support KV-cache decode")
        ulysses_group = get_parallel_state().ulysses_group
        ulysses_size = get_parallel_state().ulysses_size
        ulysses_rank = get_parallel_state().ulysses_rank
        if self.num_heads % ulysses_size != 0:
            raise ValueError(
                f"DeepSeek-V4 Ulysses SP requires num_attention_heads ({self.num_heads}) "
                f"divisible by ulysses_size ({ulysses_size})"
            )
        local_num_heads = self.num_heads // ulysses_size
        # Compressors / Lightning Indexer window across the full sequence, so
        # gather the local shard before running them. Q uses true Ulysses
        # head/sequence exchange; MQA KV stays single-head and is all-gathered.
        compressor_hidden = gather_outputs(hidden_states, gather_dim=1, group=ulysses_group)
        compressor_q_residual = gather_outputs(q_residual, gather_dim=1, group=ulysses_group)
        compressor_position_ids = gather_outputs(position_ids, gather_dim=-1, group=ulysses_group)
        # Use the same [B, S, H, D] Ulysses layout as FA (seq_dim=1, head_dim=2).
        q = q.transpose(1, 2).contiguous()
        q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, group=ulysses_group)
        q = q.transpose(1, 2).contiguous()
        kv = gather_outputs(kv, gather_dim=2, group=ulysses_group)
        head_start = ulysses_rank * local_num_heads
        s_aux = self.sinks.narrow(0, head_start, local_num_heads).contiguous()

    block_bias = None
    if self.compressor is not None:
        compressed_kv, block_bias = self.compressor(
            compressor_hidden,
            compressor_q_residual,
            compressor_position_ids,
            past_key_values,
            self.layer_idx,
            packed_sequence_slices=kwargs.get("packed_sequence_slices"),
            packed_compression_metadata=kwargs.get("packed_compression_metadata"),
        )
        kv = torch.cat([kv, compressed_kv], dim=2)

    if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
        if block_bias is not None:
            attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
        else:
            attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    kwargs = {key: value for key, value in kwargs.items() if key != "s_aux"}
    attn_output, attn_weights = attention_interface(
        self,
        q,
        kv,
        kv,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        s_aux=s_aux,
        **kwargs,
    )

    if ulysses_enabled:
        # eager/TileLang return [B, S_full, H_local, D]; restore local seq + full heads.
        attn_output = gather_heads_scatter_seq(
            attn_output, head_dim=2, seq_dim=1, group=get_parallel_state().ulysses_group
        )

    attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)
    grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
    grouped = self.o_a_proj(grouped).flatten(2)
    output = self.o_b_proj(grouped)
    return output, attn_weights


# ================================================================
# Patch: eager_attention_forward
# 1. Dispatch DeepSeek-V4 attention to the TileLang sparse MQA kernel when
#    ``dsa_attention_implementation=tilelang``. The existing additive mask is
#    converted to a compact fixed-width index list, preserving sliding-window,
#    compressor, causal, and invalid-index semantics.
# 2. Preserve the upstream eager implementation as the default fallback.
# ================================================================
@config.replace_function("eager_attention_forward", description="Optional TileLang sparse MQA dispatch")
def deepseek_v4_eager_attention_forward_patched(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
    # --- Patch.1 ---
    attention_implementation = veomni_dsa_attention_implementation.value
    if attention_implementation not in {"eager", "tilelang"}:
        raise ValueError(
            "DeepSeek-V4 does not support "
            f"dsa_attention_implementation={attention_implementation!r}; expected 'eager' or 'tilelang'"
        )
    use_tilelang = (
        attention_implementation == "tilelang"
        and query.is_cuda
        and query.dtype == torch.bfloat16
        and key.dtype == torch.bfloat16
        and value.dtype == torch.bfloat16
        and query.shape[-1] == 1 << (query.shape[-1] - 1).bit_length()
        and isinstance(attention_mask, torch.Tensor)
        and dropout == 0
        and key.shape[1] == 1
    )
    if use_tilelang:
        batch, _, seq_len, _ = query.shape
        kv_len = key.shape[-2]
        compressed_len = max(0, kv_len - seq_len)
        compressed_budget = compressed_len
        indexer = getattr(getattr(module, "compressor", None), "indexer", None)
        if indexer is not None:
            compressed_budget = min(compressed_len, indexer.index_topk)
        selected_width = min(kv_len, module.sliding_window + compressed_budget)

        mask = attention_mask
        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, -1, -1, -1)
        allowed = mask[:, 0] if mask.dtype == torch.bool else mask[:, 0] >= 0
        _, topk_indices = allowed.to(torch.int8).topk(selected_width, dim=-1, sorted=False)
        selected_valid = allowed.gather(-1, topk_indices)
        topk_indices = topk_indices.to(torch.int32).masked_fill(~selected_valid, -1).contiguous()
        sinks = kwargs.get("s_aux", module.sinks)
        attn_output = sparse_attn_tilelang(
            query.transpose(1, 2).contiguous(),
            key[:, 0].contiguous(),
            sinks.float().contiguous(),
            topk_indices,
            scaling,
        )
        return attn_output, None
    # --- Patch.1 ---

    # --- Patch.2 ---
    # Under Ulysses SP, ``query`` only holds a head shard while the module still
    # reports the full ``num_key_value_groups``. Expand KV to the *local* query
    # head count so matmul shapes stay consistent.
    n_rep = query.shape[1] // key.shape[1]
    key_states = repeat_kv(key, n_rep)
    value_states = repeat_kv(value, n_rep)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sinks = kwargs.get("s_aux", module.sinks)
    sinks = sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training).to(value_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
    # --- Patch.2 ---


# ================================================================
# Patch: DeepseekV4Model.forward
# 1. Convert collator-provided cu-seqlens into reusable packed slices once.
# 2. Keep use_cache=False forwards stateless so the TileLang indexer can run.
# 3. Under Ulysses SP the collator keeps full ``attention_mask`` /
#    ``cu_seq_lens_*`` while slicing ``input_ids`` / local ``position_ids``.
#    Build the sliding-window mask and packed compression metadata on the full
#    sequence length so attention matches non-SP semantics after the all-gather
#    inside ``DeepseekV4Attention``.
# ================================================================
@config.override_method(
    "DeepseekV4Model.forward",
    description="Packed boundaries, SP-aware full-sequence masks, stateless indexer dispatch",
)
def deepseek_v4_model_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> MoeModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    # Stateless prefill/training must keep the cache absent: the TileLang
    # Lightning Indexer dispatch is intentionally cache-free, and creating a
    # DynamicCache here would silently force its eager decode fallback even
    # when use_cache=False.
    if past_key_values is None and use_cache:
        past_key_values = DynamicCache(config=self.config)
    return_cache = past_key_values if use_cache else None
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if position_ids is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
        position_ids = position_ids.unsqueeze(0)

    ulysses_enabled = get_parallel_state().ulysses_enabled
    ulysses_group = get_parallel_state().ulysses_group if ulysses_enabled else None
    ulysses_size = get_parallel_state().ulysses_size if ulysses_enabled else 1
    local_seq_len = inputs_embeds.shape[1]
    full_seq_len = local_seq_len * ulysses_size if ulysses_enabled else local_seq_len
    full_position_ids = (
        gather_outputs(position_ids, gather_dim=-1, group=ulysses_group) if ulysses_enabled else position_ids
    )

    cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
    if isinstance(cu_seq_lens_q, torch.Tensor) and inputs_embeds.shape[0] == 1:
        boundaries = cu_seq_lens_q.detach().cpu().tolist()
        if boundaries[0] != 0 or boundaries[-1] != full_seq_len:
            raise ValueError(
                "DeepSeek V4 packed cu_seq_lens_q must span the full sequence; "
                f"got {boundaries} for length {full_seq_len}"
            )
        packed_sequence_slices = tuple(zip(boundaries[:-1], boundaries[1:], strict=True))
        kwargs["packed_sequence_slices"] = packed_sequence_slices
        compress_rates = tuple(self.config.compress_rates.values())
        hca_rate = self.config.compress_rates["heavily_compressed_attention"]
        # Metadata is indexed by global positions / cu-seqlens; under SP the
        # collator already provides full-sequence cu-seqlens while local embeds
        # are only one shard, so materialize a full-length reference tensor.
        metadata_reference = inputs_embeds.new_empty(inputs_embeds.shape[0], full_seq_len, inputs_embeds.shape[-1])
        kwargs["packed_compression_metadata"] = build_packed_compression_metadata(
            metadata_reference,
            full_position_ids,
            packed_sequence_slices,
            compress_rates,
            block_bias_rates=(hca_rate,),
        )
        # Packed training combines independent samples in one physical row;
        # treating that row as a decode cache would merge their KV histories.
        past_key_values = None
        return_cache = None

    if isinstance(attention_mask, dict):
        causal_mask = next(iter(attention_mask.values()))
    else:
        mask_embeds = inputs_embeds
        mask_position_ids = position_ids
        if ulysses_enabled:
            # SP collator keeps the full 2D attention_mask while slicing
            # input_ids; build the 4D sliding-window mask on the full length.
            mask_embeds = inputs_embeds.new_empty(inputs_embeds.shape[0], full_seq_len, inputs_embeds.shape[-1])
            mask_position_ids = full_position_ids
        causal_mask = create_sliding_window_causal_mask(
            config=self.config,
            inputs_embeds=mask_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=mask_position_ids,
        )
    if "packed_sequence_slices" in kwargs:
        causal_mask = isolate_packed_causal_mask_(causal_mask, kwargs["packed_sequence_slices"])
    hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
    position_embeddings = {
        "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
        "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
    }

    for layer in self.layers:
        hidden_states = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=causal_mask,
            input_ids=input_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    hidden_states = self.norm(self.hc_head(hidden_states))
    return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=return_cache)


# ================================================================
# Patch: DeepseekV4Experts
# 1. Drop upstream ``@use_experts_implementation`` decorator — it would
#    dispatch to ``grouped_mm`` / HF fused paths and bypass VeOmni's fused
#    MoE kernel.
# 2. OpSlot guard for fused-MoE: when ``veomni_moe_experts_forward`` is
#    bound to a non-eager kernel, call ``fused_moe_forward`` with stacked
#    ``gate_up_proj`` and pass ``swiglu_limit`` explicitly. Otherwise fall
#    through to the eager loop.
# 3. Preserve V4's gpt-oss-style ``swiglu_limit`` clamp on gate / up
#    pre-activations (paper §2.1 — required for V4's training stability).
# Layout matches v5 upstream (direct, no transpose):
#   gate_up_proj [E, 2*I, H],  down_proj [E, H, I]
# ================================================================
@config.replace_class(
    "DeepseekV4Experts",
    description="Use v5 gate_up_proj expert layout with OpSlot-guarded VeOmni fused-MoE path",
)
class PatchedDeepseekV4Experts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = config.swiglu_limit

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        # --- Patch.2 ---
        if veomni_moe_experts_forward.use_non_eager_impl:
            return fused_moe_forward(
                num_experts=self.num_experts,
                routing_weights=top_k_weights.to(final_hidden_states.dtype),
                selected_experts=top_k_index,
                hidden_states=hidden_states,
                fc1_1_weight=None,
                fc1_2_weight=None,
                fc2_weight=self.down_proj,
                fc1_1_2_weight=self.gate_up_proj,
                swiglu_limit=self.limit,
            )
        # --- Patch.2 ---

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
            current_hidden_states = self._apply_gate(gate_up)
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # --- Patch.3 ---
        # gpt-oss-style clamped SwiGLU. Lives on the class so
        # ``@use_experts_implementation`` backends (when re-applied
        # downstream) get the same clamp semantics on top of their packed
        # gate_up output. Identical to upstream HF.
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up
        # --- Patch.3 ---


# ================================================================
# Patch: DeepseekV4MLP.forward
# Shared experts can use functional Liger SwiGLU via OpSlot. Keep the class
# because its V4-specific intermediate-size mapping is incompatible with
# LigerSwiGLUMLP construction. Eager fallback retains official FP32 clamp.
# ================================================================
@config.override_method(
    "DeepseekV4MLP.forward",
    description="Clamp-aware shared-expert SwiGLU with optional Liger fused silu-mul",
)
def deepseek_v4_mlp_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    # Official DeepSeek-V4 shared experts clamp gate/up before silu*mul. Apply
    # that first, then optionally fuse only the silu*mul via Liger. The generic
    # ``veomni_swiglu_mlp(self, x)`` path re-runs projections without clamp and
    # would change arithmetic under the default ``swiglu_limit``.
    dtype = x.dtype
    gate = self.gate_proj(x).float().clamp(max=self.config.swiglu_limit)
    up = (
        self.up_proj(x)
        .float()
        .clamp(
            min=-self.config.swiglu_limit,
            max=self.config.swiglu_limit,
        )
    )
    if veomni_swiglu_mlp.use_non_eager_impl:
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction

        return self.down_proj(LigerSiLUMulFunction.apply(gate.to(dtype), up.to(dtype)))

    hidden_states = self.act_fn(gate) * up
    return self.down_proj(hidden_states.to(dtype))


@config.override_method(
    "DeepseekV4TopKRouter.forward",
    description="Match the official DeepSeek-V4 FP32 router projection",
)
def deepseek_v4_topk_router_forward_patched(
    self,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    device_type = flat.device.type if isinstance(flat.device.type, str) and flat.device.type != "mps" else "cpu"
    with maybe_autocast(device_type=device_type, enabled=False):
        logits = F.linear(flat.float(), self.weight.float())
    correction_bias = self.e_score_correction_bias.float()
    scores = self.score_fn(logits)
    indices = torch.topk(scores + correction_bias, self.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


@config.override_method(
    "DeepseekV4HashRouter.forward",
    description="Match the official DeepSeek-V4 FP32 hash-router projection",
)
def deepseek_v4_hash_router_forward_patched(
    self,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    device_type = flat.device.type if isinstance(flat.device.type, str) and flat.device.type != "mps" else "cpu"
    with maybe_autocast(device_type=device_type, enabled=False):
        logits = F.linear(flat.float(), self.weight.float())
    scores = self.score_fn(logits)
    indices = self.tid2eid[input_ids.reshape(-1)].long()
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


# ================================================================
# Patch: DeepseekV4ForCausalLM.forward
# 1. OpSlot guard for fused cross-entropy loss; falls back to the eager
#    HF loss path when no fused kernel is bound. Returns the unified
#    ``MoeCausalLMOutputWithLogProbs`` so callers can read per-token
#    log-probs and entropy alongside the loss (required by RL/PPO-style
#    trainers).
# 2. OpSlot guard for ``load_balancing_loss``; falls back to the upstream
#    ``load_balancing_loss_func`` (which V4 re-defines in-module — not
#    imported from ``transformers``).
# ================================================================
@config.override_method(
    "DeepseekV4ForCausalLM.forward",
    description="OpSlot guard for fused cross entropy in DeepseekV4ForCausalLM.forward",
)
def deepseek_v4_forcausallm_forward_patched(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> MoeCausalLMOutputWithLogProbs:
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )

    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
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
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            if fused_linear_aux is not None:
                logits = None
    else:
        logits = self.lm_head(hidden_states)
    # --- Patch.1 ---

    aux_loss = None
    if output_router_logits:
        # --- Patch.2 ---
        if veomni_load_balancing_loss.use_non_eager_impl:
            aux_loss = veomni_load_balancing_loss(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
        else:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
        # ``load_balancing_loss_func`` returns scalar ``int`` 0 when
        # ``router_logits`` is None / not a tuple — guard before composing
        # so we don't trip ``int.to(...)`` on the eager fallback.
        if labels is not None and isinstance(aux_loss, torch.Tensor):
            loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)
        # --- Patch.2 ---

    return MoeCausalLMOutputWithLogProbs(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
        fused_linear_aux=fused_linear_aux,
    )


# ================================================================
# Patch: DeepseekV4ForCausalLM.get_parallel_plan
# 1. Register VeOmni EP parallel plan on the v5 generated class.
# ================================================================
@config.override_method(
    "DeepseekV4ForCausalLM.get_parallel_plan",
    description="Register DeepseekV4 expert parallel plan for v5 generated modeling",
)
def deepseek_v4_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()
