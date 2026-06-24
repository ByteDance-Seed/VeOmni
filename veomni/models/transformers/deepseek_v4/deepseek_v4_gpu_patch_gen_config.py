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
1. ``DeepseekV4Experts`` — drops upstream ``@use_experts_implementation``
   (which would otherwise dispatch to ``grouped_mm`` and bypass VeOmni's
   fused MoE kernel). Keeps the v5 stacked ``gate_up_proj [E, 2*I, H]`` /
   ``down_proj [E, H, I]`` layout and the gpt-oss-style ``swiglu_limit``
   clamp. Dispatch is OpSlot-guarded (``veomni_moe_experts_forward``):
   non-eager -> ``fused_moe_forward``; eager -> per-expert loop.
2. ``DeepseekV4Indexer.forward`` / ``DeepseekV4CSACompressor.forward`` /
   ``DeepseekV4Attention.forward`` — opt-in DeepSeek sparse attention path:
   cuDNN Frontend DSA indexer plus FlashMLA sparse prefill forward paired with
   cuDNN Frontend DSA backward for CSA layers.
3. ``DeepseekV4ForCausalLM.forward`` — OpSlot guard for fused
   cross-entropy (``veomni_causal_lm_loss``) + ``MoeCausalLMOutputWithLogProbs``
   so callers can read per-token log-probs / entropy alongside the loss.
4. Register ``get_parallel_plan`` on ``DeepseekV4ForCausalLM``.

Intentionally NOT patched:

- ``DeepseekV4RMSNorm`` / ``DeepseekV4UnweightedRMSNorm`` — DeepSeek-V4 ships
  two RMSNorm flavours (the second is unweighted and used inside the
  HCA/CSA compressors). LigerRMSNorm replaces only the standard form, and a
  blind swap would shadow the unweighted variant. RoPE-determinism /
  batch-invariant RMSNorm are wired separately at runtime by future
  ``device_patch.py`` infra (mirroring DeepseekV3) when needed.
- ``DeepseekV4MLP`` — also used as ``shared_experts`` with a custom
  ``moe_intermediate_size`` (via ``attribute_map["intermediate_size"] =
  "moe_intermediate_size"``). ``LigerSwiGLUMLP.__init__`` rejects the
  ``intermediate_size`` kwarg pattern that DeepSeek-V4 uses, so swapping
  would break shared-expert construction. Same reasoning as DeepseekV3.
- ``apply_rotary_pos_emb`` — DeepSeek-V4 uses a *partial* RoPE (the
  trailing ``qk_rope_head_dim`` slice only, with the leading nope channels
  untouched) plus an interleaved ``repeat_interleave(2)`` cos/sin layout
  that ``liger_rotary_pos_emb`` does not implement. SKILL.md flags this
  exact case (partial_rotary -> liger NaN).
- ``DeepseekV4Model.forward`` — top-level ``hidden_states`` is *4D*
  (``[B, S, hc_mult, D]`` for the manifold-constrained Hyper-Connection
  residual stack); existing VeOmni SP collators assume 3D
  ``[B, S, D]`` at the embed boundary. Leaving the upstream forward
  untouched keeps single-rank training / inference correct while a 4D-aware
  SP path is designed.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.deepseek_v4.modeling_deepseek_v4 import load_balancing_loss_func
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.ops import fused_moe_forward
from veomni.ops.dispatch import OpsConfigSlot, OpSlot
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import MoeCausalLMOutputWithLogProbs


# OpSlot declarations — mirrored into the generated module via
# ``add_post_import_block`` below. The duplicate at module scope here is
# only for IDE/type-check friendliness while authoring this file; the
# runtime slots used by the generated modeling are bound at model-build
# time by ``_bind_veomni_ops()`` in ``veomni/models/auto.py``.
veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
veomni_load_balancing_loss = OpSlot("load_balancing_loss", "standard")
veomni_dsa_indexer_backend = OpsConfigSlot("dsa_indexer_backend")
veomni_dsa_attention_backend = OpsConfigSlot("dsa_attention_backend")


config = PatchConfig(
    source_module="transformers.models.deepseek_v4.modeling_deepseek_v4",
    target_file="patched_modeling_deepseek_v4_gpu.py",
    description="DeepseekV4 with VeOmni fused-MoE + OpSlot-guarded fused-CE patches",
)

config.add_import("veomni.ops", names=["fused_moe_forward"])

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
    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    veomni_load_balancing_loss = OpSlot("load_balancing_loss", "standard")
    veomni_dsa_indexer_backend = OpsConfigSlot("dsa_indexer_backend")
    veomni_dsa_attention_backend = OpsConfigSlot("dsa_attention_backend")


    def _deepseek_v4_compact_sparse_indices(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = indices >= 0
        order = torch.argsort((~valid).to(torch.int64), dim=-1, stable=True)
        compacted = indices.gather(-1, order)
        compacted = torch.where(compacted >= 0, compacted, torch.zeros_like(compacted))
        return compacted, valid.sum(dim=-1).to(torch.int32)


    def _deepseek_v4_build_flashmla_indices(
        attention_mask: torch.Tensor | None,
        top_k_indices: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        base_kv_len: int,
        sliding_window: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        window = min(sliding_window, base_kv_len)
        query_positions = torch.arange(seq_len, device=top_k_indices.device).view(1, seq_len, 1)
        offsets = torch.arange(window - 1, -1, -1, device=top_k_indices.device).view(1, 1, window)
        dense_indices = (query_positions - offsets).expand(batch_size, -1, -1)
        dense_valid = dense_indices >= 0
        if attention_mask is not None:
            if attention_mask.dim() != 4:
                raise ValueError("dsa_attention_backend='flashmla_cudnn' requires a 4D attention mask")
            dense_mask = attention_mask[:, 0, :, :base_kv_len].gather(-1, dense_indices.clamp(0, base_kv_len - 1))
            dense_valid = dense_valid & torch.isfinite(dense_mask) & (dense_mask >= 0)
        dense_indices = torch.where(dense_valid, dense_indices, torch.full_like(dense_indices, -1))
        compressed_indices = torch.where(top_k_indices >= 0, top_k_indices + base_kv_len, top_k_indices)
        sparse_indices = torch.cat((dense_indices, compressed_indices), dim=-1)
        curr_len = sparse_indices.shape[-1]
        if curr_len % 128 != 0:
            pad_len = 128 - (curr_len % 128)
            pad = torch.full(
                (*sparse_indices.shape[:-1], pad_len),
                -1,
                dtype=sparse_indices.dtype,
                device=sparse_indices.device,
            )
            sparse_indices = torch.cat((sparse_indices, pad), dim=-1)
        return _deepseek_v4_compact_sparse_indices(sparse_indices)
    """
)


def _deepseek_v4_compact_sparse_indices(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = indices >= 0
    order = torch.argsort((~valid).to(torch.int64), dim=-1, stable=True)
    compacted = indices.gather(-1, order)
    compacted = torch.where(compacted >= 0, compacted, torch.zeros_like(compacted))
    return compacted, valid.sum(dim=-1).to(torch.int32)


def _deepseek_v4_build_flashmla_indices(
    attention_mask: torch.Tensor | None,
    top_k_indices: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    base_kv_len: int,
    sliding_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    window = min(sliding_window, base_kv_len)
    query_positions = torch.arange(seq_len, device=top_k_indices.device).view(1, seq_len, 1)
    offsets = torch.arange(window - 1, -1, -1, device=top_k_indices.device).view(1, 1, window)
    dense_indices = (query_positions - offsets).expand(batch_size, -1, -1)
    dense_valid = dense_indices >= 0
    if attention_mask is not None:
        if attention_mask.dim() != 4:
            raise ValueError("dsa_attention_backend='flashmla_cudnn' requires a 4D attention mask")
        dense_mask = attention_mask[:, 0, :, :base_kv_len].gather(-1, dense_indices.clamp(0, base_kv_len - 1))
        dense_valid = dense_valid & torch.isfinite(dense_mask) & (dense_mask >= 0)
    dense_indices = torch.where(dense_valid, dense_indices, torch.full_like(dense_indices, -1))
    compressed_indices = torch.where(top_k_indices >= 0, top_k_indices + base_kv_len, top_k_indices)
    sparse_indices = torch.cat((dense_indices, compressed_indices), dim=-1)
    curr_len = sparse_indices.shape[-1]
    if curr_len % 128 != 0:
        pad_len = 128 - (curr_len % 128)
        pad = torch.full(
            (*sparse_indices.shape[:-1], pad_len),
            -1,
            dtype=sparse_indices.dtype,
            device=sparse_indices.device,
        )
        sparse_indices = torch.cat((sparse_indices, pad), dim=-1)
    return _deepseek_v4_compact_sparse_indices(sparse_indices)


# ================================================================
# Patch: DeepseekV4Experts
# 1. Drop upstream ``@use_experts_implementation`` decorator — it would
#    dispatch to ``grouped_mm`` / HF fused paths and bypass VeOmni's fused
#    MoE kernel.
# 2. OpSlot guard for fused-MoE: when ``veomni_moe_experts_forward`` is
#    bound to a non-eager kernel, call ``fused_moe_forward`` with stacked
#    ``gate_up_proj``. Otherwise fall through to the eager loop.
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


@config.override_method(
    "DeepseekV4Indexer.forward",
    description="Use cuDNN Frontend DSA indexer kernels when supported",
)
def deepseek_v4_indexer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: Cache | None,
    layer_idx: int,
) -> torch.LongTensor:
    batch, seq_len, _ = hidden_states.shape
    cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if cache_layer is None:
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
        chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("indexer", kv, gate)

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

    compressed_len = compressed_kv.shape[1]
    top_k = min(self.index_topk, compressed_len)
    if compressed_len == 0:
        return compressed_kv.new_empty((batch, seq_len, 0), dtype=torch.long)

    weights = self.weights_proj(hidden_states).float() * self.weights_scaling
    indexer_backend = veomni_dsa_indexer_backend.value
    if indexer_backend not in ("eager", "cudnn"):
        raise ValueError(f"Unknown dsa_indexer_backend={indexer_backend!r}; expected 'eager' or 'cudnn'")
    if indexer_backend == "cudnn":
        from veomni.ops.kernels.deepseek_sparse_attention.flashmla_cudnn import indexer_select_topk

        unsupported_reasons = []
        if not hidden_states.is_cuda:
            unsupported_reasons.append("hidden_states must be CUDA")
        if past_key_values is not None:
            unsupported_reasons.append("KV cache is not supported")
        if q.dtype not in (torch.bfloat16, torch.float16):
            unsupported_reasons.append(f"q dtype must be bf16/fp16, got {q.dtype}")
        if compressed_kv.dtype not in (torch.bfloat16, torch.float16):
            unsupported_reasons.append(f"k dtype must be bf16/fp16, got {compressed_kv.dtype}")
        if weights.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            unsupported_reasons.append(f"weights dtype must be bf16/fp16/fp32, got {weights.dtype}")
        if self.num_heads not in (32, 64):
            unsupported_reasons.append(f"qhead_per_kv_head must be 32 or 64, got {self.num_heads}")
        if unsupported_reasons:
            raise ValueError("dsa_indexer_backend='cudnn' is not supported: " + "; ".join(unsupported_reasons))
        return indexer_select_topk(
            q,
            compressed_kv,
            weights.to(q.dtype),
            top_k,
            ratio=self.compress_rate,
            qhead_per_kv_head=self.num_heads,
            sm_scale=self.softmax_scale,
        )

    scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores) * self.softmax_scale
    index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)

    causal_threshold = (position_ids + 1) // self.compress_rate
    entry_indices = torch.arange(compressed_len, device=index_scores.device)
    future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
    index_scores = index_scores.masked_fill(future_mask, float("-inf"))
    top_k_indices = index_scores.topk(top_k, dim=-1).indices
    invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
    return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)


@config.override_method(
    "DeepseekV4CSACompressor.forward",
    description="Return top-k indices needed by the opt-in DSA attention backend",
)
def deepseek_v4_csa_compressor_forward_patched(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: Cache | None,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, _ = hidden_states.shape
    cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

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
    if top_k_indices.shape[-1] < self.indexer.index_topk:
        pad = top_k_indices.new_full((*top_k_indices.shape[:2], self.indexer.index_topk - top_k_indices.shape[-1]), -1)
        top_k_indices = torch.cat((top_k_indices, pad), dim=-1)
    compressed_len = compressed_kv.shape[2]
    valid = top_k_indices >= 0
    safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
    block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
    block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
    return compressed_kv, block_bias[..., :compressed_len], top_k_indices


@config.override_method(
    "DeepseekV4Attention.forward",
    description="Use FlashMLA sparse prefill forward with cuDNN FE DSA backward for CSA layers",
)
def deepseek_v4_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings[self.rope_layer_type]

    q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
    q = self.q_b_norm(q)
    q = apply_rotary_pos_emb(q, cos, sin)

    kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
    kv = apply_rotary_pos_emb(kv, cos, sin)

    if past_key_values is not None:
        kv = past_key_values.update(kv, kv, self.layer_idx)[0]

    block_bias = None
    top_k_indices = None
    base_kv_len = kv.shape[2]
    if self.compressor is not None:
        compressor_result = self.compressor(hidden_states, q_residual, position_ids, past_key_values, self.layer_idx)
        if self.layer_type == "compressed_sparse_attention":
            compressed_kv, block_bias, top_k_indices = compressor_result
        else:
            compressed_kv, block_bias = compressor_result
        kv = torch.cat([kv, compressed_kv], dim=2)

    if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
        if block_bias is not None:
            attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
        else:
            attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)

    attention_backend = veomni_dsa_attention_backend.value
    if attention_backend not in ("eager", "flashmla_cudnn"):
        raise ValueError(f"Unknown dsa_attention_backend={attention_backend!r}; expected 'eager' or 'flashmla_cudnn'")
    if attention_backend == "flashmla_cudnn" and self.layer_type == "compressed_sparse_attention":
        from veomni.ops.kernels.deepseek_sparse_attention.flashmla_cudnn import (
            check_flash_mla_sparse_forward_compatible,
            flash_mla_sparse_attention_with_cudnn_backward,
        )

        unsupported_reasons = []
        if not hidden_states.is_cuda:
            unsupported_reasons.append("hidden_states must be CUDA")
        if past_key_values is not None:
            unsupported_reasons.append("KV cache is not supported")
        if self.training and self.attention_dropout != 0:
            unsupported_reasons.append("attention_dropout must be 0")
        if top_k_indices is None:
            unsupported_reasons.append("CSA top-k indices are missing")
        rope_dim = cos.shape[-1] * 2
        if rope_dim != 64:
            unsupported_reasons.append(f"qk_rope_head_dim must be 64, got {rope_dim}")
        if self.head_dim != 512:
            unsupported_reasons.append(f"head_dim must be 512, got {self.head_dim}")
        if unsupported_reasons:
            raise ValueError(
                "dsa_attention_backend='flashmla_cudnn' is not supported: " + "; ".join(unsupported_reasons)
            )

        q_bshd = q.transpose(1, 2)
        kv_bshd = kv.transpose(1, 2)
        q_pe = q_bshd[..., -rope_dim:]
        k_pe = kv_bshd[..., -rope_dim:]
        q_nope = q_bshd.clone()
        q_nope[..., -rope_dim:] = 0
        sparse_indices, topk_length = _deepseek_v4_build_flashmla_indices(
            attention_mask,
            top_k_indices,
            batch_size=hidden_states.shape[0],
            seq_len=hidden_states.shape[1],
            base_kv_len=base_kv_len,
            sliding_window=self.sliding_window,
        )
        attn_sink = self.sinks.float()
        compatible, reason = check_flash_mla_sparse_forward_compatible(
            q_pe,
            k_pe,
            kv_bshd,
            q_nope,
            sparse_indices,
            attn_sink,
            topk_length,
        )
        if not compatible:
            raise ValueError("dsa_attention_backend='flashmla_cudnn' is not supported: " + reason)
        attn_output = flash_mla_sparse_attention_with_cudnn_backward(
            q_pe.contiguous(),
            k_pe.contiguous(),
            kv_bshd.contiguous(),
            q_nope.contiguous(),
            sparse_indices,
            attn_sink=attn_sink,
            topk_length=topk_length,
            softmax_scale=self.scaling,
        )
        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)
        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, None

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        q,
        kv,
        kv,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        s_aux=self.sinks,
        **kwargs,
    )

    attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

    grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
    grouped = self.o_a_proj(grouped).flatten(2)
    output = self.o_b_proj(grouped)
    return output, attn_weights


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
