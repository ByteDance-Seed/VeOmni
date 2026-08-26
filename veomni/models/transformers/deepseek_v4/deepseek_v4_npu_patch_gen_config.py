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
Patch configuration for DeepseekV4 NPU patched modeling generation.

Regen command:
patchgen veomni.models.transformers.deepseek_v4.deepseek_v4_npu_patch_gen_config -o veomni/models/transformers/deepseek_v4/generated --diff

NPU reuses every GPU structural and numerics patch verbatim (RMSNorm/RoPE/SwiGLU
dispatch, mHC dispatch, packed attention, indexer, model forward, fused-MoE
experts, fused-CE ForCausalLM.forward, parallel plan) by import rather than
duplication, mirroring the ``deepseek_v3`` GPU/NPU pair. This is safe rather
than merely convenient:

- ``DeepseekV4RMSNorm.forward`` / ``DeepseekV4UnweightedRMSNorm.forward`` /
  ``DeepseekV4MLP.forward`` dispatch to Liger kernels only when their OpSlot
  is bound to a non-eager implementation; Liger requires CUDA, so these fall
  straight through to the shared eager arithmetic on NPU without any change
  needed here.
- ``DeepseekV4Indexer.forward`` / ``eager_attention_forward`` gate their
  TileLang fast paths behind ``.is_cuda`` (and SM90 checks inside
  ``veomni.ops.kernels.deepseek_v4``). The module-level import of
  ``sparse_attn_tilelang`` / ``v4_lighting_indexer`` is lazy-safe on NPU:
  ``veomni/ops/kernels/deepseek_v4/__init__.py`` only imports TileLang inside
  the wrapper *bodies*, guarded by ``_require_tilelang_sm90()``, which is
  never reached because the ``.is_cuda`` condition short-circuits first. Both
  functions fall straight through to the eager PyTorch computation on NPU.
- The mHC pre/post/head patches are OpSlot-guarded
  (``veomni_mhc_{pre,post,head}``); ``mhc_implementation`` defaults to
  ``"eager"`` (see ``OpsImplementationConfig.mhc_implementation`` —
  ``tilelang`` is documented SM90+ only), so the pure-PyTorch branch already
  in these functions is what actually runs on NPU without any change.
- ``DeepseekV4Experts.forward`` dispatches through the OpSlot-guarded
  ``fused_moe_forward``, which already has an NPU backend
  (``moe_implementation=fused_npu`` — see
  ``veomni/ops/kernels/moe/npu_group_gemm.py``); no per-model MoE change
  needed.
- Ulysses SP support inside ``DeepseekV4Attention.forward`` /
  ``DeepseekV4Model.forward`` is orthogonal to device backend (plain
  ``torch.distributed`` collectives via ``sequence_parallel``), but is
  untested on NPU with this model — keep ``ulysses_size: 1`` in the NPU
  training config until it has been validated.

NPU-only additions (not registered on the GPU config — see each patch below
for why they are scoped to this file rather than shared):

1. ``DeepseekV4HCACompressor`` / ``DeepseekV4CSACompressor`` / ``DeepseekV4Indexer``
   ``__init__`` — shard ``position_bias`` on dim-1 instead of FSDP2's default
   dim-0.
2. ``DeepseekV4HCACompressor`` / ``DeepseekV4CSACompressor`` ``forward`` —
   anchor gradient participation for packed micro-batches with zero
   compression windows.

Intentionally NOT patched (same rationale as the GPU config, restated here so
NPU readers don't have to cross-reference):

- ``apply_rotary_pos_emb`` — DeepSeek-V4 uses a *partial* RoPE (the
  trailing ``qk_rope_head_dim`` slice only, with the leading nope channels
  untouched) plus an interleaved ``repeat_interleave(2)`` cos/sin layout
  that neither Liger's ``liger_rotary_pos_emb`` nor the generic NPU
  ``apply_rotary_pos_emb_npu`` kernel (``veomni/ops/kernels/rotary/npu.py``)
  implement — that kernel assumes a leading-slice partial rotary layout, not
  V4's trailing-slice + ``repeat_interleave(2)`` layout. Forcing either
  kernel in would silently change numerics. Wire a dedicated
  ``device_patch.py`` (mirroring ``deepseek_v3/device_patch.py``) once a
  verified NPU kernel for this exact layout exists.
- ``DeepseekV4Attention.forward`` — eager-only on every backend
  (``_supports_flash_attn/_supports_sdpa/_supports_flex_attn = False``); set
  ``model.ops_implementation.attn_implementation: eager`` in the training
  config for NPU runs (see ``configs/text/deepseek_v4_npu.yaml``).
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4HCACache,
    apply_rotary_pos_emb,
)

from veomni.models.transformers.deepseek_v4.packed_utils import (
    compress_packed_windows,
    packed_compressed_block_bias,
)
from veomni.patchgen.patch_spec import PatchConfig


from .deepseek_v4_gpu_patch_gen_config import (
    PatchedDeepseekV4Experts,
    deepseek_v4_decoder_layer_forward_patched,
    deepseek_v4_forcausallm_forward_patched,
    deepseek_v4_get_parallel_plan_patched,
    deepseek_v4_hyper_connection_forward_patched,
    deepseek_v4_hyper_head_forward_patched,
    deepseek_v4_mlp_forward_patched,
    deepseek_v4_model_forward_patched,
    deepseek_v4_rms_norm_forward_patched,
    deepseek_v4_rotary_embedding_forward_patched,
    deepseek_v4_unweighted_rmsnorm_forward_patched,
)


config = PatchConfig(
    source_module="transformers.models.deepseek_v4.modeling_deepseek_v4",
    target_file="patched_modeling_deepseek_v4_npu.py",
    description="DeepseekV4 NPU sibling — reuses every GPU structural/numerics patch, plus NPU-only FSDP2 hardening",
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
        "CompressedCandidates",
        "build_packed_compression_metadata",
        "build_packed_sparse_attention_indices",
        "build_sparse_attention_indices",
        "compress_packed_windows",
        "isolate_packed_causal_mask_",
        "mask_sparse_attention_indices",
        "packed_compressed_block_bias",
        "packed_compressed_causal_ranges",
    ],
)

# Same rationale as the GPU config: surface MoeCausalLMOutputWithLogProbs so
# the reused ForCausalLM.forward can return per-token log-probs / entropy as
# constructor fields (FSDP2 unshard-hook safe — see GPU config comment).
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "MoeCausalLMOutputWithLogProbs"],
)
config.drop_import_names("MoeCausalLMOutputWithPast")

# The reused TopKRouter.forward calls the router-replay hook, so the generated
# NPU module needs the same names the GPU one imports.
config.add_import(
    "veomni.utils.moe_router_replay",
    names=["get_active_replay", "maybe_replay_indices"],
)

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
def deepseek_v4_indexer_forward_npu(
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
            apply_rope=apply_rotary_pos_emb,
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

        compressed = self.kv_norm(
            (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype))
            .sum(dim=2, dtype=torch.float32)
            .to(new_kv.dtype)
        )
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
    if indexer_implementation not in {"eager", "npu", "tilelang"}:
        raise ValueError(
            "DeepSeek-V4 does not support "
            f"dsa_indexer_implementation={indexer_implementation!r}; expected 'eager', 'npu', or 'tilelang'"
        )
    canonical_positions = torch.arange(seq_len, device=position_ids.device).unsqueeze(0).expand_as(position_ids)
    packed_ranges = None
    if packed_compression_metadata is not None and cache_layer is None:
        packed_ranges = packed_compressed_causal_ranges(packed_compression_metadata[self.compress_rate])
    use_npu = (
        indexer_implementation == "npu"
        and hidden_states.device.type == "npu"
        and q.dtype == torch.bfloat16
        and compressed_kv.dtype == torch.bfloat16
        and cache_layer is None
        and packed_ranges is None
        and compressed_len > 0
        and torch.equal(position_ids, canonical_positions)
    )
    if use_npu:
        from veomni.ops.kernels.deepseek_v4.npu_lightning_indexer import npu_lightning_indexer

        top_k_indices, _ = npu_lightning_indexer(
            q, compressed_kv, weights, top_k, compress_rate=self.compress_rate
        )
        return top_k_indices.to(torch.long)

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
def deepseek_v4_attention_forward_npu(
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
    q = self.q_b_norm(self.q_b_proj(q_residual).view(*hidden_shape))
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
    compressed_kv = None
    compressed_topk_indices = None
    original_kv = kv
    if self.compressor is not None:
        compressed_kv, block_bias, compressed_topk_indices = self.compressor(
            compressor_hidden,
            compressor_q_residual,
            compressor_position_ids,
            past_key_values,
            self.layer_idx,
            packed_sequence_slices=kwargs.get("packed_sequence_slices"),
            packed_compression_metadata=kwargs.get("packed_compression_metadata"),
            return_topk_indices=True,
            build_block_bias=True,
        )
        kv = torch.cat([kv, compressed_kv], dim=2)

    if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
        if block_bias is not None:
            attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
        else:
            attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)


    attention_implementation = veomni_dsa_attention_implementation.value
    canonical_positions = torch.arange(
        compressor_position_ids.shape[-1], device=compressor_position_ids.device
    ).unsqueeze(0).expand_as(compressor_position_ids)
    use_npu_sparse_mla = (
        attention_implementation == "npu"
        and hidden_states.device.type == "npu"
        and q.dtype == torch.bfloat16
        and original_kv.dtype == torch.bfloat16
        and past_key_values is None
        and not ulysses_enabled
        and kwargs.get("packed_sequence_slices") is None
        and torch.equal(compressor_position_ids, canonical_positions)
    )
    if use_npu_sparse_mla:
        from veomni.ops.kernels.deepseek_v4.npu_sparse_flash_mla import npu_sparse_flash_mla

        top_k_indices = (
            compressed_topk_indices if self.layer_type == "compressed_sparse_attention" else None
        )
        cmp_ratio = self.compressor.compress_rate if self.compressor is not None else 1
        attn_output = npu_sparse_flash_mla(
            q.transpose(1, 2).contiguous(),
            original_kv.transpose(1, 2).contiguous(),
            None if compressed_kv is None else compressed_kv.transpose(1, 2).contiguous(),
            top_k_indices,
            sinks=s_aux.float(),
            softmax_scale=self.scaling,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=self.sliding_window - 1,
            ori_win_right=0,
        )
        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)
        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, None
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
def deepseek_v4_eager_attention_forward_npu(
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
    if attention_implementation not in {"eager", "npu", "tilelang"}:
        raise ValueError(
            "DeepSeek-V4 does not support "
            f"dsa_attention_implementation={attention_implementation!r}; expected 'eager', 'npu', or 'tilelang'"
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


def deepseek_v4_topk_router_forward_npu(
    self,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    device_type = flat.device.type if isinstance(flat.device.type, str) and flat.device.type != "mps" else "cpu"
    with maybe_autocast(device_type=device_type, enabled=False):
        logits = F.linear(flat.float(), self.weight.float())
    correction_bias = self.e_score_correction_bias.to(logits.device).float()
    scores = self.score_fn(logits)
    indices = torch.topk(scores + correction_bias, self.top_k, dim=-1, sorted=False).indices
    if get_active_replay() is not None:
        indices = maybe_replay_indices(self, scores, indices)
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


def deepseek_v4_hash_router_forward_npu(
    self,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    device_type = flat.device.type if isinstance(flat.device.type, str) and flat.device.type != "mps" else "cpu"
    with maybe_autocast(device_type=device_type, enabled=False):
        logits = F.linear(flat.float(), self.weight.float())
    scores = self.score_fn(logits)
    indices = self.tid2eid.to(input_ids.device)[input_ids.reshape(-1)].long()
    if get_active_replay() is not None:
        indices = maybe_replay_indices(self, scores, indices)
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

# Structural + numerics patches reused verbatim from the GPU config. Keeping
# these byte-identical across backends guarantees GPU/NPU checkpoint and
# numerics parity.
# ================================================================
config.override_method(
    "DeepseekV4RMSNorm.forward",
    replacement=deepseek_v4_rms_norm_forward_patched,
    description="OpSlot guard for Liger fused weighted RMSNorm with official eager FP32 fallback",
)

config.override_method(
    "DeepseekV4UnweightedRMSNorm.forward",
    replacement=deepseek_v4_unweighted_rmsnorm_forward_patched,
    description="OpSlot guard for Liger fused unweighted RMSNorm",
)

config.override_method(
    "DeepseekV4RotaryEmbedding.forward",
    replacement=deepseek_v4_rotary_embedding_forward_patched,
    description="Retain FP32 cos/sin for inference and use activation dtype for checkpoint-stable training",
)

config.override_method(
    "DeepseekV4MLP.forward",
    replacement=deepseek_v4_mlp_forward_patched,
    description="Clamp-aware shared-expert SwiGLU with optional Liger fused silu-mul",
)

config.override_method(
    "DeepseekV4TopKRouter.forward",
    replacement=deepseek_v4_topk_router_forward_npu,
    description="Match the official DeepSeek-V4 FP32 router projection",
)

config.override_method(
    "DeepseekV4HashRouter.forward",
    replacement=deepseek_v4_hash_router_forward_npu,
    description="Match the official DeepSeek-V4 FP32 hash-router projection",
)

config.override_method(
    "DeepseekV4HyperConnection.forward",
    replacement=deepseek_v4_hyper_connection_forward_patched,
    description="Dispatch DeepSeek V4 mHC pre/Sinkhorn/collapse through an OpSlot",
)

config.override_method(
    "DeepseekV4HyperHead.forward",
    replacement=deepseek_v4_hyper_head_forward_patched,
    description="Dispatch the final DeepSeek V4 mHC collapse through an OpSlot",
)

config.override_method(
    "DeepseekV4DecoderLayer.forward",
    replacement=deepseek_v4_decoder_layer_forward_patched,
    description="Dispatch DeepSeek V4 mHC residual post-mixing through an OpSlot",
)

config.override_method(
    "DeepseekV4Indexer.forward",
    replacement=deepseek_v4_indexer_forward_npu,
    description="Optional TileLang Lightning Indexer dispatch (no-ops to eager on NPU)",
)

config.override_method(
    "DeepseekV4Attention.forward",
    replacement=deepseek_v4_attention_forward_npu,
    description="Packed compressor path + Ulysses SP for DeepSeek-V4 eager/TileLang attention",
)

# NOTE: applied as a manual decorator call (rather than the ``replacement=``
# kwarg used above for ``override_method``/``replace_class``) since
# ``replace_function`` reuse across sibling configs is not otherwise exercised
# in-tree; this form is equivalent to ``@config.replace_function(...)`` and
# does not depend on a ``replacement=`` kwarg existing on that decorator.
config.replace_function(
    "eager_attention_forward",
    description="Optional TileLang sparse MQA dispatch (no-ops to eager on NPU)",
)(deepseek_v4_eager_attention_forward_npu)

config.override_method(
    "DeepseekV4Model.forward",
    replacement=deepseek_v4_model_forward_patched,
    description="Packed boundaries, SP-aware full-sequence masks, stateless indexer dispatch",
)

config.replace_class(
    "DeepseekV4Experts",
    replacement=PatchedDeepseekV4Experts,
    description="Use v5 gate_up_proj expert layout with OpSlot-guarded VeOmni fused-MoE path (fused_npu backend)",
)

config.override_method(
    "DeepseekV4ForCausalLM.forward",
    replacement=deepseek_v4_forcausallm_forward_patched,
    description="OpSlot guard for fused cross entropy in DeepseekV4ForCausalLM.forward",
)

config.override_method(
    "DeepseekV4ForCausalLM.get_parallel_plan",
    replacement=deepseek_v4_get_parallel_plan_patched,
    description="Register DeepseekV4 expert parallel plan for v5 generated modeling",
)


# ================================================================
# NPU-only: shard compressor/indexer position_bias on dim-1
# ================================================================
# ``DeepseekV4HCACompressor`` / ``DeepseekV4CSACompressor`` / ``DeepseekV4Indexer``
# each own a ``position_bias`` param shaped ``(compress_rate, head_dim * k)``.
# ``compress_rate`` can be as small as 4, so FSDP2's default dim-0 sharding leaves
# most ranks with an empty local shard once the FSDP world size exceeds
# ``compress_rate`` — the kind of large-world-size FSDP deployment this NPU config
# targets (``ep_size: 8`` over 16 ranks in ``configs/text/deepseek_v4_npu.yaml``).
# These three classes also own normal-sized (evenly-shardable) Linear weights, so
# wrapping the whole module as replicate-only would waste memory on those.
# ``head_dim * k`` is a large, reliably-divisible power of 2 (512/1024/256 for
# this model), so redirecting only ``position_bias`` to shard on dim-1 (via
# ``fully_shard``'s ``shard_placement_fn``, see ``torch_parallelize.py``'s
# ``_veomni_shard_placement_fn``) avoids the empty-shard case at no memory cost
# and with no ``forward()``-logic changes. Scoped to this NPU config rather than
# the shared GPU one since GPU deployments of this model have not been run at a
# world size where ``compress_rate`` sharding produces an empty local shard.
_POSITION_BIAS_SHARD_DIM_DESCRIPTION = (
    "Shard position_bias on dim-1 (large, evenly-divisible) instead of FSDP2's default "
    "dim-0 (compress_rate, can be as small as 4) -- see torch_parallelize.py's "
    "`_veomni_shard_placement_fn`."
)


@config.override_method("DeepseekV4HCACompressor.__init__", description=_POSITION_BIAS_SHARD_DIM_DESCRIPTION)
def deepseek_v4_hca_compressor_init_patched(self, config: "DeepseekV4Config") -> None:
    nn.Module.__init__(self)
    self.compress_rate = config.compress_rates["heavily_compressed_attention"]
    self.head_dim = config.head_dim
    self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
    self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
    self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))
    self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.rotary_emb = DeepseekV4RotaryEmbedding(config)
    self.position_bias._veomni_fsdp_shard_dim = 1


@config.override_method("DeepseekV4Indexer.__init__", description=_POSITION_BIAS_SHARD_DIM_DESCRIPTION)
def deepseek_v4_indexer_init_patched(self, config: "DeepseekV4Config") -> None:
    nn.Module.__init__(self)
    self.compress_rate = config.compress_rates["compressed_sparse_attention"]
    self.num_heads = config.index_n_heads
    self.head_dim = config.index_head_dim
    self.index_topk = config.index_topk
    self.softmax_scale = self.head_dim**-0.5
    self.weights_scaling = self.num_heads**-0.5
    self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
    self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
    self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
    self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
    self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
    self.rotary_emb = DeepseekV4RotaryEmbedding(config)
    self.position_bias._veomni_fsdp_shard_dim = 1


@config.override_method("DeepseekV4CSACompressor.__init__", description=_POSITION_BIAS_SHARD_DIM_DESCRIPTION)
def deepseek_v4_csa_compressor_init_patched(self, config: "DeepseekV4Config") -> None:
    nn.Module.__init__(self)
    self.compress_rate = config.compress_rates["compressed_sparse_attention"]
    self.head_dim = config.head_dim
    self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
    self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
    self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
    self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.rotary_emb = DeepseekV4RotaryEmbedding(config)
    self.indexer = DeepseekV4Indexer(config)
    self.position_bias._veomni_fsdp_shard_dim = 1


# ================================================================
# NPU-only: packed compressed-attention gradient-participation anchor
# ================================================================
# A packed micro-batch where every sequence is shorter than compress_rate
# produces zero compression windows; ``compress_packed_windows`` then returns a
# fresh zero tensor detached from the autograd graph, so ``kv_proj`` /
# ``gate_proj`` / ``position_bias`` / ``kv_norm`` would receive no gradient in
# that case while ranks with at least one full window do. FSDP2 sizes a
# bucket's gradient reduce-scatter by the set of params that actually received
# grads, so the two kinds of ranks would issue different-sized collectives for
# the same layer bucket — HCCL validates this and raises, so this is scoped as
# an NPU-only hardening patch. Anchoring the output to these params (multiplied
# by exactly 0.0, so the forward value is unchanged) keeps them attached to the
# graph regardless of whether a full window was formed, so gradient
# participation for these four params stays uniform across data-dependent
# micro-batch contents.
@config.override_method(
    "DeepseekV4HCACompressor.forward",
    description="Keep HCA compression local to packed sequences, with a rank-uniform gradient anchor for zero-window micro-batches",
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
    return_topk_indices: bool = False,
    build_block_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, None]:
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
            apply_rope=apply_rotary_pos_emb,
        )
        if compressed.shape[1] == 0:
            anchor = (self.kv_norm(kv[..., : self.head_dim]).sum() + gate.sum() + self.position_bias.sum()) * 0.0
            compressed = compressed + anchor.to(compressed.dtype)
        block_bias = packed_compressed_block_bias(rate_metadata) if build_block_bias else None
        result = (compressed.unsqueeze(1), block_bias)
        return (*result, None) if return_topk_indices else result

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
        result = (compressed_kv, None)
        return (*result, None) if return_topk_indices else result

    if build_block_bias:
        entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
        causal_threshold = (position_ids + 1) // self.compress_rate
        block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
        block_bias = block_bias.masked_fill(
            entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
            float("-inf"),
        )
    else:
        block_bias = None
    result = (compressed_kv, block_bias)
    return (*result, None) if return_topk_indices else result


@config.override_method(
    "DeepseekV4CSACompressor.forward",
    description="Keep CSA compression and indexing local to packed sequences, with a rank-uniform gradient anchor for zero-window micro-batches",
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
    return_topk_indices: bool = False,
    build_block_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
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
            apply_rope=apply_rotary_pos_emb,
        )
        # The indexer submodule is intentionally NOT anchored here: its outputs
        # are non-differentiable top-k indices, so its params already receive no
        # gradient on every rank uniformly, and anchoring them would create the
        # very asymmetry this patch removes.
        if compressed.shape[1] == 0:
            anchor = (self.kv_norm(kv[..., : self.head_dim]).sum() + gate.sum() + self.position_bias.sum()) * 0.0
            compressed = compressed + anchor.to(compressed.dtype)
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
        if build_block_bias:
            compressed_len = compressed_kv.shape[2]
            valid = top_k_indices >= 0
            safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
            block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
            block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
            block_bias = block_bias[..., :compressed_len]
        else:
            block_bias = None
        result = (compressed_kv, block_bias)
        return (*result, top_k_indices) if return_topk_indices else result

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
        compressed = self.kv_norm(
            (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype))
            .sum(dim=2, dtype=torch.float32)
            .to(new_kv.dtype)
        )
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
    if build_block_bias:
        compressed_len = compressed_kv.shape[2]
        valid = top_k_indices >= 0
        safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
        block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
        block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
        block_bias = block_bias[..., :compressed_len]
    else:
        block_bias = None
    result = (compressed_kv, block_bias)
    return (*result, top_k_indices) if return_topk_indices else result
