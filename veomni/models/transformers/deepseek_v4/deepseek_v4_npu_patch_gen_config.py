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
from torch import nn
from transformers.cache_utils import Cache
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
    deepseek_v4_attention_forward_patched,
    deepseek_v4_decoder_layer_forward_patched,
    deepseek_v4_eager_attention_forward_patched,
    deepseek_v4_forcausallm_forward_patched,
    deepseek_v4_get_parallel_plan_patched,
    deepseek_v4_hash_router_forward_patched,
    deepseek_v4_hyper_connection_forward_patched,
    deepseek_v4_hyper_head_forward_patched,
    deepseek_v4_indexer_forward_patched,
    deepseek_v4_mlp_forward_patched,
    deepseek_v4_model_forward_patched,
    deepseek_v4_rms_norm_forward_patched,
    deepseek_v4_rotary_embedding_forward_patched,
    deepseek_v4_topk_router_forward_patched,
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
        "build_packed_compression_metadata",
        "compress_packed_windows",
        "isolate_packed_causal_mask_",
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
    replacement=deepseek_v4_topk_router_forward_patched,
    description="Match the official DeepSeek-V4 FP32 router projection",
)

config.override_method(
    "DeepseekV4HashRouter.forward",
    replacement=deepseek_v4_hash_router_forward_patched,
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
    replacement=deepseek_v4_indexer_forward_patched,
    description="Optional TileLang Lightning Indexer dispatch (no-ops to eager on NPU)",
)

config.override_method(
    "DeepseekV4Attention.forward",
    replacement=deepseek_v4_attention_forward_patched,
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
)(deepseek_v4_eager_attention_forward_patched)

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
        if compressed.shape[1] == 0:
            anchor = (self.kv_norm(kv[..., : self.head_dim]).sum() + gate.sum() + self.position_bias.sum()) * 0.0
            compressed = compressed + anchor.to(compressed.dtype)
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
