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
Patch configuration for DeepseekV4 NPU VeomniKernel replacements.

Regen command:
patchgen veomni.models_kernel.transformers.deepseek_v4.deepseek_v4_npu_patch_gen_config -o veomni/models_kernel/transformers/deepseek_v4/generated --diff

Reuses the GPU structural patches. NPU-only extras: shard ``position_bias``
on dim-1, and keep zero-window packed compressor grads attached.
"""

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4HCACache,
    apply_rotary_pos_emb,
)

from veomni.kernels import VeomniKernel
from veomni.models_kernel.transformers.deepseek_v4.packed_utils import (
    compress_packed_windows,
    packed_compressed_block_bias,
    shard_packed_compression_metadata,
)
from veomni.models_kernel.utils.kernel_utils import resolve_kernel_impl
from veomni.patchgen.patch_spec import PatchConfig

from .deepseek_v4_gpu_patch_gen_config import (
    PatchedDeepseekV4Experts,
    apply_rotary_pos_emb_patched,
    deepseek_v4_attention_forward_patched,
    deepseek_v4_attention_init_patched,
    deepseek_v4_decoder_layer_forward_patched,
    deepseek_v4_decoder_layer_init_patched,
    deepseek_v4_eager_attention_forward_patched,
    deepseek_v4_forcausallm_forward_patched,
    deepseek_v4_forcausallm_init_patched,
    deepseek_v4_get_parallel_plan_patched,
    deepseek_v4_hash_router_forward_patched,
    deepseek_v4_hyper_connection_forward_patched,
    deepseek_v4_hyper_connection_init_patched,
    deepseek_v4_hyper_head_forward_patched,
    deepseek_v4_hyper_head_init_patched,
    deepseek_v4_indexer_forward_patched,
    deepseek_v4_mlp_forward_patched,
    deepseek_v4_mlp_init_patched,
    deepseek_v4_model_forward_patched,
    deepseek_v4_rms_norm_forward_patched,
    deepseek_v4_rms_norm_init_patched,
    deepseek_v4_rotary_embedding_forward_patched,
    deepseek_v4_topk_router_forward_patched,
    deepseek_v4_unweighted_rmsnorm_forward_patched,
    deepseek_v4_unweighted_rmsnorm_init_patched,
)
from .deepseek_v4_gpu_patch_gen_config import (
    config as gpu_config,
)


# Names resolved at codegen time from generated imports.
get_parallel_state = None
all_gather_compressed_rows = None
empty_compressed_rows = None
exchange_compressor_halos = None
local_window_token_indices = None
plan_compressor_shard = None


config = PatchConfig(
    source_module="transformers.models.deepseek_v4.modeling_deepseek_v4",
    target_file="patched_modeling_deepseek_v4_npu.py",
    description="DeepseekV4 with VeomniKernel NPU replacements plus FSDP2 hardening",
)

config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
config.drop_imported_names.update(gpu_config.drop_imported_names)

config.override_method(
    "DeepseekV4RMSNorm.__init__",
    replacement=deepseek_v4_rms_norm_init_patched,
    description="Construct a local rms_norm VeomniKernel",
)
config.override_method(
    "DeepseekV4RMSNorm.forward",
    replacement=deepseek_v4_rms_norm_forward_patched,
    description="Always call the local rms_norm VeomniKernel",
)
config.override_method(
    "DeepseekV4UnweightedRMSNorm.__init__",
    replacement=deepseek_v4_unweighted_rmsnorm_init_patched,
    description="Construct a local unweighted rms_norm VeomniKernel",
)
config.override_method(
    "DeepseekV4UnweightedRMSNorm.forward",
    replacement=deepseek_v4_unweighted_rmsnorm_forward_patched,
    description="Always call the local unweighted rms_norm VeomniKernel",
)
config.override_method(
    "DeepseekV4RotaryEmbedding.forward",
    replacement=deepseek_v4_rotary_embedding_forward_patched,
    description="Retain FP32 cos/sin for inference and use activation dtype for checkpoint-stable training",
)
config.replace_function(
    "apply_rotary_pos_emb",
    replacement=apply_rotary_pos_emb_patched,
    description="Always call rope deepseek_v4 VeomniKernel",
)
config.override_method(
    "DeepseekV4MLP.__init__",
    replacement=deepseek_v4_mlp_init_patched,
    description="Construct a local swiglu_mlp VeomniKernel",
)
config.override_method(
    "DeepseekV4MLP.forward",
    replacement=deepseek_v4_mlp_forward_patched,
    description="Always call the local swiglu_mlp VeomniKernel",
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
    "DeepseekV4HyperConnection.__init__",
    replacement=deepseek_v4_hyper_connection_init_patched,
    description="Construct a local mhc pre VeomniKernel",
)
config.override_method(
    "DeepseekV4HyperConnection.forward",
    replacement=deepseek_v4_hyper_connection_forward_patched,
    description="Always call the local mhc pre VeomniKernel",
)
config.override_method(
    "DeepseekV4HyperHead.__init__",
    replacement=deepseek_v4_hyper_head_init_patched,
    description="Construct a local mhc head VeomniKernel",
)
config.override_method(
    "DeepseekV4HyperHead.forward",
    replacement=deepseek_v4_hyper_head_forward_patched,
    description="Always call the local mhc head VeomniKernel",
)
config.override_method(
    "DeepseekV4DecoderLayer.__init__",
    replacement=deepseek_v4_decoder_layer_init_patched,
    description="Construct a local mhc post VeomniKernel",
)
config.override_method(
    "DeepseekV4DecoderLayer.forward",
    replacement=deepseek_v4_decoder_layer_forward_patched,
    description="Always call the local mhc post VeomniKernel",
)
config.override_method(
    "DeepseekV4Indexer.forward",
    replacement=deepseek_v4_indexer_forward_patched,
    description="Always call the local dsa_indexer deepseek_v4 VeomniKernel",
)
config.override_method(
    "DeepseekV4Attention.__init__",
    replacement=deepseek_v4_attention_init_patched,
    description="Construct a local dsa_attention deepseek_v4 VeomniKernel",
)
config.override_method(
    "DeepseekV4Attention.forward",
    replacement=deepseek_v4_attention_forward_patched,
    description="Packed compressor path + Ulysses SP for DeepSeek-V4 sparse attention",
)
config.replace_function(
    "eager_attention_forward",
    description="Always call the local dsa_attention deepseek_v4 VeomniKernel",
)(deepseek_v4_eager_attention_forward_patched)
config.override_method(
    "DeepseekV4Model.forward",
    replacement=deepseek_v4_model_forward_patched,
    description="Packed boundaries, SP-aware full-sequence masks, stateless indexer dispatch",
)
config.replace_class(
    "DeepseekV4Experts",
    replacement=PatchedDeepseekV4Experts,
    description="Always call moe_experts VeomniKernel on v5 gate_up_proj weights",
)
config.override_method(
    "DeepseekV4ForCausalLM.__init__",
    replacement=deepseek_v4_forcausallm_init_patched,
    description="Bind ForCausalLMLoss and load_balancing_loss VeomniKernels",
)
config.override_method(
    "DeepseekV4ForCausalLM.forward",
    replacement=deepseek_v4_forcausallm_forward_patched,
    description="Always call ForCausalLMLoss and load_balancing_loss VeomniKernels",
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
    self.veomni_dsa_indexer = VeomniKernel(
        "dsa_indexer",
        "deepseek_v4",
        resolve_kernel_impl("dsa_indexer_implementation"),
    )


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
    # Accepted and ignored. ``DeepseekV4Attention.forward`` holds one compressor whose
    # class is chosen by layer type and calls it through a single call site, so the two
    # compressors have to take the same arguments; only the CSA one owns a Lightning
    # Indexer and so only it has anything to do with this. Defaulted, so an HCA
    # compressor called directly is unaffected.
    build_indexer_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, None]:
    if (packed_sequence_slices is None) != (packed_compression_metadata is None):
        raise ValueError("Packed sequence slices and compression metadata must be provided together")
    batch, _, _ = hidden_states.shape
    cache_layer: DeepseekV4HCACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    parallel_state = get_parallel_state()
    cp_enabled = parallel_state.cp_enabled and cache_layer is None
    if cp_enabled:
        cp_group = parallel_state.cp_group
        cp_rank = parallel_state.cp_rank
        local_seq_len = hidden_states.shape[1]
        rate = self.compress_rate
        shard = plan_compressor_shard(
            role="DeepSeek V4 HCA compressor",
            rate=rate,
            local_seq_len=local_seq_len,
            cp_rank=cp_rank,
            cp_size=parallel_state.cp_size,
            packed_compression_metadata=packed_compression_metadata,
            device=kv.device,
        )
        kv, gate = exchange_compressor_halos(kv, gate, rate, cp_group)

    if cache_layer is None and packed_sequence_slices is not None and packed_compression_metadata is not None:
        rate_metadata = packed_compression_metadata[self.compress_rate]
        if cp_enabled:
            rate_metadata = shard_packed_compression_metadata(
                rate_metadata,
                window_begin=shard.begin,
                window_end=shard.end,
                local_seq_len=local_seq_len,
                cp_rank=cp_rank,
                halo=rate,
            )
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
        if cp_enabled:
            compressed = all_gather_compressed_rows(compressed, shard.counts, cp_group)
        block_bias = packed_compressed_block_bias(rate_metadata) if build_block_bias else None
        result = (compressed.unsqueeze(1), block_bias)
        return (*result, None) if return_topk_indices else result

    if cp_enabled:
        window_indices, first_window_position = local_window_token_indices(
            shard, rate=rate, local_seq_len=local_seq_len, cp_rank=cp_rank, device=kv.device
        )
        flat_indices = window_indices.reshape(-1)
        chunk_kv, chunk_gate = kv[:, flat_indices], gate[:, flat_indices]
    elif cache_layer is None:
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
        compressed = (
            empty_compressed_rows(chunk_kv, chunk_gate, self.head_dim)
            if cp_enabled
            else chunk_kv.new_zeros((batch, 0, self.head_dim))
        )

    if cache_layer is not None:
        compressed = cache_layer.update_compressor_states("compressor", compressed)
    if cp_enabled:
        compressed = all_gather_compressed_rows(compressed, shard.counts, cp_group)
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
    # Accepted, refused, and never forwarded on this backend. The shared attention
    # forward passes ``_builds_indexer_kl``'s answer down here, so the parameter
    # exists because the call site is shared. The two indexer call sites below stay
    # on their bare-tensor return.
    build_indexer_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if (packed_sequence_slices is None) != (packed_compression_metadata is None):
        raise ValueError("Packed sequence slices and compression metadata must be provided together")
    if build_indexer_loss:
        raise NotImplementedError(
            "dsa_indexer_loss is not implemented on NPU: the objective's student "
            "distribution is the TileLang Lightning Indexer's per-slot scores, and that "
            "kernel is CUDA-only. Set dsa_indexer_loss: false under model.model_config."
        )
    batch, seq_len, _ = hidden_states.shape
    cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    parallel_state = get_parallel_state()
    cp_enabled = parallel_state.cp_enabled and cache_layer is None
    if cp_enabled:
        cp_group = parallel_state.cp_group
        cp_rank = parallel_state.cp_rank
        local_seq_len = hidden_states.shape[1]
        rate = self.compress_rate
        shard = plan_compressor_shard(
            role="DeepSeek V4 CSA compressor",
            rate=rate,
            local_seq_len=local_seq_len,
            cp_rank=cp_rank,
            cp_size=parallel_state.cp_size,
            packed_compression_metadata=packed_compression_metadata,
            device=kv.device,
        )
        kv, gate = exchange_compressor_halos(kv, gate, rate, cp_group)

    if cache_layer is None and packed_sequence_slices is not None and packed_compression_metadata is not None:
        rate_metadata = packed_compression_metadata[self.compress_rate]
        if cp_enabled:
            rate_metadata = shard_packed_compression_metadata(
                rate_metadata,
                window_begin=shard.begin,
                window_end=shard.end,
                local_seq_len=local_seq_len,
                cp_rank=cp_rank,
                halo=rate,
            )
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
        if cp_enabled:
            compressed = all_gather_compressed_rows(compressed, shard.counts, cp_group)
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

    prior_kv = prior_gate = None
    if cp_enabled:
        window_indices, first_window_position = local_window_token_indices(
            shard, rate=rate, local_seq_len=local_seq_len, cp_rank=cp_rank, device=kv.device
        )
        flat_indices = window_indices.reshape(-1)
        chunk_kv, chunk_gate = kv[:, flat_indices], gate[:, flat_indices]
        if first_window_position >= rate:
            previous_indices = window_indices[0] - rate
            prior_kv = kv[:, previous_indices, : self.head_dim]
            prior_gate = gate[:, previous_indices, : self.head_dim] + self.position_bias[:, : self.head_dim].to(
                gate.dtype
            )
    elif cache_layer is None:
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
        compressed = (
            empty_compressed_rows(chunk_kv, chunk_gate, self.head_dim)
            if cp_enabled
            else chunk_kv.new_zeros((batch, 0, self.head_dim))
        )

    if cache_layer is not None:
        compressed = cache_layer.update_compressor_states("compressor", compressed)
    if cp_enabled:
        compressed = all_gather_compressed_rows(compressed, shard.counts, cp_group)
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
