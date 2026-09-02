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
Patch configuration for DeepseekV4 GPU VeomniKernel replacements.

Regen command:
patchgen veomni.models_kernel.transformers.deepseek_v4.deepseek_v4_gpu_patch_gen_config -o veomni/models_kernel/transformers/deepseek_v4/generated --diff

RMS, unweighted RMS, SwiGLU, routed experts, mHC, DSA indexer / attention,
apply-RoPE, CausalLM, and load-balancing always call local VeomniKernel
handles. Packed compressors, Ulysses SP, FP32 routers, and rotary table
dtype stay as structural patches.
"""

from functools import partial
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
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.kernels import VeomniKernel
from veomni.models_kernel.transformers.deepseek_v4.packed_utils import (
    CompressedCandidates,
    build_packed_compression_metadata,
    build_packed_sparse_attention_indices,
    build_sparse_attention_indices,
    compress_packed_windows,
    isolate_packed_causal_mask_,
    mask_sparse_attention_indices,
    packed_compressed_block_bias,
    packed_compressed_causal_ranges,
    scatter_topk_block_bias,
)
from veomni.models_kernel.utils.kernel_utils import (
    empty_bias,
    linear_bias,
    resolve_kernel_impl,
    resolve_moe_impl,
)
from veomni.models_kernel.utils.loss_utils import ForCausalLMLoss
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import MoeCausalLMOutputWithLogProbs
from veomni.utils.moe_router_replay import get_active_replay, maybe_replay_indices


# Names resolved at codegen time from generated imports.
get_parallel_state = None
gather_seq_scatter_heads = None
gather_heads_scatter_seq = None
gather_outputs = None


config = PatchConfig(
    source_module="transformers.models.deepseek_v4.modeling_deepseek_v4",
    target_file="patched_modeling_deepseek_v4_gpu.py",
    description="DeepseekV4 with VeomniKernel RMS / RoPE / SwiGLU / MoE / mHC / DSA / fused loss",
)

config.add_import("functools", names=["partial"])
config.add_import("veomni.kernels", names=["VeomniKernel"])
config.add_import(
    "veomni.models_kernel.utils.kernel_utils",
    names=["empty_bias", "linear_bias", "resolve_kernel_impl", "resolve_moe_impl"],
)
config.add_import(
    "veomni.models_kernel.utils.loss_utils",
    names=["ForCausalLMLoss"],
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
    "veomni.models_kernel.transformers.deepseek_v4.packed_utils",
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
        "scatter_topk_block_bias",
    ],
)
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "MoeCausalLMOutputWithLogProbs"],
)
config.drop_import_names("MoeCausalLMOutputWithPast")
config.add_import(
    "veomni.utils.moe_router_replay",
    names=["get_active_replay", "maybe_replay_indices"],
)


# ================================================================
# Patch: DeepSeek V4 RMSNorm
# ================================================================
@config.override_method(
    "DeepseekV4RMSNorm.__init__",
    description="Construct a local rms_norm VeomniKernel",
)
def deepseek_v4_rms_norm_init_patched(self, hidden_size, eps: float = 1e-6) -> None:
    nn.Module.__init__(self)
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.veomni_rms_norm = VeomniKernel("rms_norm", "standard", resolve_kernel_impl("rms_norm_implementation"))


@config.override_method(
    "DeepseekV4RMSNorm.forward",
    description="Always call the local rms_norm VeomniKernel",
)
def deepseek_v4_rms_norm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)


@config.override_method(
    "DeepseekV4UnweightedRMSNorm.__init__",
    description="Construct a local unweighted rms_norm VeomniKernel",
)
def deepseek_v4_unweighted_rmsnorm_init_patched(self, eps: float = 1.0e-6) -> None:
    nn.Module.__init__(self)
    self.eps = eps
    impl = resolve_kernel_impl("rms_norm_implementation")
    if impl in {"npu", "triton"}:
        impl = "eager"
    self.veomni_unweighted_rms_norm = VeomniKernel("rms_norm", "unweighted", impl)


@config.override_method(
    "DeepseekV4UnweightedRMSNorm.forward",
    description="Always call the local unweighted rms_norm VeomniKernel",
)
def deepseek_v4_unweighted_rmsnorm_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    return self.veomni_unweighted_rms_norm(x, eps=self.eps)


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


@config.replace_function(
    "apply_rotary_pos_emb",
    description="Always call rope deepseek_v4 VeomniKernel",
)
def apply_rotary_pos_emb_patched(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    impl = resolve_kernel_impl("rotary_pos_emb_implementation")
    if impl in {"npu", "liger_kernel"}:
        impl = "eager"
    rope = VeomniKernel("rope", "deepseek_v4", impl)
    return rope(x, cos, sin, unsqueeze_dim=unsqueeze_dim)


# ================================================================
# Patch: mHC always-call
# ================================================================
@config.override_method(
    "DeepseekV4HyperConnection.__init__",
    description="Construct a local mhc pre VeomniKernel",
)
def deepseek_v4_hyper_connection_init_patched(self, config: "DeepseekV4Config"):
    nn.Module.__init__(self)
    self.hc_mult = config.hc_mult
    self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
    self.hc_eps = config.hc_eps
    self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
    mix = (2 + self.hc_mult) * self.hc_mult
    self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
    self.base = nn.Parameter(torch.empty(mix))
    self.scale = nn.Parameter(torch.empty(3))
    self.veomni_mhc_pre = VeomniKernel("mhc", "pre", resolve_kernel_impl("mhc_implementation"))


@config.override_method(
    "DeepseekV4HyperConnection.forward",
    description="Always call the local mhc pre VeomniKernel",
)
def deepseek_v4_hyper_connection_forward_patched(
    self,
    hidden_streams: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return self.veomni_mhc_pre(
        hidden_streams,
        self.fn,
        self.scale,
        self.base,
        self.input_norm.eps,
        self.hc_mult,
        self.hc_sinkhorn_iters,
        self.hc_eps,
    )


@config.override_method(
    "DeepseekV4HyperHead.__init__",
    description="Construct a local mhc head VeomniKernel",
)
def deepseek_v4_hyper_head_init_patched(self, config: "DeepseekV4Config"):
    nn.Module.__init__(self)
    self.hc_mult = config.hc_mult
    self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
    self.eps = config.hc_eps
    self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
    self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
    self.hc_scale = nn.Parameter(torch.empty(1))
    self.veomni_mhc_head = VeomniKernel("mhc", "head", resolve_kernel_impl("mhc_implementation"))


@config.override_method(
    "DeepseekV4HyperHead.forward",
    description="Always call the local mhc head VeomniKernel",
)
def deepseek_v4_hyper_head_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    return self.veomni_mhc_head(
        x,
        self.hc_fn,
        self.hc_scale,
        self.hc_base,
        self.input_norm.eps,
        self.hc_mult,
        self.eps,
    )


@config.override_method(
    "DeepseekV4DecoderLayer.__init__",
    description="Construct a local mhc post VeomniKernel",
)
def deepseek_v4_decoder_layer_init_patched(self, config: "DeepseekV4Config", layer_idx: int):
    super().__init__()
    self.layer_idx = layer_idx
    self.self_attn = DeepseekV4Attention(config, layer_idx)
    self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
    self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.attn_hc = DeepseekV4HyperConnection(config)
    self.ffn_hc = DeepseekV4HyperConnection(config)
    self.veomni_mhc_post = VeomniKernel("mhc", "post", resolve_kernel_impl("mhc_implementation"))


@config.override_method(
    "DeepseekV4DecoderLayer.forward",
    description="Always call the local mhc post VeomniKernel",
)
def deepseek_v4_decoder_layer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> torch.Tensor:
    post, comb, collapsed = self.attn_hc(hidden_states)
    attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
    hidden_states = self.veomni_mhc_post(attn_output, hidden_states, post, comb)

    post, comb, collapsed = self.ffn_hc(hidden_states)
    mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
    return self.veomni_mhc_post(mlp_output, hidden_states, post, comb)


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
    return_topk_indices: bool = False,
    build_block_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, CompressedCandidates]:
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
        compressed_kv = compressed.unsqueeze(1)
        candidates = CompressedCandidates(
            range_starts=rate_metadata["range_starts"],
            range_ends=rate_metadata["range_ends"],
        )
        block_bias = packed_compressed_block_bias(rate_metadata) if build_block_bias else None
        return (compressed_kv, block_bias, candidates) if return_topk_indices else (compressed_kv, block_bias)

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
        # `sum` follows autocast's fp32_set_opt_dtype policy: an implicit `dtype`
        # returns fp32 under autocast and leaks through `kv_norm` into the
        # bf16-only TileLang kernels. Accumulate in fp32 explicitly, cast back.
        compressed = self.kv_norm(
            (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype))
            .sum(dim=2, dtype=torch.float32)
            .to(chunk_kv.dtype)
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
        return (*result, CompressedCandidates()) if return_topk_indices else result

    causal_threshold = (position_ids + 1) // self.compress_rate
    candidates = CompressedCandidates(
        range_starts=torch.zeros_like(causal_threshold, dtype=torch.int32),
        range_ends=causal_threshold.to(torch.int32),
    )
    block_bias = None
    if build_block_bias:
        entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
        block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
        block_bias = block_bias.masked_fill(
            entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
            float("-inf"),
        )
    return (compressed_kv, block_bias, candidates) if return_topk_indices else (compressed_kv, block_bias)


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
    return_topk_indices: bool = False,
    build_block_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, CompressedCandidates]:
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
        candidates = CompressedCandidates(topk_indices=top_k_indices)
        block_bias = (
            scatter_topk_block_bias(compressed_kv, top_k_indices, batch, seq_len) if build_block_bias else None
        )
        return (compressed_kv, block_bias, candidates) if return_topk_indices else (compressed_kv, block_bias)

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
        # See the HCA compressor above: `sum` needs an explicit `dtype` under autocast.
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
    candidates = CompressedCandidates(topk_indices=top_k_indices)
    block_bias = scatter_topk_block_bias(compressed_kv, top_k_indices, batch, seq_len) if build_block_bias else None
    return (compressed_kv, block_bias, candidates) if return_topk_indices else (compressed_kv, block_bias)


# ================================================================
# Patch: DeepseekV4Indexer.forward
# 1. Dispatch CUDA prefill/training index scoring to the TileLang Lightning
#    Indexer when ``dsa_indexer_implementation=tilelang``. Cache/decode and unusual
#    position layouts retain the upstream eager implementation.
# ================================================================
@config.override_method(
    "DeepseekV4Indexer.__init__",
    description="Construct a local dsa_indexer deepseek_v4 VeomniKernel",
)
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
    self.veomni_dsa_indexer = VeomniKernel(
        "dsa_indexer",
        "deepseek_v4",
        resolve_kernel_impl("dsa_indexer_implementation"),
    )


@config.override_method(
    "DeepseekV4Indexer.forward", description="Always call the local dsa_indexer deepseek_v4 VeomniKernel"
)
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

        # See the HCA compressor above: `sum` needs an explicit `dtype` under autocast.
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
    top_k = min(self.index_topk, max(compressed_len, 1))

    packed_ranges = None
    if packed_compression_metadata is not None and cache_layer is None:
        packed_ranges = packed_compressed_causal_ranges(packed_compression_metadata[self.compress_rate])
    query = q.transpose(0, 1).contiguous()
    query_weights = weights.transpose(0, 1).contiguous()
    query_range_starts = None if packed_ranges is None else packed_ranges[0]
    query_range_ends = None if packed_ranges is None else packed_ranges[1]
    if query_range_starts is None:
        canonical_positions = torch.arange(seq_len, device=position_ids.device).unsqueeze(0).expand_as(position_ids)
        if not torch.equal(position_ids, canonical_positions):
            query_range_starts = torch.zeros(seq_len, device=q.device, dtype=torch.int32)
            query_range_ends = ((position_ids[0] + 1) // self.compress_rate).to(torch.int32)
    parallel_state = get_parallel_state()
    if parallel_state.ulysses_enabled:
        if query_range_starts is None and query_range_ends is None:
            query_range_starts = torch.zeros(seq_len, device=q.device, dtype=torch.int32)
            query_positions = torch.arange(seq_len, device=q.device, dtype=torch.int32)
            query_range_ends = (query_positions + 1) // self.compress_rate
        if seq_len % parallel_state.ulysses_size != 0:
            raise ValueError(
                f"DeepSeek-V4 indexer sequence length ({seq_len}) must be divisible by "
                f"Ulysses size ({parallel_state.ulysses_size})"
            )
        local_seq_len = seq_len // parallel_state.ulysses_size
        query_start = parallel_state.ulysses_rank * local_seq_len
        query_end = query_start + local_seq_len
        query = query[query_start:query_end]
        query_weights = query_weights[query_start:query_end]
        if query_range_starts is not None and query_range_ends is not None:
            query_range_starts = query_range_starts[query_start:query_end]
            query_range_ends = query_range_ends[query_start:query_end]

    _, top_k_indices = self.veomni_dsa_indexer(
        query,
        compressed_kv.transpose(0, 1).contiguous(),
        query_weights,
        self.compress_rate,
        top_k,
        cu_seqlen_ks=query_range_starts,
        cu_seqlen_ke=query_range_ends,
    )
    if parallel_state.ulysses_enabled:
        top_k_indices = gather_outputs(
            top_k_indices,
            gather_dim=1,
            group=parallel_state.ulysses_group,
        )
    return top_k_indices.to(torch.long)


# ================================================================
# Patch: DeepseekV4Attention
# 1. Pass the collator-provided packed sequence slices into compressors.
# 2. Ulysses SP: all-to-all Q heads, sequence all-gather for MQA KV and
#    compressor inputs (windows/indexers need the full sequence), then
#    scatter attention outputs back to the local sequence shard.
# ================================================================
@config.override_method(
    "DeepseekV4Attention.__init__",
    description="Construct a local dsa_attention deepseek_v4 VeomniKernel",
)
def deepseek_v4_attention_init_patched(self, config: "DeepseekV4Config", layer_idx: int):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.layer_type = config.layer_types[layer_idx]
    self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"
    self.num_heads = config.num_attention_heads
    self.num_key_value_groups = config.num_attention_heads
    self.head_dim = config.head_dim
    self.sliding_window = config.sliding_window
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.scaling = self.head_dim**-0.5

    self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
    self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
    self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
    self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
    self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
    self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.o_a_proj = DeepseekV4GroupedLinear(
        self.num_heads * self.head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
    )
    self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
    self.sinks = nn.Parameter(torch.empty(self.num_heads))
    self.compressor = COMPRESSOR_CLASSES[self.layer_type](config) if self.layer_type != "sliding_attention" else None
    self.veomni_dsa_attention = VeomniKernel(
        "dsa_attention",
        "deepseek_v4",
        resolve_kernel_impl("dsa_attention_implementation"),
    )


@config.override_method(
    "DeepseekV4Attention.forward",
    description="Packed compressor path + Ulysses SP for DeepSeek-V4 sparse attention",
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
    compressed_candidates = None
    # The device and dtype terms mirror what ``eager_attention_forward`` requires
    # before it can dispatch to TileLang. Without them this reads the config string
    # alone and claims the compact path on hosts where the kernel cannot run and the
    # dispatch silently falls back to eager -- which then ignores the indices and
    # uses the dense mask, so the compact work is wasted at best.
    use_compact_sparse_indices = (
        self.veomni_dsa_attention.impl == "tilelang"
        and past_key_values is None
        and q.is_cuda
        and q.dtype == torch.bfloat16
    )
    # ``DeepseekV4Model.forward`` withholds the dense mask exactly when the packed
    # metadata is sufficient to validate candidates on its own, so its absence is
    # the signal to take the mask-free path and skip every O(S^2) intermediate.
    mask_free_sparse = use_compact_sparse_indices and attention_mask is None
    if self.compressor is not None:
        compressor_output = self.compressor(
            compressor_hidden,
            compressor_q_residual,
            compressor_position_ids,
            past_key_values,
            self.layer_idx,
            packed_sequence_slices=kwargs.get("packed_sequence_slices"),
            packed_compression_metadata=kwargs.get("packed_compression_metadata"),
            return_topk_indices=use_compact_sparse_indices,
            build_block_bias=not mask_free_sparse,
        )
        if use_compact_sparse_indices:
            compressed_kv, block_bias, compressed_candidates = compressor_output
        else:
            compressed_kv, block_bias = compressor_output
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
    if mask_free_sparse:
        kwargs["sparse_topk_indices"] = build_packed_sparse_attention_indices(
            position_ids=compressor_position_ids,
            sliding_window=self.sliding_window,
            compressed_len=kv.shape[-2] - q.shape[-2],
            candidates=compressed_candidates,
        )
    elif use_compact_sparse_indices:
        kwargs["sparse_topk_indices"] = build_sparse_attention_indices(
            batch_size=q.shape[0],
            seq_len=q.shape[-2],
            sliding_window=self.sliding_window,
            compressed_len=kv.shape[-2] - q.shape[-2],
            compressed_indices=compressed_candidates.topk_indices if compressed_candidates is not None else None,
            device=q.device,
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
# Always call the local dsa_attention deepseek_v4 VeomniKernel. Convert a
# dense additive mask into compact top-k indices when the caller did not
# already provide them.
# ================================================================
@config.replace_function(
    "eager_attention_forward", description="Always call the local dsa_attention deepseek_v4 VeomniKernel"
)
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
    del value, dropout
    topk_indices = kwargs.get("sparse_topk_indices")
    if topk_indices is None:
        batch, _, seq_len, _ = query.shape
        kv_len = key.shape[-2]
        if attention_mask is None:
            topk_indices = (
                torch.arange(kv_len, device=query.device, dtype=torch.int32)
                .view(1, 1, -1)
                .expand(batch, seq_len, -1)
                .contiguous()
            )
        else:
            mask = attention_mask
            if mask.shape[0] == 1 and batch > 1:
                mask = mask.expand(batch, -1, -1, -1)
            allowed = mask[:, 0] if mask.dtype == torch.bool else mask[:, 0] >= 0
            _, topk_indices = allowed.to(torch.int8).topk(kv_len, dim=-1, sorted=False)
            selected_valid = allowed.gather(-1, topk_indices)
            topk_indices = topk_indices.to(torch.int32).masked_fill(~selected_valid, -1).contiguous()
    elif attention_mask is not None:
        topk_indices = mask_sparse_attention_indices(attention_mask, topk_indices)
    sinks = kwargs.get("s_aux", module.sinks)
    attn_output = module.veomni_dsa_attention(
        query.transpose(1, 2).contiguous(),
        key[:, 0].contiguous(),
        sinks,
        topk_indices,
        sm_scale=scaling,
    )
    return attn_output, None


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

    # The TileLang sparse kernel reads a compact candidate list, and packed
    # metadata already pins down every constraint a dense mask would encode, so
    # the O(S^2) mask and block bias are skipped entirely on that path.
    mask_free_sparse = False

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
        # Packed training disables the cache below, so TileLang attention is the
        # only mask consumer left and it can validate candidates on its own.
        # ``eager_attention_forward`` declines the TileLang dispatch for non-bf16
        # or host tensors, and its dense fallback needs the mask to stay causal,
        # so mirror those two runtime conditions before dropping the mask.
        mask_free_sparse = (
            resolve_kernel_impl("dsa_attention_implementation") == "tilelang"
            and not isinstance(attention_mask, dict)
            and inputs_embeds.dtype == torch.bfloat16
            and inputs_embeds.is_cuda
        )
        # Dropping the mask is only sound if it masked nothing out. The check on
        # ``boundaries`` above already establishes that every position belongs to
        # some sequence, so a zero here contradicts the caller's own cu-seqlens --
        # but ``build_packed_sparse_attention_indices`` rebuilds candidates from
        # ``position_ids`` alone, so an unnoticed zero would silently make a padded
        # token attendable and move the loss. VeOmni's collator guarantees all-ones
        # on this path (see ``data_collator.py``: SP slices ``input_ids`` but keeps
        # the full mask), yet this is a public entry point, so verify rather than
        # trust. Reading the mask costs one device sync on a branch that already
        # pays for ``cu_seq_lens_q.cpu()`` a few lines up, so this adds no new
        # class of stall.
        if mask_free_sparse and isinstance(attention_mask, torch.Tensor) and not bool(attention_mask.all()):
            raise ValueError(
                "DeepSeek V4 packed attention received an attention_mask with masked-out "
                "positions alongside cu_seq_lens_q that span the full sequence. Express "
                "padding through cu_seq_lens_q, which the sparse path reads, instead of a "
                "dense mask, which it drops."
            )
        # Metadata is indexed by global positions / cu-seqlens; under SP the
        # collator already provides full-sequence cu-seqlens while local embeds
        # are only one shard, so materialize a full-length reference tensor.
        metadata_reference = inputs_embeds.new_empty(inputs_embeds.shape[0], full_seq_len, inputs_embeds.shape[-1])
        kwargs["packed_compression_metadata"] = build_packed_compression_metadata(
            metadata_reference,
            full_position_ids,
            packed_sequence_slices,
            compress_rates,
            block_bias_rates=() if mask_free_sparse else (hca_rate,),
        )
        # Packed training combines independent samples in one physical row;
        # treating that row as a decode cache would merge their KV histories.
        past_key_values = None
        return_cache = None

    if mask_free_sparse:
        causal_mask = None
    elif isinstance(attention_mask, dict):
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
    if causal_mask is not None and "packed_sequence_slices" in kwargs:
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
# 2. Always call moe_experts with the stacked ``gate_up_proj`` layout and
#    V4's gpt-oss-style ``swiglu_limit`` clamp.
# Layout matches v5 upstream (direct, no transpose):
#   gate_up_proj [E, 2*I, H],  down_proj [E, H, I]
# ================================================================
@config.replace_class(
    "DeepseekV4Experts",
    description="Always call moe_experts VeomniKernel on v5 gate_up_proj weights",
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
        self.veomni_moe = VeomniKernel("moe_experts", "standard", resolve_moe_impl())

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        unused = empty_bias(self.gate_up_proj)
        return self.veomni_moe(
            hidden_states,
            top_k_weights.to(hidden_states.dtype),
            top_k_index,
            unused,
            unused,
            self.down_proj,
            self.gate_up_proj,
            num_experts=self.num_experts,
            swiglu_limit=self.limit,
        )


# ================================================================
# Patch: DeepseekV4MLP — shared experts. HuggingFace's MLP has no clamp;
# the kernel is still called so a later swiglu impl swap stays local.
# ================================================================
@config.override_method(
    "DeepseekV4MLP.__init__",
    description="Construct a local swiglu_mlp VeomniKernel",
)
def deepseek_v4_mlp_init_patched(self, config):
    nn.Module.__init__(self)
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
    self.act_fn = ACT2FN[config.hidden_act]
    self.veomni_swiglu_mlp = VeomniKernel("swiglu_mlp", "standard", resolve_kernel_impl("swiglu_mlp_implementation"))


@config.override_method(
    "DeepseekV4MLP.forward",
    description="Always call the local swiglu_mlp VeomniKernel",
)
def deepseek_v4_mlp_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    return self.veomni_swiglu_mlp(
        x,
        self.gate_proj.weight,
        linear_bias(self.gate_proj),
        self.up_proj.weight,
        linear_bias(self.up_proj),
        self.down_proj.weight,
        linear_bias(self.down_proj),
    )


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
    if get_active_replay() is not None:
        indices = maybe_replay_indices(self, scores, indices)
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
    if get_active_replay() is not None:
        indices = maybe_replay_indices(self, scores, indices)
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


# ================================================================
# Patch: DeepseekV4ForCausalLM
# ================================================================
@config.override_method(
    "DeepseekV4ForCausalLM.__init__",
    description="Bind ForCausalLMLoss and load_balancing_loss VeomniKernels",
)
def deepseek_v4_forcausallm_init_patched(self, config):
    super().__init__(config)
    self.model = DeepseekV4Model(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.router_aux_loss_coef = config.router_aux_loss_coef
    self.num_experts = config.num_local_experts
    self.num_experts_per_tok = config.num_experts_per_tok
    impl = resolve_kernel_impl("cross_entropy_loss_implementation", npu_as="chunk_loss")
    self.veomni_ce = VeomniKernel("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, kernel=self.veomni_ce)
    self.veomni_lb = VeomniKernel(
        "load_balancing_loss",
        "standard",
        resolve_kernel_impl("load_balancing_loss_implementation"),
    )
    self.post_init()


@config.override_method(
    "DeepseekV4ForCausalLM.forward",
    description="Always call ForCausalLMLoss and load_balancing_loss VeomniKernels",
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

    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        loss, logits, fused_linear_aux = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            hidden_states=hidden_states,
            weights=self.lm_head.weight,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states)

    aux_loss = None
    if output_router_logits:
        router_logits = outputs.router_logits
        if router_logits is None or not isinstance(router_logits, tuple):
            aux_loss = 0
        else:
            gate = torch.cat([layer.reshape(-1, layer.shape[-1]) for layer in router_logits], dim=0)
            mask = attention_mask if isinstance(attention_mask, torch.Tensor) else gate.new_empty(0)
            aux_loss = self.veomni_lb(gate, mask, top_k=self.num_experts_per_tok)
        if labels is not None and isinstance(aux_loss, torch.Tensor):
            loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

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
