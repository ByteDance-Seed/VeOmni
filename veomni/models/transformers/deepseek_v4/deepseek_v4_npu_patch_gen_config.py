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
dispatch, mHC dispatch, packed attention, model forward, fused-MoE
experts, fused-CE ForCausalLM.forward, parallel plan) by import rather than
duplication, mirroring the ``deepseek_v3`` GPU/NPU pair. This is safe rather
than merely convenient:

- ``DeepseekV4RMSNorm.forward`` / ``DeepseekV4UnweightedRMSNorm.forward`` /
  ``DeepseekV4MLP.forward`` dispatch to Liger kernels only when their OpSlot
  is bound to a non-eager implementation; Liger requires CUDA, so these fall
  straight through to the shared eager arithmetic on NPU without any change
  needed here.
- ``eager_attention_forward`` gates its
  TileLang fast paths behind ``.is_cuda`` (and SM90 checks inside
  ``veomni.ops.kernels.deepseek_v4``). The module-level import of
  ``sparse_attn_tilelang`` / ``v4_lighting_indexer`` is lazy-safe on NPU:
  ``veomni/ops/kernels/deepseek_v4/__init__.py`` only imports TileLang inside
  the wrapper *bodies*, guarded by ``_require_tilelang_sm90()``, which is
  never reached because the ``.is_cuda`` condition short-circuits first. It
  falls straight through to the eager PyTorch computation on NPU.
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
3. ``DeepseekV4Indexer.forward`` — dispatch the CANN Lightning Indexer under
   the NPU-only execution contract while retaining eager/TileLang compatibility paths.

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
    _builds_indexer_kl,
    _indexer_loss_enabled,
    _split_indexer_output,
    deepseek_v4_attention_forward_patched,
    deepseek_v4_decoder_layer_forward_patched,
    deepseek_v4_eager_attention_forward_patched,
    deepseek_v4_forcausallm_forward_patched,
    deepseek_v4_get_parallel_plan_patched,
    deepseek_v4_hash_router_forward_patched,
    deepseek_v4_hyper_connection_forward_patched,
    deepseek_v4_hyper_head_forward_patched,
    deepseek_v4_mlp_forward_patched,
    deepseek_v4_model_forward_patched,
    deepseek_v4_rms_norm_forward_patched,
    deepseek_v4_rotary_embedding_forward_patched,
    deepseek_v4_sparse_moe_block_init_patched,
    deepseek_v4_topk_router_forward_patched,
    deepseek_v4_unweighted_rmsnorm_forward_patched,
    indexer_kl_terms,
    veomni_qat_fake_quant_act,
    veomni_qat_fake_quant_expert_weight,
    veomni_qat_fake_quant_kv,
    veomni_qat_linear,
)


def deepseek_v4_indexer_forward_npu_patched(
    self,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: Cache | None,
    layer_idx: int,
    packed_sequence_slices: tuple[tuple[int, int], ...] | None = None,
    packed_compression_metadata: dict[int, dict[str, torch.Tensor]] | None = None,
    build_indexer_loss: bool = False,
) -> torch.LongTensor | tuple[torch.LongTensor, torch.Tensor]:
    if (packed_sequence_slices is None) != (packed_compression_metadata is None):
        raise ValueError("Packed sequence slices and compression metadata must be provided together")

    # --- Patch.2 ---
    # The indexer trains on its own KL alone (DeepSeek-V3.2 §2.1: "we detach the
    # indexer input from the computational graph for separate optimization"). Until
    # the scores started coming back out of here the graph was severed only by
    # accident, because this forward returned integer indices, which carry no
    # gradient; from here on this detach is the only thing keeping the auxiliary
    # objective from reaching the language-modelling one.
    #
    # ``build_indexer_loss`` arrives from ``DeepseekV4Attention.forward``, which owns
    # the model config and evaluated ``_builds_indexer_kl`` once for this layer. This
    # module keeps only scalars off the config it was constructed with, and deriving
    # the answer a second time here is what would let the detach, the return arity and
    # the compressor's unpacking disagree inside a single call.
    if build_indexer_loss:
        hidden_states = hidden_states.detach()
        q_residual = q_residual.detach()
    # --- Patch.2 ---

    batch, seq_len, _ = hidden_states.shape
    cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    # --- Patch.2 ---
    # Under context parallelism the queries arrive already sharded, but a top-k
    # value names a slot in the enclosing CSA compressor's compressed KV, which is
    # replicated. So the compressed *keys* have to stay global, and the indexer
    # runs the same own-your-windows-then-all-gather compression its compressor
    # does -- it cannot reuse that result, because it summarises the same windows
    # through its own projections at ``index_head_dim``. Only the query axis is
    # local, and ``query_offset`` is what keeps a local query row addressing its
    # absolute position.
    parallel_state = get_parallel_state()
    cp_enabled = parallel_state.cp_enabled and cache_layer is None
    query_offset = 0
    if cp_enabled:
        cp_group = parallel_state.cp_group
        cp_rank = parallel_state.cp_rank
        local_seq_len = seq_len
        rate = self.compress_rate
        query_offset = cp_rank * local_seq_len
        shard = plan_compressor_shard(
            role="DeepSeek V4 Lightning Indexer",
            rate=rate,
            local_seq_len=local_seq_len,
            cp_rank=cp_rank,
            cp_size=parallel_state.cp_size,
            packed_compression_metadata=packed_compression_metadata,
            device=kv.device,
        )
        # Every guard is above this line, so no rank enters a collective while its
        # peers are still deciding whether to raise.
        kv, gate = exchange_compressor_halos(kv, gate, rate, cp_group)

    # The caller hands over the *global* packed metadata alongside a local shard,
    # exactly as the attention forward hands it to the compressors: only the module
    # holding the hidden states knows they are one shard, so only it can shard the
    # metadata. Both the compression below and the per-query ranges further down
    # read the sharded copy.
    rate_metadata = None
    if cache_layer is None and packed_compression_metadata is not None:
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
    # --- Patch.2 ---

    prior_kv = prior_gate = None
    if rate_metadata is not None:
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
    elif cp_enabled:
        # This rank's own windows, out of the haloed buffer in window order.
        # Mirrors the CSA compressor, which windows the same tokens at the model
        # head dim.
        window_indices, first_window_position = local_window_token_indices(
            shard, rate=rate, local_seq_len=local_seq_len, cp_rank=cp_rank, device=kv.device
        )
        flat_indices = window_indices.reshape(-1)
        chunk_kv, chunk_gate = kv[:, flat_indices], gate[:, flat_indices]
        if first_window_position >= rate:
            # The window before the first owned one, read out of the left halo. It
            # fills the very slots the decode path fills from the cache. Global
            # window 0 has no predecessor, so rank 0 leaves that slot at zero-kv /
            # -inf-gate and never reads the halo's zeros.
            previous_indices = window_indices[0] - rate
            prior_kv = kv[:, previous_indices, : self.head_dim]
            prior_gate = gate[:, previous_indices, : self.head_dim] + self.position_bias[:, : self.head_dim].to(
                gate.dtype
            )
    elif cache_layer is None:
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
        chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("indexer", kv, gate)

    if chunk_kv is None:
        pass  # The packed branch above already produced ``compressed``.
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
        compressed = empty_compressed_rows(chunk_kv, chunk_gate, self.head_dim)

    if cp_enabled:
        compressed = all_gather_compressed_rows(compressed, shard.counts, cp_group)
    # Covers the packed, windowed and empty branches above, all of which leave
    # `compressed` in the form the indexer's K cache holds.
    compressed = veomni_qat_fake_quant_act(compressed)
    compressed_kv = compressed if cache_layer is None else cache_layer.update_compressor_states("indexer", compressed)

    cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
    q = veomni_qat_linear(self.q_b_proj, q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
    q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)
    # Both sides of the index logits are rounded, so Q is quantized like K --
    # in contrast to the main attention, whose Q stays BF16.
    q = veomni_qat_fake_quant_act(q)
    # `weights_proj` stays unquantized: it produces one score per head, so its
    # [index_n_heads, hidden_size] weight has too few rows to tile at 128 in the
    # first place, and inference keeps it BF16.
    weights = self.scorer.weights_proj(hidden_states).float() * (
        self.scorer.weights_scaling * self.scorer.softmax_scale
    )
    compressed_len = compressed_kv.shape[1]
    top_k = min(self.index_topk, compressed_len)

    # --- Patch.1 ---
    indexer_implementation = veomni_dsa_indexer_implementation.value
    if indexer_implementation not in {"eager", "npu", "tilelang"}:
        raise ValueError(
            "DeepSeek-V4 does not support "
            f"dsa_indexer_implementation={indexer_implementation!r}; expected 'eager', 'npu' or 'tilelang'"
        )
    # A local query row ``i`` is global row ``query_offset + i``; off the context
    # parallel path ``query_offset`` is zero and this is the arange it always was.
    canonical_positions = (
        (torch.arange(seq_len, device=position_ids.device) + query_offset).unsqueeze(0).expand_as(position_ids)
    )
    packed_ranges = None if rate_metadata is None else packed_compressed_causal_ranges(rate_metadata)
    single_full_sequence = packed_sequence_slices is None or (
        len(packed_sequence_slices) == 1
        and packed_sequence_slices[0][0] == 0
        and packed_sequence_slices[0][1] == seq_len
    )
    use_npu = (
        indexer_implementation == "npu"
        and hidden_states.device.type == "npu"
        and cache_layer is None
        and not cp_enabled
        and not parallel_state.ulysses_enabled
        and single_full_sequence
        and compressed_len > 0
        and torch.equal(position_ids, canonical_positions)
    )
    if indexer_implementation == "npu" and not use_npu and compressed_len > 0:
        raise ValueError(
            "dsa_indexer_implementation='npu' was requested outside the fused Lightning Indexer "
            "contract (training/prefill, one full sequence with canonical positions, no SP/CP)"
        )
    if use_npu:
        from veomni.ops.kernels.deepseek_v4.npu_lightning_indexer import npu_lightning_indexer

        top_k_indices, _ = npu_lightning_indexer(q, compressed_kv, weights, top_k, compress_rate=self.compress_rate)
        return top_k_indices.to(torch.long)
    # Operand dtypes are the kernel's contract and are enforced by
    # ``v4_lighting_indexer`` itself, which reports the offending dtype. Only
    # structural conditions belong here.
    use_tilelang = (
        indexer_implementation == "tilelang"
        and hidden_states.is_cuda
        and self.num_heads <= 64
        and self.num_heads % 8 == 0
        and self.head_dim >= 32
        and self.head_dim == 1 << (self.head_dim - 1).bit_length()
        and cache_layer is None
        and compressed_len > 0
        and (packed_ranges is not None or torch.equal(position_ids, canonical_positions))
    )
    if indexer_implementation == "tilelang" and not use_tilelang:
        # Names ``dsa_indexer_loss`` when that is what selected the implementation:
        # the objective requires ``tilelang``, so a user who enabled it and then lands
        # here would otherwise get an error about a flag they never chose.
        chosen_by = " (required by dsa_indexer_loss)" if build_indexer_loss else ""
        raise ValueError(
            f"dsa_indexer_implementation='tilelang'{chosen_by} was requested but the TileLang indexer "
            f"does not support this call: is_cuda={hidden_states.is_cuda}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, decode={cache_layer is not None}, "
            f"compressed_len={compressed_len}, packed={packed_ranges is not None}"
        )
    if use_tilelang:
        query = q.transpose(0, 1).contiguous()
        query_weights = weights.transpose(0, 1).contiguous()
        query_range_starts = None if packed_ranges is None else packed_ranges[0]
        query_range_ends = None if packed_ranges is None else packed_ranges[1]
        # Either sequence-parallel mode has to spell out each query's visible
        # compressed interval, because the kernel's default derives it from the
        # query's *row*, which is no longer its position.
        if cp_enabled and query_range_starts is None:
            query_range_starts = torch.zeros(seq_len, device=q.device, dtype=torch.int32)
            query_positions = torch.arange(seq_len, device=q.device, dtype=torch.int32) + query_offset
            query_range_ends = (query_positions + 1) // self.compress_rate
        # Ulysses partitions the full-sequence queries here and stitches the
        # selection back together below; CP received them already partitioned and
        # wants the result per shard, so both halves fall away together. One flag
        # for both, so a slice can never happen without its matching all-gather.
        ulysses_query_partition = parallel_state.ulysses_enabled and not cp_enabled
        if ulysses_query_partition:
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

        index_score, top_k_indices = v4_lighting_indexer(
            query,
            compressed_kv.transpose(0, 1).contiguous(),
            query_weights,
            self.compress_rate,
            top_k,
            cu_seqlen_ks=query_range_starts,
            cu_seqlen_ke=query_range_ends,
        )
        if ulysses_query_partition:
            top_k_indices = gather_outputs(
                top_k_indices,
                gather_dim=1,
                group=parallel_state.ulysses_group,
            )
        # --- Patch.2 ---
        # ``index_score`` needs no all-gather to match: the two branches are mutually
        # exclusive, because ``_indexer_loss_enabled`` refuses ``ulysses_size > 1``
        # outright (a head shard would make the teacher's head sum partial), so a
        # partitioned score can never be the one being returned.
        if build_indexer_loss:
            return top_k_indices.to(torch.long), index_score
        # --- Patch.2 ---
        return top_k_indices.to(torch.long)
    # --- Patch.1 ---

    # No refusal for the loss here, deliberately: reaching this line under
    # ``dsa_indexer_loss`` would discard the scores the KL trains against, but it
    # cannot happen. ``_indexer_loss_enabled`` admits the objective only when
    # ``dsa_indexer_implementation`` is ``tilelang``, and the refusal above already
    # rejects that value whenever ``use_tilelang`` came out false -- for every
    # caller, not just this one, and before the module does any work. A second
    # refusal here would be unreachable by construction, and an unreachable ``raise``
    # that no test can exercise is worse than none: it reads as the protection while
    # the one doing the work sits elsewhere.
    scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores) * self.scorer.softmax_scale
    eager_weights = self.scorer.weights_proj(hidden_states).float() * self.scorer.weights_scaling
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



config = PatchConfig(
    source_module="transformers.models.deepseek_v4.modeling_deepseek_v4",
    target_file="patched_modeling_deepseek_v4_npu.py",
    description="DeepseekV4 NPU sibling — reuses every GPU structural/numerics patch, plus NPU-only FSDP2 hardening",
)

config.add_import("veomni.ops", names=["fused_moe_forward"])
# ``sparse_mqa_target_fwd`` is the indexer loss's teacher kernel. The objective
# needs both the TileLang indexer and the TileLang attention (see
# ``_indexer_loss_enabled``), and the TileLang sparse attention declines any
# non-CUDA tensor, so the branches reusing it are dead on NPU and refuse on the
# first attention call. The import exists only so patchgen can emit a module that
# type-checks.
config.add_import(
    "veomni.ops.kernels.deepseek_v4",
    names=["sparse_attn_tilelang", "sparse_mqa_target_fwd", "v4_lighting_indexer"],
)
config.add_import(
    "veomni.distributed.parallel_state",
    names=["get_parallel_state"],
)
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=[
        "gather_heads_scatter_seq",
        "gather_outputs",
        "gather_seq_scatter_heads",
        "reduce_sequence_parallel_loss",
    ],
)
# The GPU attention/indexer forwards reused below include context-parallel
# branches. CP is rejected at model build on NPU (see
# ``check_context_parallel_supported``), so those branches are dead here; the
# imports exist only so patchgen can emit a module that type-checks.
config.add_import(
    "veomni.distributed.context_parallel",
    names=[
        "all_gather_compressed_rows",
        "all_gather_kv",
        "empty_compressed_rows",
        "exchange_compressor_halos",
        "local_window_token_indices",
        "plan_compressor_shard",
    ],
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
        "scatter_topk_block_bias",
        "shard_packed_compression_metadata",
    ],
)

# Same rationale as the GPU config: surface MoeCausalLMOutputWithLogProbs so
# the reused ForCausalLM.forward can return per-token log-probs / entropy as
# constructor fields (FSDP2 unshard-hook safe — see GPU config comment).
config.add_import(
    "veomni.utils.model_outputs",
    names=[
        "FusedLinearAuxOutput",
        "FusedLinearAuxOutputMixin",
        "MoeCausalLMOutputWithLogProbs",
        "MoeModelOutputWithIndexerKL",
    ],
)
config.drop_import_names("MoeCausalLMOutputWithPast")

# The reused TopKRouter.forward calls the router-replay hook, so the generated
# NPU module needs the same names the GPU one imports.
config.add_import(
    "veomni.utils.moe_router_replay",
    names=["get_active_replay", "maybe_replay_indices"],
)

# The reused attention / indexer / compressor forwards route their projections
# and stored KV through the GPU config's QAT helpers, so the generated NPU module
# needs them too. Emitting them here costs nothing on Ascend: `qat_implementation`
# has no NPU backend, so the slot stays at "none" and every helper is a
# passthrough that never reaches the SM90-only kernels. Importing
# `veomni.ops.qat` is likewise safe -- TileLang loads inside the kernel
# wrappers, not at import.
config.add_import(
    "veomni.ops.qat",
    names=[
        "fp4_fake_quant_weight",
        "fp8_fake_quant_act",
        "fp8_fake_quant_act_prefix",
        "fp8_fake_quant_stacked_weight",
        "qat_linear",
    ],
)
config.add_helper(veomni_qat_linear)
config.add_helper(veomni_qat_fake_quant_kv)
config.add_helper(veomni_qat_fake_quant_act)
config.add_helper(veomni_qat_fake_quant_expert_weight)

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
    veomni_qat_implementation = OpsConfigSlot("qat_implementation")
    """
)

# The reused indexer/attention/model/ForCausalLM forwards read the indexer-loss
# gate, so the generated NPU module needs the same helpers the GPU one defines.
# Registered by reference rather than restated, so the two backends cannot drift
# apart on a predicate whose whole purpose is to be read identically from the
# three call sites that decide the forward's arity.
config.add_helper(_indexer_loss_enabled)
config.add_helper(_builds_indexer_kl)
config.add_helper(_split_indexer_output)
config.add_helper(indexer_kl_terms)

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
    replacement=deepseek_v4_indexer_forward_npu_patched,
    description="NPU Lightning Indexer dispatch with eager/TileLang compatibility paths",
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
    "DeepseekV4SparseMoeBlock.__init__",
    replacement=deepseek_v4_sparse_moe_block_init_patched,
    description="Flag routed experts to use the conservative max_M bound under non-distinct hash routing",
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
    self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
    self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
    self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
    self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
    self.rotary_emb = DeepseekV4RotaryEmbedding(config)
    self.scorer = DeepseekV4IndexerScorer(config)
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
    # Accepted and ignored, matching the GPU config's HCA compressor: the shared
    # ``DeepseekV4Attention.forward`` holds one compressor whose class is chosen by
    # layer type and calls it through a single call site, so both compressors have to
    # take the same arguments. Only the CSA one owns a Lightning Indexer. Dead on NPU
    # either way -- ``_indexer_loss_enabled`` refuses anything but the TileLang
    # indexer, which is CUDA-only -- but the signature has to line up with the call.
    build_indexer_loss: bool = False,
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
        compressed = veomni_qat_fake_quant_kv(compressed, self.rotary_emb.config.qk_rope_head_dim)
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

    compressed = veomni_qat_fake_quant_kv(compressed, self.rotary_emb.config.qk_rope_head_dim)
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
    # Accepted, refused, and never forwarded on this backend. The shared attention
    # forward passes ``_builds_indexer_kl``'s answer down here, so the parameter
    # exists because the call site is shared -- ``tests/models/
    # test_generated_call_site_signatures.py`` is what enforces that. The two indexer
    # call sites below stay on their bare-tensor return and this file needs no
    # ``_split_indexer_output``.
    build_indexer_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if (packed_sequence_slices is None) != (packed_compression_metadata is None):
        raise ValueError("Packed sequence slices and compression metadata must be provided together")
    if build_indexer_loss:
        # Reachable: ``dsa_indexer_implementation`` is a plain ``Literal`` with no
        # hardware gate, so ``tilelang`` parses on NPU and ``_indexer_loss_enabled``
        # then admits the objective. Everything after this point would quietly
        # disagree with it -- the indexer is called without the flag and returns bare
        # top-k indices, and the attention forward eventually fails its own wiring
        # check with a message about an internal invariant rather than about the two
        # lines of YAML that caused it. Say the true thing here instead.
        raise NotImplementedError(
            "dsa_indexer_loss is not implemented on NPU: the objective's student "
            "distribution is the TileLang Lightning Indexer's per-slot scores, and that "
            "kernel is CUDA-only. Set dsa_indexer_loss: false under model.model_config."
        )
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
        compressed = veomni_qat_fake_quant_kv(compressed, self.rotary_emb.config.qk_rope_head_dim)
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
        compressed = self.kv_norm((new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2))
        positions = torch.arange(n_windows, device=compressed.device)
        positions = positions * self.compress_rate + first_window_position
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
        compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

    compressed = veomni_qat_fake_quant_kv(compressed, self.rotary_emb.config.qk_rope_head_dim)
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
