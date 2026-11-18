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
Patch configuration for KimiK3 GPU patched modeling generation.

Regen command:
patchgen veomni.models.transformers.kimi_k3.kimi_k3_gpu_patch_gen_config -o veomni/models/transformers/kimi_k3/generated --diff

Source module: the vendored, HF-compatible modeling at
``veomni.models.transformers.kimi_k3.source.modeling_kimi_k3``
(transformers 5.16.1 ships no ``kimi_k3``, so the upstream modeling files are
vendored verbatim under ``source/`` and patched on top).

Patches (skeleton):
1.  ``KimiRMSNorm``  --instance-local ``VeomniOp`` for fused RMSNorm.
2.  ``KimiMoEGate.forward``  --drop the inference-only ``assert not self.training``.
3.  ``KimiBlockSparseMLP``   --replaced by ``PatchedKimiK3Experts``: fused
    ``[E, 2I, H]`` / ``[E, H, I]`` expert weights plus an instance-local
    ``moe_experts`` ``VeomniOp``. SituGLU experts bind the ``kimi_k3``
    variant (NPU group-gemm + SituGLU, EP alltoall-aware); other activations
    bind the ``standard`` variant.
4.  ``KimiSparseMoeBlock``   --replaced by ``PatchedKimiK3SparseMoeBlock``:
    training-mode forward (latent-MoE down/up projections + optional norm +
    shared experts), dropping the inference-only ``moe_infer`` path.
5.  ``KimiLinearModel.__init__`` --respect the requested ``_attn_implementation``
    instead of forcing ``flash_attention_2`` (MLA dispatches through the
    instance-local ``veomni_attn`` VeomniOp; see item 20).
6.  ``KimiK3ForConditionalGeneration.__init__`` --validate / remap the vision
    tower's ``_attn_implementation`` before construction (sdpa falls back to
    eager; unsupported names raise at construction instead of key-erroring
    at the first image forward; the ``text_config`` selection is left to
    ``VeOmniPretrainedConfig``'s root->child propagation, which honors
    explicit per-child selections) and keep the linear-attention decoder
    layers in the root FSDP2 unit by clearing the aggregated
    ``_no_split_modules`` after ``post_init``.
7.  ``KimiLinearForCausalLM.forward`` --fused CE path via the
    ``VeomniOp``-bound ``loss_function``; returns
    ``CausalLMOutputWithLogProbs`` so ``fused_linear_aux`` is a constructor
    field (FSDP2 pre-backward unshard safety).
8.  ``KimiK3ForConditionalGeneration.forward`` --fused CE path. Calls
    ``language_model.model`` directly (the HF original reads ``outputs[0]``
    from the CausalLM output whose first tuple slot is ``loss`` --never the
    hidden states) so the last hidden state is always available, then applies
    ``language_model.lm_head`` itself.
9.  ``KimiPreTrainedModel._init_weights`` / ``KimiK3PreTrainedModel._init_weights`` --    initialize the fused expert parameters in place (NPU-safe; no host-side
    copy into sharded/DTensor params) and zero the KDA ``dt_bias`` (HF leaves
    it as uninitialized ``torch.empty``; ``A_log`` and the ShortConvolution
    weights are already initialized at construction).
10. ``get_parallel_plan`` on ``KimiLinearForCausalLM`` /
    ``KimiK3ForConditionalGeneration`` --EP plan for the fused expert weights.
11. ``SituAndMul`` --instance-local ``VeomniOp`` for the fused Triton SituGLU
    kernel (``situ_and_mul`` op, ``kimi_k3`` variant).
12. ``KimiDeltaAttention.__init__`` --instance-local ``chunk_kda`` ``VeomniOp``
    (``kimi_k3`` variant): the fused AscendC kernel when configured, the
    modeling reference otherwise. The vendored source calls the module-level
    eager ``chunk_kda`` at its chunk call site (the portable reference, so
    the source stays runnable); the generated forward re-binds the call to
    this handle.
13. ``KimiK3ForConditionalGeneration.forward`` --ulysses sequence-parallel
    data flow: gather the per-rank sequence shards (input_ids / mask /
    labels / position_ids) to the full sequence before the multimodal merge
    (vision tower runs redundantly per rank), slice the merged embeds and
    labels back to the local shard, and hand the LM a full-sequence
    ``cache_position`` so causal masks / position ids match the gathered
    sequences inside the attention layers.
14. ``KimiDeltaAttention.forward`` / ``KimiMLAAttention.forward`` --SP
    gather at entry / narrow at exit: linear attention
    is sequential across tokens and MLA is full-sequence, so both run on the
    full (gathered) sequence on every SP rank and narrow the output back to
    the local shard. The MLA shard-sized causal mask is dropped (all-ones
    training packs); the KDA cu_seqlens derive from the full-sequence mask.
15. ``multihead_attention`` --lazy flash-attn import: the source's guarded
    import is flattened to ``flash_attn_varlen_func = None`` by codegen, so
    the vision varlen fallback would call ``None`` on non-NPU devices; the
    replacement imports flash-attn at call time instead.
16. ``KimiK3ForConditionalGeneration.forward`` --SP label contract: the merge
    is length-preserving on pre-expanded rows, so the SP collator's causal
    pre-shifted labels scatter 1:1 onto the pad slots with no realignment
    (the dataloader-side pre-shift stays the single causal shift).
17. ``KimiK3ForConditionalGeneration.forward`` --packed varlen metadata: the
    merge is length-preserving, so the collator's segment-aware
    ``position_ids`` / ``cu_seqlens`` stay valid and are forwarded (plus the
    coalesced ``cu_seqlens`` the KDA/MLA layers read) into the inner model;
    without them softmax attention and the recurrent KDA state would run as
    one long sequence and leak across packed samples.
    ``KimiLinearForCausalLM.forward`` forwards the collator's kwargs the same
    way for text-only batches.
18. ``MoonViT3dPretrainedModel._init_weights`` --vision-tower init under the
    FSDP2 meta flow: composite-model smart_apply dispatches sub-models to their
    own ``_init_weights`` and the transformers default chain has no
    branch for the learnable 2D pos-emb (or the KDA o_norm / to_empty-zeroed
    tower norms), so ``to_empty`` leaves raw device memory in the pos-emb --
    zeroed NPU pages silently drop visual position information; fsdp offload
    leaves uninitialized CPU memory that poisons image features with NaN.
19. ``KimiDecoderLayer.forward`` / ``KimiDecoderLayer._forward_attn_residual`` /
    ``KimiDecoderLayer._attn_selective_recompute`` / ``KimiDecoderLayer.__call__``
    --EP selective recompute, moved out of the vendored source together with
    its call sites (upstream has no expert parallelism, so the behaviour is
    not derivable from the vendored file): under EP, skip the whole-layer
    checkpoint and checkpoint only the collective-free attention stack. The
    source keeps plain upstream-shaped ``self._run_attn(...)`` calls so the
    vendored module stays runnable as its own re-sync reference.
20. ``KimiMLAAttention.__init__`` / ``KimiMLAAttention.forward`` --instance-local
    attention ``VeomniOp`` (``veomni_attn``) bound at construction from
    ``ops_config.attn_implementation``: the forward dispatches through the
    handle, so an unregistered attention implementation fails at construction
    instead of raising from inside forward (mirrors the peer models' attention
    binding and drops the ``ALL_ATTENTION_FUNCTIONS`` forward-time lookup from
    the generated file).
21. ``MoonViTEncoderLayer.__init__`` / ``.attention_qkvpacked`` /
    ``MoonViTEncoderLayer.forward`` / ``MoonViT3dEncoder.forward`` --the vision
    tower joins the op registry: an instance-local ``vision_attention``
    (``kimi_k3`` variant) ``VeomniOp`` bound at construction (rows ``eager`` /
    ``flash_attention_2`` / ``veomni_flash_attention_2``, all module-resolved
    from the modeling file like the standard family's ``eager`` row, so the
    lazy-import ``multihead_attention`` patch serves the flash row), the packed
    varlen call dispatches through the handle instead of the
    ``VL_VISION_ATTENTION_FUNCTIONS`` dict, and the encoder hoists the cu_seqlens
    host copy once per forward (threaded as ``cu_seqlens_host``; the NPU fusion
    kernel reads it via ``q_cu_seqlens_host`` / ``k_cu_seqlens_host``) so
    per-layer calls skip the D2H sync.
"""

import os
from functools import partial
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from veomni.models.loss_utils import ForCausalLMLoss
from veomni.ops import VeomniOp
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import (
    CausalLMOutputWithLogProbs,
)


config = PatchConfig(
    source_module="veomni.models.transformers.kimi_k3.source.modeling_kimi_k3",
    target_file="patched_modeling_kimi_k3_gpu.py",
    description="KimiK3 (MoonViT3d + KimiLinear) with VeOmni fused-MoE / fused-loss / training-mode patches",
)

# Surface ``CausalLMOutputWithLogProbs`` so the patched ``forward``s can return
# the ``fused_linear_aux`` payload as a constructor field. Mutating
# ``output.log_probs`` / ``output.entropy`` after constructing the base-class
# output would bypass ModelOutput pytree flattening, breaking FSDP2's
# pre-backward unshard hook on ``lm_head`` (parallels VeOmni #731's qwen3_5_moe
# fix).
config.add_import(
    "veomni.utils.model_outputs",
    names=["CausalLMOutputWithLogProbs", "FusedLinearAuxOutput", "FusedLinearAuxOutputMixin"],
)
# The patched forwards return the *WithLogProbs variants instead.
config.drop_import_names("CausalLMOutputWithPast")
config.drop_import_names("LlavaCausalLMOutputWithPast")

config.add_import("functools", names=["partial"])
config.add_import("math", names=["sqrt"])
config.add_import("os", is_from_import=False)
config.add_import("veomni.models.loss_utils", names=["ForCausalLMLoss"])
config.add_import("veomni.ops", names=["VeomniOp"])
config.add_import(
    "veomni.distributed.sequence_parallel.data",
    names=["gather_outputs", "slice_input_tensor"],
)
config.add_import(
    "veomni.distributed.sequence_parallel.ulysses",
    names=["gather_heads_scatter_seq", "gather_seq_scatter_heads"],
)
# Parallel state for the SP / EP policy patches below. The vendored source no
# longer imports this (upstream has no EP and no sequence parallel, so the
# policy is patch-owned); the generated file gets it here.
config.add_import(
    "veomni.distributed.parallel_state",
    names=["get_parallel_state"],
)


# FSDP2 DTensor-safe embedding padding-row zeroing (owned by the generated
# modeling: its init patches are the only callers; the plain source keeps the
# HF one-liner).


@config.add_helper
def _shard_dim0_offset_length(global_size: int, num_shards: int, rank: int) -> tuple[int, int]:
    """Offset and length of ``rank``'s shard under DTensor's even ``Shard(0)``
    split (torch.chunk semantics: ceil-sized chunks, the last shard absorbing
    the remainder)."""
    chunk_size = (global_size + num_shards - 1) // num_shards
    offset = rank * chunk_size
    return offset, min(chunk_size, global_size - offset)


@config.add_helper
def _zero_embedding_padding_row(weight: torch.Tensor, padding_idx: int) -> None:
    """Zero an embedding's padding row without cross-rank collectives.

    ``weight.data[padding_idx].zero_()`` on a ``Shard(0)`` DTensor triggers an
    implicit Shard -> Replicate redistribution: under FSDP2 CPU offload the
    materialized-on-CPU tensor has no communication backend (hard crash), and
    even on device the indexed write lands in a discarded replicate copy, so
    the padding row silently stays non-zero. Zeroing the owning rank's local
    shard needs no collective and actually takes effect; plain (non-DTensor)
    weights keep the original one-liner.
    """
    from torch.distributed.tensor import DTensor

    if not isinstance(weight, DTensor):
        weight[padding_idx].zero_()
        return
    local = weight.to_local()
    for mesh_dim, placement in enumerate(weight.placements):
        if not placement.is_shard() or placement.dim != 0:
            continue
        coordinate = weight.device_mesh.get_coordinate()
        if coordinate is None:
            return
        rank_along_dim = coordinate[mesh_dim]
        num_shards = weight.device_mesh.size(mesh_dim)
        # DTensor's even Shard(0) split is torch.chunk semantics (ceil-sized
        # chunks, the last shard absorbing the remainder), not floor +
        # remainder -- which indexes one row past the last uneven shard and
        # crashes initialization (e.g. the native vocab 163840 over 3 FSDP
        # shards).
        offset, length = _shard_dim0_offset_length(weight.shape[0], num_shards, rank_along_dim)
        if offset <= padding_idx < offset + length:
            local[padding_idx - offset].zero_()
        return
    if padding_idx < local.shape[0]:
        local[padding_idx].zero_()


# Sequence-parallel KDA (head-sharded ulysses all-to-all; see the
# KimiDeltaAttention.forward override for the dispatch).


@config.add_helper
def _kda_forward_sp_head_shard(module, hidden_states, attention_mask, **kwargs):
    """Head-sharded KDA training forward under sequence parallel.

    Linear attention (short conv, gated chunked recurrence, per-head gates)
    is independent across heads, so instead of the full-sequence gather --
    where every SP rank redundantly computes the whole layer over all
    tokens -- the ranks exchange [sequence shard, all heads] <->
    [full sequence, head shard] with ulysses-style all-to-alls, mirroring
    the flash wrapper's SP hooks used by the MLA layers. Projections and
    the output projection run on the local sequence shard; conv + chunk_kda
    + o_norm run on the full (unpadded) sequence restricted to this rank's
    head slice. Per-head parameters (A_log, dt_bias) are narrowed locally.
    The all-to-alls operate on the padded layout (equal shard lengths);
    unpad/repad use the full-sequence attention_mask exactly like the
    non-SP path.
    """
    import torch.distributed as dist

    state = get_parallel_state()
    group = state.sp_group
    sp_size = dist.get_world_size(group)
    sp_rank = dist.get_rank(group)
    H, Dh, Dk = module.num_heads, module.head_dim, module.head_k_dim
    assert H % sp_size == 0, f"num_heads ({H}) must be divisible by sp_size ({sp_size})"
    hs = H // sp_size

    # -- projections on the local (padded) sequence shard ----------------
    q_proj_states = module.q_proj(hidden_states)
    k_proj_states = module.k_proj(hidden_states)
    v_proj_states = module.v_proj(hidden_states)
    g = module.f_b_proj(module.f_a_proj(hidden_states))
    beta = module.b_proj(hidden_states)  # fp32 cast after the a2a (dtype-uniform pack)
    if module.use_full_rank_gate:
        g_post = module.g_proj(hidden_states)
    else:
        g_post = module.g_b_proj(module.g_a_proj(hidden_states))

    # -- [seq shard, all heads] -> [full seq, head shard] ----------------
    # Five tensors (q/k/v/g/beta) cross the SP group in ONE all-to-all:
    # each (Ts, C) is viewed as (Ts, sp, C/sp) and the per-rank slices are
    # concatenated, so the fused (Ts, sp*S) layout scatters directly into
    # rank blocks. The 2-D (seq, channel) layout keeps the exchange on the
    # fast _all_to_all_single path (scatter/gather dims 0/1). The o_norm
    # gate (g_post) does NOT cross -- Qwen3.5-MoE LASP pattern: it is a
    # pure per-token projection with no head dependency, so o_norm runs
    # after the restore a2a on the [seq shard, all heads] layout (-20%
    # payload).
    _tensors = [q_proj_states, k_proj_states, v_proj_states, g, beta]
    _x2d = torch.cat(
        [t.squeeze(0).view(-1, sp_size, t.shape[-1] // sp_size) for t in _tensors],
        dim=2,
    ).view(_tensors[0].shape[1], -1)
    _x2d = gather_seq_scatter_heads(_x2d, seq_dim=0, head_dim=1, group=group)
    _splits = [t.shape[-1] // sp_size for t in _tensors]
    q_proj_states, k_proj_states, v_proj_states, g, beta = (t.unsqueeze(0) for t in torch.split(_x2d, _splits, dim=1))
    beta = beta.float()
    full_len = q_proj_states.shape[1]

    # -- unpad on the full sequence (mask arrives full-sequence) ---------
    # With an all-ones packed mask the collator's cu_seqlens (+ its hoisted
    # host copy cu_seqlens_cpu) stay valid; a padded mask re-derives cu from
    # the unpad indices, and the host copy no longer matches, so it is
    # dropped in that branch.
    cu_seqlens = kwargs.get("cu_seqlens")
    cu_seqlens_cpu = kwargs.get("cu_seqlens_cpu")
    indices = None
    if attention_mask is not None:
        indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -full_len:])
        cu_seqlens_cpu = None
        q_proj_states = index_first_axis(rearrange(q_proj_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        k_proj_states = index_first_axis(rearrange(k_proj_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        v_proj_states = index_first_axis(rearrange(v_proj_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        g = index_first_axis(rearrange(g, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        beta = index_first_axis(rearrange(beta, "b s ... -> (b s) ..."), indices).unsqueeze(0)

    # Modification (round-3): one host materialization of the varlen
    # boundaries feeds the three short convolutions below (see the non-SP
    # forward): without it each ShortConvolution re-runs cu_seqlens.tolist()
    # (a device sync) per KDA layer.
    conv_cu_host = None
    if cu_seqlens is not None:
        conv_cu_host = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.tolist()

    # -- short convs on the full varlen sequence (segments intact) -------
    cs_qk = (sp_rank * hs * Dk, (sp_rank + 1) * hs * Dk)
    cs_v = (sp_rank * hs * Dh, (sp_rank + 1) * hs * Dh)
    q, _ = module.q_conv1d(
        x=q_proj_states,
        cache=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=conv_cu_host,
        channel_slice=cs_qk,
    )
    k, _ = module.k_conv1d(
        x=k_proj_states,
        cache=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=conv_cu_host,
        channel_slice=cs_qk,
    )
    v, _ = module.v_conv1d(
        x=v_proj_states,
        cache=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=conv_cu_host,
        channel_slice=cs_v,
    )

    # -- per-head parameters -> local head slice -------------------------
    A_log = module.A_log[sp_rank * hs : (sp_rank + 1) * hs]
    dt_bias = module.dt_bias.view(H, Dh)[sp_rank * hs : (sp_rank + 1) * hs].reshape(-1)

    q = rearrange(q, "... (h d) -> ... h d", d=Dk)
    k = rearrange(k, "... (h d) -> ... h d", d=Dk)
    v = rearrange(v, "... (h d) -> ... h d", d=Dh)
    g = rearrange(g, "... (h d) -> ... h d", d=Dh)

    o, _ = module.veomni_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=module.gate_lower_bound is not None,
        lower_bound=module.gate_lower_bound,
        transpose_state_layout=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        disable_recompute=True,
    )

    # -- repad, then [full seq, head shard] -> [seq shard, all heads] ----
    if attention_mask is not None:
        o = pad_input(o.squeeze(0), indices, hidden_states.shape[0], full_len)
    o = rearrange(o, "b t h d -> b t (h d)")
    # 2-D (seq, channel) fast-path restore: scatter sequence, gather the
    # fused channel dim back to all heads (rank order = head order).
    o = gather_heads_scatter_seq(o.squeeze(0), head_dim=1, seq_dim=0, group=group).unsqueeze(0)
    # -- output norm on the local sequence shard (gate never crossed) ----
    o = rearrange(o, "b t (h d) -> b t h d", d=Dh)
    g_post = rearrange(g_post, "b t (h d) -> b t h d", d=Dh)
    o = module.o_norm(o, g_post)
    o = rearrange(o, "b t h d -> b t (h d)")
    o = module.o_proj(o)
    return o


# SituGLU (instance-local VeomniOp, fused Triton kernel)


@config.override_method(
    "SituAndMul.__init__",
    description="Construct a local situ_and_mul VeomniOp",
)
def situ_and_mul_init_patched(self, beta: float = 1.0, linear_beta: float | None = None, config=None):
    nn.Module.__init__(self)
    self.beta = beta
    self.linear_beta = linear_beta
    impl = "eager" if config is None else config.ops_config.situ_and_mul_implementation
    self.veomni_situ_and_mul = VeomniOp("situ_and_mul", "kimi_k3", impl)


@config.override_method(
    "SituAndMul.forward",
    description="Always call the local situ_and_mul VeomniOp",
)
def situ_and_mul_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    return self.veomni_situ_and_mul(x, beta=self.beta, linear_beta=self.linear_beta)


# KDA chunked linear attention (instance-local VeomniOp, fused AscendC kernel)

# Patch 12: the source declares the per-instance kernel handle
# (``self.veomni_chunk_kda``) and calls it from both training paths; the
# override below defines that handle as a ``chunk_kda`` ``VeomniOp``
# (``kimi_k3`` variant) resolved from the ops selection installed at
# model-build time, mirroring the rms_norm / situ_and_mul instance-local
# pattern. Signatures are kwarg-compatible; extra K3 args
# (``A_log``/``dt_bias``/``transpose_state_layout``) pass through.


@config.override_method(
    "KimiDeltaAttention.__init__",
    description="Define the instance-local chunk_kda VeomniOp declared by the source",
)
def kimi_delta_attention_init_patched(self, config: "KimiLinearConfig", layer_idx: int):
    super().__init__()
    self.config = config
    self.mode = "chunk"

    self.hidden_size = config.hidden_size
    self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
    self.head_dim = config.linear_attn_config["head_dim"]
    self.num_heads = config.linear_attn_config["num_heads"]
    self.head_k_dim = self.head_dim
    self.num_k_heads = self.num_heads

    self.layer_idx = layer_idx

    assert self.mode in ["chunk", "fused_recurrent"], f"Not supported mode `{self.mode}`."

    projection_k_size = self.head_k_dim * self.num_k_heads
    projection_size = self.head_dim * self.num_heads

    self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
    self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
    self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

    self.q_conv1d = ShortConvolution(
        hidden_size=projection_k_size,
        kernel_size=self.conv_size,
        activation="silu",
    )
    self.k_conv1d = ShortConvolution(
        hidden_size=projection_k_size,
        kernel_size=self.conv_size,
        activation="silu",
    )
    self.v_conv1d = ShortConvolution(
        hidden_size=projection_size,
        kernel_size=self.conv_size,
        activation="silu",
    )

    self.A_log = torch.nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))

    self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
    self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

    self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))

    self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

    self.use_full_rank_gate = config.linear_attn_config.get("use_full_rank_gate", False)
    self.gate_lower_bound = config.linear_attn_config.get("gate_lower_bound", None)
    if self.use_full_rank_gate:
        self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
    else:
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

    self.o_norm = FusedRMSNormGated(self.head_dim, eps=config.rms_norm_eps, activation="sigmoid")
    self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)
    # Modification: define the instance-local ``chunk_kda`` kernel handle
    # (the source declares and calls it without binding a default): the
    # fused AscendC kernel when configured, the modeling eager reference
    # otherwise.
    self.veomni_chunk_kda = VeomniOp("chunk_kda", "kimi_k3", config.ops_config.chunk_kda_implementation)


@config.override_method(
    "KimiDeltaAttention.forward",
    description=(
        "Sequence-parallel KDA: head-sharded ulysses all-to-all by default "
        "(K3_KDA_HEAD_SHARD=1) or the legacy full-sequence gather (K3_KDA_HEAD_SHARD=0)"
    ),
)
def kimi_delta_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    cache_params=None,
    **kwargs,
):
    if attention_mask is not None:
        if attention_mask.dim() != 2:
            attention_mask = kwargs.get("padding_mask")

        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len] "
                "(0 = padding). 3D masks are not supported here.",
            )
    # Modification (SP): training under SP runs the head-sharded ulysses
    # all-to-all -- see _kda_forward_sp_head_shard. K3_KDA_HEAD_SHARD=0
    # selects the legacy full-sequence gather path below instead (also used
    # for the cache / inference case).
    if get_parallel_state().sp_enabled and cache_params is None and os.environ.get("K3_KDA_HEAD_SHARD", "1") == "1":
        return _kda_forward_sp_head_shard(self, hidden_states, attention_mask, **kwargs)
    # Modification (SP, legacy path): linear attention is sequential across
    # tokens, so gather the local shard into the full sequence before the
    # causal convs / projections and narrow the output back afterwards.
    # Every SP rank redundantly computes the full sequence; cu_seqlens /
    # attention_mask arrive full-sequence.
    if get_parallel_state().sp_enabled:
        hidden_states = gather_outputs(hidden_states, gather_dim=1, group=get_parallel_state().sp_group)
    use_cache = cache_params is not None
    batch_size, q_len, _ = hidden_states.shape
    mode = "fused_recurrent" if use_cache and q_len == 1 else self.mode
    if mode == "fused_recurrent":
        # The ``fused_recurrent_kda`` binding in this build is the eager
        # fallback stub (no recurrent implementation is vendored and no op
        # variant is registered): reject the cached-decode request here
        # instead of crashing from inside the kernel shim mid-forward.
        # No-cache generation (prefill) keeps the chunk path below.
        raise NotImplementedError(
            "cached single-token generation is not supported in this build: only the "
            "chunk KDA path has an implementation (fused_recurrent_kda is a stub). "
            "Run generation without cache or prefill in chunk mode."
        )
    if self.training:
        assert mode == "chunk", "Only chunk mode is supported in training."

    cu_seqlens = kwargs.get("cu_seqlens")
    # Host copy of the collator's cu_seqlens, hoisted once per forward by
    # the model forward: the AscendC entry consumes Host metadata, so
    # reusing it avoids a per-layer D2H sync. Dropped when the mask-based
    # unpad re-derives cu (the copy no longer matches).
    cu_seqlens_cpu = kwargs.get("cu_seqlens_cpu")
    indices = None
    if attention_mask is not None:
        indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
        cu_seqlens_cpu = None
        hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

    # Modification (round-3): one host materialization of the varlen
    # boundaries feeds the three short convolutions below -- without it
    # each ShortConvolution re-runs cu_seqlens.tolist() (a device sync)
    # per KDA layer, exactly the per-layer host copy the AscendC entries
    # avoid through cu_seqlens_cpu.
    conv_cu_host = None
    if cu_seqlens is not None:
        conv_cu_host = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.tolist()

    conv_state_q, conv_state_k, conv_state_v = None, None, None
    recurrent_state = None
    if cache_params is not None:
        if cache_params.conv_states[self.layer_idx] is not None:
            conv_state_q, conv_state_k, conv_state_v = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    q_proj_states = self.q_proj(hidden_states)
    k_proj_states = self.k_proj(hidden_states)
    v_proj_states = self.v_proj(hidden_states)
    q, conv_state_q = self.q_conv1d(
        x=q_proj_states,
        cache=conv_state_q,
        output_final_state=use_cache,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=conv_cu_host,
    )
    k, conv_state_k = self.k_conv1d(
        x=k_proj_states,
        cache=conv_state_k,
        output_final_state=use_cache,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=conv_cu_host,
    )
    v, conv_state_v = self.v_conv1d(
        x=v_proj_states,
        cache=conv_state_v,
        output_final_state=use_cache,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=conv_cu_host,
    )
    g = self.f_b_proj(self.f_a_proj(hidden_states))
    g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
    beta = self.b_proj(hidden_states).float()

    q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
    k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
    v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

    if mode == "chunk":
        o, recurrent_state = self.veomni_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound,
            transpose_state_layout=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            disable_recompute=True,
        )
    else:
        o, recurrent_state = fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=self.gate_lower_bound,
            transpose_state_layout=True,
            cu_seqlens=cu_seqlens,
        )
    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = recurrent_state
        cache_params.conv_states[self.layer_idx] = (conv_state_q, conv_state_k, conv_state_v)

    if self.use_full_rank_gate:
        g = self.g_proj(hidden_states)
    else:
        g = self.g_b_proj(self.g_a_proj(hidden_states))
    g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
    o = self.o_norm(o, g)

    o = rearrange(o, "b t h d -> b t (h d)")
    o = self.o_proj(o)
    if attention_mask is not None:
        o = pad_input(o.squeeze(0), indices, batch_size, q_len)
    if get_parallel_state().sp_enabled:
        o = slice_input_tensor(o, dim=1, group=get_parallel_state().sp_group)

    return o


# Decoder layer: EP selective recompute (moved out of the vendored source,
# call sites included -- the source keeps plain self._run_attn(...) calls so
# it stays runnable). Upstream has no expert parallelism, so this policy is
# pure VeOmni behaviour and lives in the patch layer alongside the SP
# routing -- the generated .diff shows it on every re-sync.


@config.override_method(
    "KimiDecoderLayer._attn_selective_recompute",
    description=(
        "EP selective-recompute policy (VeOmni-only; upstream has no EP): "
        "checkpoint only the collective-free attention stack under EP"
    ),
)
def kimi_decoder_layer_attn_selective_recompute_patched(self) -> bool:
    # NPU/EP (MindSpeed-style selective recompute): under EP the MoE block
    # contains alltoall dispatch/combine collectives. Non-reentrant
    # whole-layer gradient checkpointing replays them inside backward at
    # rank-skewed moments, desynchronizing the collectives (all-NaN grads
    # after a few training steps, observed during EP bring-up). When EP is
    # on we therefore skip whole-layer checkpointing and checkpoint ONLY
    # the attention stack (collective-free); the MoE block keeps its
    # activations so it is never replayed.
    return self.training and getattr(self, "gradient_checkpointing", False) and get_parallel_state().ep_enabled


@config.override_method(
    "KimiDecoderLayer.__call__",
    description=(
        "Skip the whole-layer GradientCheckpointingLayer checkpoint under EP "
        "(forward() checkpoints the attention stack itself)"
    ),
)
def kimi_decoder_layer_call_patched(self, *args, **kwargs):
    if self._attn_selective_recompute():
        # Skip the whole-layer checkpoint (see _attn_selective_recompute)
        # and run nn.Module.__call__ directly; forward() checkpoints the
        # attention stack itself.
        return super(GradientCheckpointingLayer, self).__call__(*args, **kwargs)
    return super().__call__(*args, **kwargs)


@config.override_method(
    "KimiDecoderLayer.forward",
    description=(
        "EP selective recompute at the call site too (moved out of the vendored "
        "source together with the policy method; upstream has no expert "
        "parallelism): checkpoint only the collective-free attention stack under EP"
    ),
)
def kimi_decoder_layer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: tuple[torch.Tensor] | None = None,
    output_attentions: bool | None = False,
    use_cache: bool | None = False,
    block_residual: torch.Tensor | None = None,
    **kwargs,
):
    if self.use_attn_residuals:
        return self._forward_attn_residual(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            output_attentions,
            use_cache,
            block_residual,
            **kwargs,
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    # NPU/EP (selective recompute): checkpoint the attention
    # stack only; the MoE block below keeps its activations so the EP
    # alltoall collectives are never replayed inside backward.
    if self._attn_selective_recompute():
        hidden_states = torch.utils.checkpoint.checkpoint(
            self._run_attn,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            output_attentions,
            use_cache,
            use_reentrant=False,
            **kwargs,
        )
    else:
        hidden_states = self._run_attn(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            output_attentions,
            use_cache,
            **kwargs,
        )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    if hasattr(self, "block_sparse_moe"):
        hidden_states = self.block_sparse_moe(hidden_states)
    else:
        hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


@config.override_method(
    "KimiDecoderLayer._forward_attn_residual",
    description=(
        "Attention-residual forward carrying the same EP selective-recompute call site (see KimiDecoderLayer.forward)"
    ),
)
def kimi_decoder_layer_forward_attn_residual_patched(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: tuple[torch.Tensor] | None = None,
    output_attentions: bool | None = False,
    use_cache: bool | None = False,
    block_residual: torch.Tensor | None = None,
    **kwargs,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    prefix_sum = hidden_states

    if block_residual is not None and block_residual.shape[1] > 0:
        hidden_states = _apply_attn_res(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            self.self_attention_res_proj,
            self.self_attention_res_norm,
        ).view(batch_size, seq_len, hidden_size)

    if self.layer_idx % self.attn_res_block_size == 0:
        block_residual = torch.cat([block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)], dim=1)
        prefix_sum = None

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    # NPU/EP (selective recompute): checkpoint the attention
    # stack only; the MoE block below keeps its activations so the EP
    # alltoall collectives are never replayed inside backward.
    if self._attn_selective_recompute():
        hidden_states = torch.utils.checkpoint.checkpoint(
            self._run_attn,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            output_attentions,
            use_cache,
            use_reentrant=False,
            **kwargs,
        )
    else:
        hidden_states = self._run_attn(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            output_attentions,
            use_cache,
            **kwargs,
        )

    if prefix_sum is not None:
        prefix_sum = prefix_sum + hidden_states
    else:
        prefix_sum = hidden_states

    hidden_states = _apply_attn_res(
        prefix_sum.view(-1, hidden_size),
        block_residual,
        self.mlp_res_proj,
        self.mlp_res_norm,
    ).view(batch_size, seq_len, hidden_size)

    hidden_states = self.post_attention_layernorm(hidden_states)
    if hasattr(self, "block_sparse_moe"):
        hidden_states = self.block_sparse_moe(hidden_states)
    else:
        hidden_states = self.mlp(hidden_states)

    if prefix_sum is None:
        prefix_sum = hidden_states
    else:
        prefix_sum = prefix_sum + hidden_states

    return prefix_sum, block_residual


# Vision varlen attention (flash-attn fallback on non-NPU devices)

# Patch 15: the source sets ``flash_attn_varlen_func = None`` at module level
# (patchgen cannot preserve the try/except import, so the guarded import is
# flattened to the fallback assignment). ``multihead_attention`` therefore
# calls ``None`` whenever ``npu_fusion_vision_attention`` falls back to it on
# a non-NPU device. Replace it with a variant that imports flash-attn lazily
# at call time: GPU hosts with ``veomni[gpu]`` get the real varlen kernel,
# NPU hosts never reach this fallback, and environments without flash-attn
# fail at the call site instead of at import time.


@config.replace_function(
    "multihead_attention",
    description="Vision varlen attention via a lazily-imported flash-attn",
)
def multihead_attention_patched(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None = None,
    k_cu_seqlens: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    deterministic: bool = False,
):
    """Multi-head attention using flash attention 2 (lazy import).

    Same packed varlen contract as the source ``multihead_attention``:
    q/k/v of shape ``(tot_seqlens, num_heads, head_dim)`` plus cumulative
    sequence lengths; returns ``(tot_seqlens, num_heads * head_dim)``.
    """
    from flash_attn import flash_attn_varlen_func

    attn_out = flash_attn_varlen_func(
        q,
        k,
        v,
        q_cu_seqlens,
        k_cu_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
        deterministic=deterministic,
    )
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    return attn_out.flatten(start_dim=-2)


# MLA attention: instance-local VeomniOp bound at construction (round-3: the
# forward dispatches through the handle, so an unsupported attention
# implementation fails at construction instead of raising from inside
# forward).


@config.override_method(
    "KimiMLAAttention.__init__",
    description=(
        "Bind the instance-local attention VeomniOp (veomni_attn) at construction so an "
        "unsupported attention implementation fails here instead of raising from inside "
        "forward (full-body override: patchgen has no init-append mechanism)"
    ),
)
def kimi_mla_attention_init_patched(self, config: "KimiLinearConfig", layer_idx: int):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    self.attention_dropout = getattr(config, "attention_dropout", 0.0)

    try:
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.use_nope = config.mla_use_nope
        self.scaling = self.q_head_dim ** (-0.5)
    except Exception as e:
        raise ValueError(f"Kimi MLA config is not found or not properly formatted: {e}") from e

    if self.q_lora_rank is not None:
        self.q_a_proj = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
        )
        self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank, config=config)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.q_head_dim,
            bias=False,
        )
    else:
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.q_head_dim,
            bias=False,
        )
    self.kv_a_proj_with_mqa = nn.Linear(
        self.hidden_size,
        self.kv_lora_rank + self.qk_rope_head_dim,
        bias=False,
    )
    self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank, config=config)
    self.kv_b_proj = nn.Linear(
        self.kv_lora_rank,
        self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
        bias=False,
    )
    self.o_proj = nn.Linear(
        self.num_heads * self.v_head_dim,
        self.hidden_size,
        bias=False,
    )
    self.is_causal = True
    assert self.use_nope

    self.use_output_gate = getattr(config, "mla_use_output_gate", False)
    if self.use_output_gate:
        projection_size = self.num_heads * self.v_head_dim
        self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

    self.rotary_emb = None
    # Modification (round-3): instance-local attention VeomniOp -- forward
    # dispatches through this handle, so an unregistered attention
    # implementation fails here, at construction, instead of raising
    # from inside forward.
    self.veomni_attn = VeomniOp("attention", "standard", self.config.ops_config.attn_implementation)


# MLA under sequence parallel (gather/narrow around the attention)


@config.override_method(
    "KimiMLAAttention.forward",
    description=(
        "Sequence-parallel MLA: full-sequence gather/narrow around the attention, or "
        "head-sharded ulysses inside the veomni_* attention adapters; dispatches "
        "through the instance-local veomni_attn VeomniOp bound at construction"
    ),
)
def kimi_mla_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    **kwargs,
):
    # NPU/SP (Ulysses sequence parallel): gather the local
    # sequence shard into the full sequence before the attention (MLA is
    # full-sequence); the output is narrowed back to the local shard at
    # the end. Every SP rank redundantly computes the full sequence.
    #
    # SP head sharding (MLA): when the attention
    # implementation is a veomni_* adapter (Ulysses SP is built into
    # the registered adapters, gated by the parallel state), skip the
    # gather/narrow entirely -- the adapter performs the ulysses
    # all-to-all internally (scatter heads, gather sequence), so each
    # rank computes H/sp heads over the full sequence instead of all
    # heads redundantly. MLA is rope-free (use_nope), so the local
    # shard needs no position handling.
    _attn_impl = getattr(self.config, "_attn_implementation", "eager")
    sp_ulysses_attn = get_parallel_state().sp_enabled and _attn_impl.startswith("veomni_")
    if get_parallel_state().sp_enabled and not sp_ulysses_attn:
        hidden_states = gather_outputs(hidden_states, gather_dim=1, group=get_parallel_state().sp_group)
        if attention_mask is not None and attention_mask.shape[-2] != hidden_states.shape[1]:
            # The causal mask was built against the local sequence shard
            # (e.g. [1, 1, S/sp, S] under SP). Rebuild the causal mask at
            # full length -- valid for the all-ones training packs
            # (padding positions under SP are not supported yet).
            seq_len_full = hidden_states.shape[1]
            min_value = torch.finfo(hidden_states.dtype).min
            attention_mask = torch.full(
                (1, 1, seq_len_full, seq_len_full),
                min_value,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            attention_mask = torch.triu(attention_mask, diagonal=1)
    elif sp_ulysses_attn:
        # The veomni wrapper reassembles the sequence inside its
        # all-to-all and follows FA-family mask semantics (causal via
        # is_causal; 2D padding masks are dropped, packing is the
        # supported varlen form), so a full-length mask would not match
        # the local shard anyway -- force it to None.
        attention_mask = None
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.q_head_dim)
    key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

    if self.q_lora_rank is not None:
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    else:
        q_states = self.q_proj(hidden_states)
    q_states = q_states.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    if self.config._attn_implementation == "flash_attention_2" and self.q_head_dim != self.v_head_dim:
        value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

    # Modification (round-3): dispatch through the instance-local veomni_attn
    # VeomniOp bound in __init__ -- a wrong or unregistered attention
    # implementation now fails at construction instead of raising from
    # inside forward, and the ALL_ATTENTION_FUNCTIONS forward-time lookup
    # (with its eager_attention_forward fallback branch) drops out of the
    # generated file.
    # Legacy SP path: the full sequence was gathered above, so Q/K/V are
    # already global. The raw eager/sdpa rows cannot run the ulysses
    # exchanges themselves, so tell the attention wrapper to skip its SP
    # check (it would otherwise reject non-veomni_* implementations
    # under SP); the veomni_* rows run the exchanges inside the wrapper
    # and keep the default.
    _call_kwargs = kwargs
    if get_parallel_state().sp_enabled and not sp_ulysses_attn:
        _call_kwargs = {**kwargs, "skip_ulysses": True}
    attn_output, _ = self.veomni_attn(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **_call_kwargs,
    )

    if self.config._attn_implementation == "flash_attention_2" and self.q_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, : self.v_head_dim]

    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    if self.use_output_gate:
        g = self.g_proj(hidden_states).sigmoid()
        attn_output = attn_output * g
    attn_output = self.o_proj(attn_output)
    if get_parallel_state().sp_enabled and not sp_ulysses_attn:
        # With the veomni_* adapter the output is already in
        # the local-shard layout -- no narrowing needed.
        attn_output = slice_input_tensor(attn_output, dim=1, group=get_parallel_state().sp_group)
    return attn_output


# RMSNorm (instance-local VeomniOp, functional fused kernel)


@config.override_method(
    "KimiRMSNorm.__init__",
    description="Construct a local rms_norm VeomniOp",
)
def kimi_rmsnorm_init_patched(self, hidden_size, eps=1e-6, config=None):
    nn.Module.__init__(self)
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    impl = "eager" if config is None else config.ops_config.rms_norm_implementation
    self.veomni_rms_norm = VeomniOp("rms_norm", "standard", impl)


@config.override_method(
    "KimiRMSNorm.forward",
    description="Always call the local rms_norm VeomniOp",
)
def kimi_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # aclnnRmsNorm (the torch_npu impl) has no mixed-dtype overload and the
    # shared kernel rejects misaligned inputs; align the affine weight here
    # (plain tensors under mixed precision keep an fp32 master weight).
    weight = self.weight.to(hidden_states.dtype)
    return self.veomni_rms_norm(hidden_states, weight, eps=self.variance_epsilon)


# MoE gate (training-mode routing)


@config.override_method(
    "KimiMoEGate.forward",
    description="Drop the inference-only `assert not self.training` so routing runs in training mode",
)
def kimi_moe_gate_forward_patched(self, hidden_states):
    bsz, seq_len, h = hidden_states.shape
    # compute gating score
    hidden_states = hidden_states.view(-1, h)
    logits = F.linear(
        hidden_states.type(torch.float32),
        self.weight.type(torch.float32),
        None,
    )
    if self.moe_router_activation_func == "sigmoid":
        scores = logits.sigmoid()
    elif self.moe_router_activation_func == "softmax":
        scores = logits.softmax(dim=1)
    else:
        raise NotImplementedError(
            f"insupportable scoring function for MoE gating: {self.moe_router_activation_func}",
        )

    # select top-k experts
    # Modification: removed `assert not self.training` (inference-only guard).
    scores = scores.view(bsz * seq_len, -1)
    scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
    if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )  # [n, num_expert_group]
        group_idx = torch.topk(
            group_scores,
            k=self.topk_group,
            dim=-1,
            sorted=False,
        )[1]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, num_expert_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, num_expert_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                bsz * seq_len,
                self.num_expert_group,
                self.num_experts // self.num_expert_group,
            )
            .reshape(bsz * seq_len, -1)
        )  # [n, e]
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
    else:
        tmp_scores = scores_for_choice
    _, topk_idx = torch.topk(
        tmp_scores,
        k=self.top_k,
        dim=-1,
        sorted=False,
    )
    topk_weight = scores.gather(1, topk_idx)

    # norm gate to sum 1
    if self.top_k > 1 and self.moe_renormalize:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
    # must multiply the scaling factor
    topk_weight = topk_weight * self.routed_scaling_factor

    return topk_idx, topk_weight


# Routed experts (fused weights + VeOmni fused-MoE dispatch)


@config.replace_class(
    "KimiBlockSparseMLP",
    description=(
        "Use v5 gate_up_proj expert weights ([E, 2I, H] F.linear layout) and the VeOmni fused "
        "MoE path. SituGLU experts bind the kimi_k3 moe_experts VeomniOp variant (NPU "
        "group-gemm + SituGLU activation, EP-aware); other activations bind the standard "
        "variant."
    ),
)
class PatchedKimiK3Experts(nn.Module):
    """Routed experts stored as 3-D tensors (VeOmni v5 fused layout).

    Weight layout (F.linear convention, same as ``PatchedQwen3MoeExperts``):

    * ``gate_up_proj``: ``[E, 2*I, H]`` --per-expert ``cat([w1, w3], dim=0)``
    * ``down_proj``   : ``[E, H, I]``  --per-expert ``w2``

    ``H`` is the routed-expert hidden size (``routed_expert_hidden_size`` for
    the latent MoE, i.e. the space the experts actually operate in), ``I`` is
    ``moe_intermediate_size``.
    """

    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_dim = config.moe_intermediate_size if intermediate_size is None else intermediate_size

        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))

        self.is_situ = config.hidden_act == "situ"
        if not self.is_situ and config.hidden_act != "silu":
            # The standard MoE implementation always applies F.silu in its
            # experts; anything else would silently train with the wrong
            # activation.
            raise ValueError(
                f"KimiK3 MoE experts implement 'situ' and 'silu' activations only, "
                f"got hidden_act={config.hidden_act!r}."
            )
        if self.is_situ:
            beta, linear_beta = _get_situ_activation_params(config)
            self.act_fn = SituAndMul(beta=beta, linear_beta=linear_beta, config=config)
        else:
            self.act_fn = ACT2FN[config.hidden_act]
        self.veomni_moe = VeomniOp(
            "moe_experts",
            "kimi_k3" if self.is_situ else "standard",
            config.ops_config.moe_implementation,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Modification: always dispatch through the instance-local
        # moe_experts VeomniOp (eager is a registered row). SituGLU experts
        # bind the kimi_k3 variant (NPU group-gemm with the SituGLU
        # activation and alltoall dispatch/combine under EP); other
        # activations bind the standard fused path.
        if self.is_situ:
            return self.veomni_moe(
                hidden_states,
                top_k_weights,
                top_k_index,
                self.gate_up_proj,
                self.down_proj,
                num_experts=self.num_experts,
                beta=self.act_fn.beta,
                linear_beta=self.act_fn.linear_beta,
            )
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
        )


# Sparse MoE block (training-mode forward)


@config.replace_class(
    "KimiSparseMoeBlock",
    description=(
        "Training-mode forward: fused PatchedKimiK3Experts + latent-MoE projections + shared "
        "experts. Drops the inference-only moe_infer/ep fields (the HF original raises "
        "NotImplementedError in training mode)."
    ),
)
class PatchedKimiK3SparseMoeBlock(nn.Module):
    """Patched Kimi sparse MoE block with 3-D expert tensors (training mode).

    Behavioural invariants compared with the original ``KimiSparseMoeBlock``:

    * Same gating logic (``KimiMoEGate`` with the patched training-mode forward).
    * Same latent-MoE down/up projection and optional RMSNorm.
    * Same shared-expert residual addition.
    * Same output shape and dtype semantics.

    The only structural change is that routed experts are stored as 3-D tensors
    inside ``PatchedKimiK3Experts`` instead of a ``ModuleList`` of per-expert
    MLPs, and the forward always runs the (differentiable) training path.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.moe_renormalize = config.moe_renormalize

        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = config.routed_expert_hidden_size if self.use_latent_moe else config.hidden_size
        self.latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)

        # NOTE: replace_class keeps the original class name in the generated
        # module, so reference KimiBlockSparseMLP (whose body is now the fused
        # PatchedKimiK3Experts implementation) here.
        self.experts = KimiBlockSparseMLP(
            config,
            hidden_size=self.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
        )
        self.gate = KimiMoEGate(config)

        if config.num_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = KimiMLP(
                config=config,
                intermediate_size=intermediate_size,
            )

        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(
                config.hidden_size,
                self.moe_hidden_size,
                bias=False,
            )
            self.routed_expert_up_proj = nn.Linear(
                self.moe_hidden_size,
                config.hidden_size,
                bias=False,
            )
            if self.latent_moe_use_norm:
                self.routed_expert_norm = KimiRMSNorm(
                    self.moe_hidden_size,
                    eps=config.rms_norm_eps,
                    config=config,
                )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if self.use_latent_moe:
            hidden_states = self.routed_expert_down_proj(hidden_states)

        # Modification: differentiable training path (the HF original raises
        # NotImplementedError here and only supports the no-grad moe_infer).
        y = self.experts(hidden_states, topk_idx, topk_weight)

        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)

        y = y.view(*orig_shape)

        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


# KimiLinearModel.__init__ (respect _attn_implementation)


@config.override_method(
    "KimiLinearModel.__init__",
    description="Respect the requested `_attn_implementation` (eager / veomni_* kernels) instead of forcing flash_attention_2",
)
def kimi_linear_model_init_patched(self, config):
    super().__init__(config)
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    self.layers = nn.ModuleList([KimiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
    self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)

    self.use_attn_residuals = getattr(config, "attn_res_block_size", None) is not None
    if self.use_attn_residuals:
        self.output_attn_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    # Modification: respect the caller-requested attention implementation
    # (eager / veomni_* NPU kernels) instead of forcing flash_attention_2.
    # KimiMLAAttention binds its instance-local veomni_attn VeomniOp in
    # __init__, so any registered implementation works and an unsupported
    # one fails at construction.
    self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
    self.gradient_checkpointing = False
    # Initialize weights and apply final processing
    self.post_init()


# KimiK3ForConditionalGeneration.__init__ (vision attention validation) --
# NOTE: patchgen's codegen has no INIT_MODIFICATION applicator (modify_init is
# a silent no-op beyond the header listing), so this must be a full
# override_method replacement of __init__.


@config.override_method(
    "KimiK3ForConditionalGeneration.__init__",
    description="Validate / remap the vision tower's `_attn_implementation` before "
    "construction (the MoonViT3d tower registers 'eager'/'flash_attention_2'/"
    "'veomni_flash_attention_2' in VL_VISION_ATTENTION_FUNCTIONS -- HF K3 has "
    "no vision 'sdpa', so an 'sdpa' request falls back to 'eager' and any other "
    "unsupported name raises at construction instead of key-erroring at the first "
    "image forward; the text_config selection is left to VeOmniPretrainedConfig's "
    "root->child propagation, which honors explicit per-child selections); also "
    "override _no_split_modules to keep vision/projector in the root FSDP2 unit "
    "(asymmetric multimodal batches deadlock otherwise)",
)
def kimi_k3_for_conditional_generation_init_patched(self, config):
    # Modification: validate / remap the vision tower's attention selection
    # before submodule construction. text_config is intentionally NOT stamped
    # here: VeOmniPretrainedConfig's root->child propagation already resolves
    # the selection with an explicit per-child choice (the mapping form)
    # ranking above the root value, and writing text_config._attn_implementation
    # here would clobber that explicit choice (round-3 review). The MoonViT3d
    # vision tower has no 'sdpa' implementation (VL_VISION_ATTENTION_FUNCTIONS
    # registers only 'eager'/'flash_attention_2'/'veomni_flash_attention_2'), so
    # 'sdpa'/'veomni_sdpa' fall back to 'eager'; any other unsupported name
    # raises at construction instead of key-erroring at the first image
    # forward.
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        vision_impl = vision_config._attn_implementation
        if vision_impl in ("sdpa", "veomni_sdpa"):
            vision_impl = "eager"
        if vision_impl not in VL_VISION_ATTENTION_FUNCTIONS:
            raise ValueError(
                f"Unsupported vision attention implementation {vision_impl!r} for the "
                "MoonViT3d vision tower; supported implementations are "
                f"{sorted(VL_VISION_ATTENTION_FUNCTIONS)} (plus 'sdpa'/'veomni_sdpa', "
                "which fall back to 'eager')"
            )
        vision_config._attn_implementation = vision_impl

    super().__init__(config)

    vt_config = VisionTowerConfig(config.vision_config)
    self.vision_tower = MoonViT3dPretrainedModel(vt_config)

    proj_config = ProjectorConfig(config.vision_config)
    if proj_config.mm_projector_type == "identity":
        self.mm_projector = IdentityMap()
    elif proj_config.mm_projector_type == "mlp":
        self.mm_projector = MLP(proj_config)
    elif proj_config.mm_projector_type == "patchmerger":
        self.mm_projector = PatchMergerMLP(proj_config)
    elif proj_config.mm_projector_type == "patchmergerv2":
        self.mm_projector = PatchMergerMLPV2(proj_config)
    else:
        raise ValueError(f"Unsupported mm_projector_type: {proj_config.mm_projector_type}")

    self.language_model = KimiLinearForCausalLM(config.text_config)
    impl = config.ops_config.cross_entropy_loss_implementation
    self.veomni_ce = VeomniOp("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, op=self.veomni_ce)
    self.post_init()

    # Modification (VeOmni FSDP2 wrap): keep the vision tower / projector
    # parameters in the root FSDP2 unit. VeOmni's build_parallelize_model
    # fully_shards every module whose class name appears in
    # _no_split_modules; with the HF K3 vision classes listed, an asymmetric
    # multimodal batch (some ranks without images) deadlocks on the vision
    # all-gather (only image ranks forward the tower). Text decoder layers
    # keep their per-layer FSDP2 units. transformers v5 aggregates submodule
    # _no_split_modules onto the model instance during post_init, so both the
    # tower class attribute and the instance attribute must be overridden
    # after super().__init__ / post_init.
    MoonViT3dPretrainedModel._no_split_modules = []
    self._no_split_modules = ["KimiDecoderLayer"]

    if hasattr(self.language_model, "dtype"):
        target_dtype = self.language_model.dtype
        self.vision_tower = self.vision_tower.to(dtype=target_dtype)
        self.mm_projector = self.mm_projector.to(dtype=target_dtype)


# MoonViT3d vision tower: instance-local vision_attention VeomniOp + hoisted
# host cu_seqlens (round-3 review: the tower was the one attention site with
# no kernel abstraction -- a hand-written dict dispatch validating nothing and
# a per-layer .tolist() D2H sync inside the NPU fusion kernel).


@config.override_method(
    "MoonViTEncoderLayer.__init__",
    description="Bind the instance-local vision_attention (kimi_k3 variant) VeomniOp at "
    "construction so an unsupported or unavailable vision implementation fails here "
    "instead of raising from inside forward (mirrors item 20's MLA binding)",
)
def moonvit_encoder_layer_init_patched(
    self,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    qkv_hidden_size: int | None = None,
    norm_type: str = "layernorm",
    mlp_type: str = "mlp2",
    *,
    attn_implementation: str = "flash_attention_2",
    activation=F.gelu,
    attn_bias: bool = False,
    linear_bias: bool = True,
    use_deterministic_attn: bool = False,
):
    super().__init__()
    self.num_heads = num_heads
    self.hidden_dim = hidden_dim
    self.qkv_hidden_size = hidden_dim if qkv_hidden_size is None else qkv_hidden_size
    self.hidden_size_per_attention_head = self.qkv_hidden_size // self.num_heads
    self.attn_implementation = attn_implementation
    # Modification: bind the op handle right after the selection lands, so
    # the row's hardware / package requirements are validated at construction
    # (a 'flash_attention_2' vision selection on a host without flash-attn
    # fails here with a registry-attributable error, not an ImportError
    # raised from inside a modeling function at the first image forward).
    self.veomni_vision_attn = VeomniOp("vision_attention", "kimi_k3", attn_implementation)
    self.use_deterministic_attn = use_deterministic_attn

    if norm_type == "layernorm":
        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
    elif norm_type == "rmsnorm":
        self.norm0 = nn.RMSNorm(hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
    else:
        raise NotImplementedError(f"Not support norm_type: {norm_type}")

    if mlp_type == "mlp2":
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation, bias=linear_bias)
    else:
        raise NotImplementedError(f"Not support mlp_type: {mlp_type}")

    self.wqkv = nn.Linear(hidden_dim, self.qkv_hidden_size * 3, bias=attn_bias)
    self.wo = nn.Linear(self.qkv_hidden_size, hidden_dim, bias=attn_bias)


@config.override_method(
    "MoonViTEncoderLayer.attention_qkvpacked",
    description="Dispatch the packed varlen vision attention through the instance-local "
    "veomni_vision_attn handle instead of the VL_VISION_ATTENTION_FUNCTIONS dict "
    "(selection validated at construction; threads the hoisted host cu_seqlens "
    "copy so the NPU fusion kernel skips the per-layer D2H sync)",
)
def moonvit_encoder_layer_attention_qkvpacked_patched(
    self,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor,
    rope_freqs_cis: torch.Tensor | None = None,
    cu_seqlens_host: list | None = None,
):
    """
    Args:
        x (torch.Tensor): (batch_size, seqlen, hidden_dim)
        cu_seqlens (torch.Tensor):
    """
    xqkv = self.wqkv(x)

    qkv_shape = xqkv.size()[:-1] + (
        3,
        self.num_heads,
        self.hidden_size_per_attention_head,
    )
    # xqkv: (batch_size, seqlen, 3, nheads, headdim)
    xqkv = xqkv.view(*qkv_shape)
    xq, xk, xv = torch.unbind(xqkv, dim=-3)

    xq, xk = apply_rope(xq, xk, rope_freqs_cis)

    # Modification: dispatch through the bound op handle (the vendored dict
    # lookup served the same three functions but validated nothing); the
    # hoisted host copy rides along as q_cu_seqlens_host / k_cu_seqlens_host
    # and is consumed only by the NPU fusion implementation.
    attn_out = self.veomni_vision_attn(
        self,
        xq,
        xk,
        xv,
        q_cu_seqlens=cu_seqlens,
        k_cu_seqlens=cu_seqlens,
        max_seqlen_k=max_seqlen,
        max_seqlen_q=max_seqlen,
        deterministic=self.use_deterministic_attn,
        q_cu_seqlens_host=cu_seqlens_host,
        k_cu_seqlens_host=cu_seqlens_host,
    )

    attn_out = self.wo(attn_out)
    return attn_out


@config.override_method(
    "MoonViTEncoderLayer.forward",
    description="Thread the hoisted host cu_seqlens copy down to the packed varlen attention call",
)
def moonvit_encoder_layer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    rope_freqs_cis: torch.Tensor | None = None,
    cu_seqlens_host: list | None = None,
):
    residual = hidden_states
    hidden_states = self.norm0(hidden_states)

    # Modification: pass the hoisted host copy through (None keeps the
    # per-call .tolist() behavior for direct callers of the layer).
    hidden_states = self.attention_qkvpacked(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis, cu_seqlens_host)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.norm1(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


@config.override_method(
    "MoonViT3dEncoder.forward",
    description="Hoist the cu_seqlens host copy once per tower forward (free when "
    "grid_thws is host-side; one D2H sync otherwise) and thread it to every "
    "layer's varlen attention call, mirroring the KDA cu_seqlens_cpu contract",
)
def moonvit_3d_encoder_forward_patched(
    self,
    hidden_states: torch.Tensor,
    grid_thws: torch.Tensor,
) -> torch.Tensor:
    rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws=grid_thws, device=hidden_states.device)

    lengths = torch.cat(
        (
            torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
            grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
        )
    )

    max_seqlen = lengths.max()
    cu_seqlens = lengths.to(hidden_states.device).cumsum(dim=0, dtype=torch.int32)
    # Modification: materialize the host copy of the cumulative lengths once
    # per tower forward instead of once per attention layer: grid_thws arrives
    # host-side in the collator flow, so the CPU cumsum needs no device sync;
    # on hosts that stage grid_thws on device this is a single .tolist() (the
    # NPU fusion kernel previously paid two D2H syncs per vision layer).
    if lengths.device.type == "cpu":
        cu_seqlens_host = lengths.cumsum(dim=0, dtype=torch.int32).tolist()
    else:
        cu_seqlens_host = cu_seqlens.tolist()
    for block in self.blocks:
        hidden_states = block(
            hidden_states,
            cu_seqlens,
            max_seqlen,
            rope_freqs_cis=rope_freqs_cis,
            cu_seqlens_host=cu_seqlens_host,
        )

    hidden_states = self.final_layernorm(hidden_states)
    return hidden_states


@config.override_method(
    "KimiK3ForConditionalGeneration.tie_weights",
    description="Accept the transformers v5 tie_weights kwargs (missing_keys / recompute_mapping) in the override",
)
def kimi_k3_tie_weights_patched(self, missing_keys=None, recompute_mapping=True):
    # Modification: forward the transformers v5 kwargs; the HF original
    # override took no arguments and crashed on `recompute_mapping=False`
    # from PreTrainedModel.post_init.
    return self.language_model.tie_weights(
        missing_keys=missing_keys,
        recompute_mapping=recompute_mapping,
    )


# KimiLinearForCausalLM.__init__ (bind the CE VeomniOp)


@config.override_method(
    "KimiLinearForCausalLM.__init__",
    description="Bind ForCausalLMLoss VeomniOp on KimiLinearForCausalLM",
)
def kimi_linear_for_causal_lm_init_patched(self, config):
    super().__init__(config)
    self.model = KimiLinearModel(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    impl = config.ops_config.cross_entropy_loss_implementation
    self.veomni_ce = VeomniOp("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, op=self.veomni_ce)
    self.post_init()


# KimiLinearForCausalLM.forward (fused CE)


@config.override_method(
    "KimiLinearForCausalLM.forward",
    description="Fused cross-entropy path via the VeomniOp-bound loss_function",
)
def kimi_linear_for_causal_lm_forward_patched(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    generation_mode=None,
    return_dict=None,
    cache_position=None,
    **kwargs,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Modification (packed varlen): forward the collator's FlashAttention
    # kwargs into the inner model so packed-sample boundaries reach the
    # attention layers. Without them a packed batch (all-ones attention mask)
    # degrades to a single-sequence causal path and softmax attention / the
    # recurrent KDA state leak across packed samples. The collator's
    # ``linear_attn_cu_seq_lens_q`` is the tail-coalesced form the linear
    # attention layers want, published under the ``cu_seqlens`` kwarg the KDA
    # layers read. Nothing changes for non-packed batches (no kwargs present).
    fa_kwargs = {}
    for _fa_key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
        if _fa_key in kwargs:
            fa_kwargs[_fa_key] = kwargs[_fa_key]
    _linear_cu = kwargs.get("linear_attn_cu_seq_lens_q")
    if _linear_cu is None:
        _linear_cu = kwargs.get("cu_seq_lens_q")
    if _linear_cu is not None:
        fa_kwargs["cu_seqlens"] = _linear_cu
    # Modification (round-3): hoist the varlen boundaries to host once per
    # forward (mirroring KimiK3ForConditionalGeneration.forward) -- the NPU
    # KDA stack (AscendC entry + short convolutions) consumes host
    # metadata, and without this hoist every KDA layer re-pays the D2H
    # copy this forward already avoids.
    _cu_linear = fa_kwargs.get("cu_seqlens")
    if isinstance(_cu_linear, torch.Tensor):
        fa_kwargs["cu_seqlens_cpu"] = _cu_linear.cpu()

    # Packed varlen metadata is only honored by the varlen attention
    # implementations: the dense eager/sdpa/veomni_sdpa path would run one
    # causal mask across the whole packed row and leak across packed
    # samples (the same rejection KimiK3ForConditionalGeneration.forward
    # applies). cu_seq_lens_q alone is not evidence of packing -- the
    # collator emits [0, length] for plain un-packed sequences -- so reject
    # only a real boundary between non-empty segments (known tail-padding
    # segments do not count). K3_ALLOW_DENSE_PACKED=1 re-enables the
    # documented debug-only dense path (used by the CPU packing tests).
    _cu_seq_lens_q = fa_kwargs.get("cu_seq_lens_q")
    if _cu_seq_lens_q is not None and os.environ.get("K3_ALLOW_DENSE_PACKED", "0") != "1":
        _attn_impl = _kimi_k3_decoder_attn_implementation(self)
        if _attn_impl in ("eager", "sdpa", "veomni_sdpa"):
            _cu_t = _cu_seq_lens_q
            if not isinstance(_cu_t, torch.Tensor):
                _cu_t = torch.as_tensor(_cu_t, dtype=torch.long)
            _seg_lens = torch.diff(_cu_t)
            _tail_pads = kwargs.get("tail_padding_length", 0)
            _tail_pads = 0 if _tail_pads is None else int(_tail_pads)
            if _tail_pads > 0 and _seg_lens.numel() > 0 and int(_seg_lens[-1]) == _tail_pads:
                _seg_lens = _seg_lens[:-1]
            _num_samples = int((_seg_lens > 0).sum())
            if _num_samples > 1:
                raise ValueError(
                    f"attention_implementation={_attn_impl!r} cannot run packed batches "
                    f"({_num_samples} samples share a row): the dense causal mask does not "
                    f"isolate them. Use a varlen implementation (flash_attention_2 / "
                    f"veomni_flash_attention_2), disable dynamic packing, or set "
                    f"K3_ALLOW_DENSE_PACKED=1 for the documented debug-only dense path."
                )

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **fa_kwargs,
    )

    hidden_states = outputs[0]
    if generation_mode:
        hidden_states = hidden_states[:, -1:]

    # Modification: fused cross-entropy path.
    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        # Modification: fused cross-entropy via the VeomniOp-bound helper
        # (fused rows consume hidden_states + weights; the eager row does
        # the projection inside the op and returns the logits).
        loss, logits, fused_linear_aux = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.vocab_size,
            hidden_states=hidden_states,
            weights=self.lm_head.weight,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states)

    return CausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        fused_linear_aux=fused_linear_aux,
    )


# KimiK3ForConditionalGeneration.forward (fused CE)


@config.override_method(
    "KimiK3ForConditionalGeneration._merge_input_ids_with_image_features",
    description=(
        "Three input forms: pre-expanded media-pad rows (the kimi_k3 data transform; "
        "length-preserving masked_scatter), legacy one-placeholder-per-image rows "
        "(vendored HF occupation-table expansion), and the no-image passthrough"
    ),
)
def kimi_k3_merge_input_ids_with_image_features_patched(
    self,
    image_features: list[torch.Tensor],
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    *,
    require_pre_expanded: bool = False,
):
    """Merge projected image features into the text embedding row.

    Dispatches on the number of media-placeholder tokens in ``input_ids``
    (one D2H read per merged forward):

    * equal to the total merged vision length -> the pre-expanded form: the
      kimi_k3 data transform already materialized one ``<|media_pad|>`` per
      merged vision token, so the merge is length-preserving -- features are
      scattered onto the pad slots and ``attention_mask`` / ``labels`` /
      ``position_ids`` pass through untouched (the transform already writes
      ``ignore_index`` across the media span, and the collator's packed
      position_ids / cu_seq_lens stay valid).
    * equal to the number of images -> the legacy HF form: one placeholder
      per image, expanded through the vendored occupation table; positions
      and the attention mask are rebuilt at the expanded length.
    * zero -> nothing to merge.

    ``require_pre_expanded`` (set by the forward under sequence parallelism,
    where the collator pre-shifts labels and the packed varlen metadata is
    only valid for a length-preserving merge) rejects the legacy form
    loudly instead of letting a sharded batch mislabel silently.
    """
    image_token_index: int = self.config.media_placeholder_token_id
    feature_lengths = [x.shape[0] for x in image_features]
    num_pads = int((input_ids == image_token_index).sum().item())

    if num_pads == 0:
        if feature_lengths:
            raise ValueError(
                f"Received {len(image_features)} image feature tensors but input_ids contains no "
                f"'{image_token_index}' media placeholders: nowhere to merge them."
            )
        return inputs_embeds, attention_mask, labels, position_ids

    if num_pads == sum(feature_lengths):
        # Pre-expanded form: masked_scatter fills True positions in
        # row-major order, which matches the sample-major order of the
        # concatenated features (the transform / collator keep images and
        # their placeholders in the same row order).
        flat_features = torch.cat(list(image_features), dim=0).to(inputs_embeds.dtype)
        scatter_mask = (input_ids == image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
        final_embedding = inputs_embeds.masked_scatter(scatter_mask, flat_features)
        return final_embedding, attention_mask, labels, position_ids

    if num_pads != len(image_features):
        raise ValueError(
            f"Cannot merge image features into input_ids: found {num_pads} "
            f"'{image_token_index}' media placeholders for {len(image_features)} images "
            f"({sum(feature_lengths)} merged vision tokens). Expected the pre-expanded "
            "form (one placeholder per merged vision token) or the legacy form "
            "(one placeholder per image)."
        )
    if require_pre_expanded:
        raise ValueError(
            "Sequence-parallel K3 training requires pre-expanded media placeholders "
            "(one '<|media_pad|>' per merged vision token, written by the kimi_k3 data "
            "transform): the legacy one-placeholder-per-image expansion changes the row "
            "length, which invalidates the SP collator's packed position_ids / cu_seq_lens "
            "and pre-shifted labels."
        )

    # -- legacy HF occupation-table expansion (vendored from the source) --
    _, embed_dim = image_features[0].shape
    image_features = torch.cat(image_features, dim=0)
    pad_token_id: int = self.config.pad_token_id
    ignore_index: int = self.config.ignore_index

    batch_size, sequence_length = input_ids.shape
    left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id))

    # 1. Create a mask to know where special image tokens are
    _token_occupation_table = torch.ones_like(input_ids.flatten())
    _token_occupation_table[input_ids.flatten() == image_token_index] = torch.tensor(
        feature_lengths, dtype=torch.long, device=input_ids.device
    )
    _token_occupation_table = _token_occupation_table.reshape(input_ids.shape)

    max_embed_dim = _token_occupation_table.sum(-1).max().item()
    assert max_embed_dim >= sequence_length, (
        f"The maximum embedding dimension ({max_embed_dim}) is less than the sequence length ({sequence_length})"
    )
    batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

    # 2. Compute the positions where text should be written
    # Calculate new positions for text tokens in merged image-text sequence.
    new_token_positions = torch.cumsum(_token_occupation_table, -1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    if left_padding:
        new_token_positions += nb_image_pad[:, None]  # offset for left padding
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    # 3. Create the full embedding, already padded to the maximum position
    final_embedding = torch.zeros(
        batch_size,
        max_embed_dim,
        embed_dim,
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    final_attention_mask = torch.zeros(
        batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
    )
    if labels is not None:
        final_labels = torch.full(
            (batch_size, max_embed_dim),
            ignore_index,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
    # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
    # set the corresponding tensors into their correct target device.
    target_device = inputs_embeds.device
    batch_indices, non_image_indices, text_to_overwrite = (
        batch_indices.to(target_device),
        non_image_indices.to(target_device),
        text_to_overwrite.to(target_device),
    )
    attention_mask = attention_mask.to(target_device)

    # 4. Fill the embeddings based on the mask.
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
    if labels is not None:
        final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

    # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
    image_to_overwrite = torch.full((batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device)
    image_to_overwrite[batch_indices, text_to_overwrite] = False
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

    if image_to_overwrite.sum() != image_features.shape[:-1].numel():
        raise ValueError(
            f"The input provided to the model are wrong. The number of image tokens is {image_to_overwrite.sum()} while"
            f" the number of image features given to the model is {image_features.shape[:-1].numel()}. "
            "This prevents correct indexing and breaks batch generation."
        )

    final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
    final_attention_mask |= image_to_overwrite
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

    # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
    batch_indices, pad_indices = torch.where(input_ids == pad_token_id)
    indices_to_mask = new_token_positions[batch_indices, pad_indices]

    final_embedding[batch_indices, indices_to_mask] = 0

    if labels is None:
        final_labels = None

    return (
        final_embedding,
        final_attention_mask,
        final_labels,
        position_ids,
    )


@config.override_method(
    "KimiK3ForConditionalGeneration.forward",
    description=(
        "Fused cross-entropy path. Calls language_model.model directly so the last hidden "
        "state is always available (the HF original reads outputs[0] of the CausalLM output "
        "whose first tuple slot is `loss`), then applies language_model.lm_head itself."
    ),
)
def kimi_k3_for_conditional_generation_forward_patched(
    self,
    input_ids=None,
    pixel_values=None,
    grid_thws=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **kwargs,
):
    assert self.vision_tower is not None, "vision_tower is not loaded"
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Modification (SP): ulysses sequence parallel -- the sequence
    # shards arrive per rank; gather them back into the full sequence so the
    # multimodal merge (and the vision tower) run on the full sequence, then
    # slice the merged embeds back to the local shard afterwards. The
    # attention layers re-gather internally (full-sequence compute), so
    # attention_mask / position_ids / cache_position stay full-sequence.
    cache_position = None
    if get_parallel_state().sp_enabled:
        if input_ids is None:
            # The multimodal merge locates media placeholders via input_ids,
            # and a sharded inputs_embeds batch cannot be gathered or merged;
            # reject the input mode instead of passing None into the SP
            # collective.
            raise ValueError(
                "KimiK3ForConditionalGeneration.forward requires input_ids under sequence "
                "parallelism: the multimodal merge locates media placeholders via input_ids, "
                "so an inputs_embeds-only batch cannot be gathered or merged on this path."
            )
        sp_group = get_parallel_state().sp_group
        input_ids = gather_outputs(input_ids, gather_dim=1, group=sp_group)
        if attention_mask is not None:
            attention_mask = gather_outputs(attention_mask, gather_dim=1, group=sp_group)
        if labels is not None:
            labels = gather_outputs(labels, gather_dim=1, group=sp_group)
        if position_ids is not None:
            position_ids = gather_outputs(position_ids, gather_dim=1, group=sp_group)

    if inputs_embeds is None:
        # 1. Extra the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text and images. On the pre-expanded training rows (one
        # <|media_pad|> per merged vision token; see the merge override) the
        # merge is length-preserving, so the collator's packed position_ids /
        # cu_seq_lens stay valid through it and the pre-shifted SP labels
        # scatter 1:1 -- no re-alignment happens here.
        if pixel_values is not None and len(pixel_values) > 0 and input_ids.shape[1] != 1:
            image_features = self._extract_image_features(pixel_values, grid_thws)
            if self.mm_projector:
                image_features = self.mm_projector(image_features)

            # NPU/FSDP2 mixed-precision fix (mirrors the source forward): align
            # the fused embeds to the *text embedding* dtype instead of the
            # vision dtype. Under mixed precision the text parameters are bf16,
            # and a fp32 vision path would leak fp32 activations into the first
            # decoder layer and the attn-residual block_residual chain.
            target_dtype = self.get_input_embeddings().weight.dtype
            image_features = [f.to(target_dtype) for f in image_features]
            inputs_embeds = inputs_embeds.to(target_dtype)  # num_tokens, embed_dim
            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                image_features,
                inputs_embeds,
                input_ids,
                attention_mask,
                labels,
                position_ids,
                require_pre_expanded=get_parallel_state().sp_enabled,
            )

        # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
        # generation with cache
        elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
            # Retrieve the first layer to inspect the logits and mask out the hidden states
            # that are set to 0
            first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

            # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
            batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

            # Get the target length
            target_length = input_ids.shape[1]
            past_length = first_layer_past_key_value.shape[-1]

            extended_attention_mask = torch.ones(
                (attention_mask.shape[0], past_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            # Filter out only the tokens that can be un-attended, this can happen
            # if one uses Llava + Fused modules where the cache on the
            # first iteration is already big enough, or if one passes custom cache
            valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
            new_batch_index = batch_index[valid_indices]
            new_non_attended_tokens = non_attended_tokens[valid_indices]

            # Zero-out the places where we don't need to attend
            extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

            attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

    # Modification (packed varlen): forward the collator's varlen metadata.
    # The pre-expanded merge is length-preserving, so the collator's
    # per-sample position_ids and cu_seq_lens stay valid through it and are
    # forwarded as-is (exactly like KimiLinearForCausalLM.forward does for
    # text-only batches). cu_seqlens is hoisted to host once per forward
    # (``cu_seqlens_cpu``) so the NPU linear-attention entries can consume
    # Host metadata without a per-layer D2H sync.
    fa_kwargs = {}
    for _fa_key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
        if _fa_key in kwargs:
            fa_kwargs[_fa_key] = kwargs[_fa_key]
    _linear_cu = kwargs.get("linear_attn_cu_seq_lens_q")
    if _linear_cu is None:
        _linear_cu = kwargs.get("cu_seq_lens_q")
    if _linear_cu is not None:
        fa_kwargs["cu_seqlens"] = _linear_cu
    _cu_linear = fa_kwargs.get("cu_seqlens")
    if isinstance(_cu_linear, torch.Tensor):
        fa_kwargs["cu_seqlens_cpu"] = _cu_linear.cpu()

    # Packed varlen metadata is only honored by the varlen attention
    # implementations: the dense eager/sdpa/veomni_sdpa MLA path would run one causal
    # mask across the whole merged row and leak across packed samples
    # (eager/sdpa stay available for non-packed batches). cu_seq_lens_q
    # alone is not evidence of packing -- the collator emits [0, length]
    # for plain un-packed sequences and the image-merge rebuild produces
    # the same single-segment shape, which the dense causal mask handles
    # exactly -- so reject only a real boundary between two non-empty
    # segments (distinct samples sharing a row; known tail-padding
    # segments do not count), instead of silently corrupting the packed
    # batch. K3_ALLOW_DENSE_PACKED=1 re-enables the
    # documented debug-only dense path (used by the CPU packing tests,
    # where no varlen kernel runs).
    _cu_seq_lens_q = fa_kwargs.get("cu_seq_lens_q")
    if _cu_seq_lens_q is not None and os.environ.get("K3_ALLOW_DENSE_PACKED", "0") != "1":
        _attn_impl = _kimi_k3_decoder_attn_implementation(self)
        if _attn_impl in ("eager", "sdpa", "veomni_sdpa"):
            _cu_t = _cu_seq_lens_q
            if not isinstance(_cu_t, torch.Tensor):
                _cu_t = torch.as_tensor(_cu_t, dtype=torch.long)
            _seg_lens = torch.diff(_cu_t)
            # Known tail padding (SP / pad-to-length) is coalesced into one
            # final segment by the collator and the merge rebuild: those
            # tokens are padding, not a sample, so drop the segment before
            # counting real samples (only when it matches the published
            # length -- otherwise stay conservative and count it).
            _tail_pads = kwargs.get("tail_padding_length", 0)
            _tail_pads = 0 if _tail_pads is None else int(_tail_pads)
            if _tail_pads > 0 and _seg_lens.numel() > 0 and int(_seg_lens[-1]) == _tail_pads:
                _seg_lens = _seg_lens[:-1]
            _num_samples = int((_seg_lens > 0).sum())
            if _num_samples > 1:
                raise ValueError(
                    f"attention_implementation={_attn_impl!r} cannot run packed batches "
                    f"({_num_samples} samples share a row): the dense causal mask does not "
                    f"isolate them. Use a varlen implementation (flash_attention_2 / "
                    f"veomni_flash_attention_2), disable dynamic packing, or set "
                    f"K3_ALLOW_DENSE_PACKED=1 for the documented debug-only dense path."
                )

    # Modification (SP): slice the merged embeds / labels back to the local
    # shard and hand the LM a full-sequence cache_position so position ids
    # and causal masks are built at full length (matching the gathered
    # sequences inside the attention layers). VeOmni's ForCausalLMLoss
    # consumes labels as-is under SP (dataloader-side pre-shift contract):
    # the pre-expanded merge scatters the pre-shifted labels 1:1, so they
    # need no realignment here. The collator pads the packed row to an
    # SP-even length, so the slice introduces no extra pads (a hand-built
    # odd row would surface as a loud cu/shape mismatch in the layers).
    if get_parallel_state().sp_enabled:
        sp_group = get_parallel_state().sp_group
        full_seq_len = inputs_embeds.shape[1]
        inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, group=sp_group)
        if labels is not None:
            labels = slice_input_tensor(labels, dim=1, group=sp_group, padding_value=self.config.ignore_index)
        cache_position = torch.arange(full_seq_len, device=inputs_embeds.device)

    # Modification: call the inner KimiLinearModel directly so the returned
    # BaseModelOutputWithPast[0] is reliably the last hidden state (the HF
    # original called the full KimiLinearForCausalLM and read outputs[0],
    # which is the `loss` slot of CausalLMOutputWithPast).
    outputs = self.language_model.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **fa_kwargs,
    )

    hidden_states = outputs[0]

    # Modification: fused cross-entropy path. The HF original computed a manual
    # attention-mask-selected CE over lm_head logits; the VeOmni path uses the
    # standard shift semantics (equivalent for all-ones masks, and image/pad
    # positions are already -100 via the merge). output_attentions /
    # output_hidden_states are not plumbed through the inner model call.
    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        # Modification: fused cross-entropy via the VeomniOp-bound helper
        # (fused rows consume hidden_states + weights; the eager row does
        # the projection inside the op and returns the logits).
        loss, logits, fused_linear_aux = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.text_config.vocab_size,
            hidden_states=hidden_states,
            weights=self.language_model.lm_head.weight,
            **kwargs,
        )
    else:
        logits = self.language_model.lm_head(hidden_states)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=None,
        attentions=None,
        fused_linear_aux=fused_linear_aux,
    )


# Weight init (fused experts + KDA dt_bias; NPU-safe in-place ops)


@config.override_method(
    "KimiPreTrainedModel._init_weights",
    description="Init fused expert weights (in-place, NPU-safe) and zero the KDA dt_bias",
)
def kimi_pretrained_init_weights_patched(self, module):
    std = self.config.initializer_range
    if isinstance(module, KimiBlockSparseMLP):
        # Modification: fused expert params are plain nn.Parameters (not
        # nn.Linear), so the default branches below never touch them.
        # In-place normal_ keeps this NPU-safe under fully_shard (no
        # host-side copy into device/DTensor params).
        module.gate_up_proj.data.normal_(mean=0.0, std=std)
        module.down_proj.data.normal_(mean=0.0, std=std)
        return
    if isinstance(module, KimiMoEGate):
        # Modification: the router is missed on every init path --
        # ``reset_parameters`` re-inits only ``weight`` (and never runs under
        # the FSDP2 meta flow anyway) while ``e_score_correction_bias`` is
        # left as uninitialized ``torch.empty`` even on the eager path.
        # Restore the construction default for the weight (kaiming, matching
        # reset_parameters) and start the bias at zero (no auxiliary-loss-free
        # correction at init).
        nn.init.kaiming_uniform_(module.weight, a=sqrt(5))
        module.e_score_correction_bias.data.zero_()
        return
    if isinstance(module, KimiRMSNorm):
        # Modification: FSDP2 meta flow materializes with to_empty (all
        # zeros) and relies on init_weights() to re-initialize; the
        # construction-time torch.ones in KimiRMSNorm.__init__ never runs
        # on meta. Without this branch every norm weight stays zero ->
        # dead forward (logits==0, loss==ln(V)) and zero upstream grads.
        module.weight.data.fill_(1.0)
        return
    if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
        # Modification: vision tower (MoonViT / PatchMerger) norms hit the
        # same to_empty zeroing; restore construction defaults (ones, zero
        # bias).
        module.weight.data.fill_(1.0)
        if getattr(module, "bias", None) is not None:
            module.bias.data.zero_()
        return
    if isinstance(module, KimiDeltaAttention):
        # Modification: HF leaves `dt_bias` as uninitialized `torch.empty`;
        # zero it so the KDA gate stays finite under random init (checked-in
        # weights overwrite this).
        # Modification (KDA zero-grad root cause): construction-time init is
        # lost to FSDP2 to_empty. ShortConvolution is an nn.Conv1d subclass
        # matching no branch below -> q/k/v conv weights stay zero ->
        # q=k=v=0 -> chunk_kda sits at an exact bilinear saddle -> zero
        # grads for the whole KDA branch. Restore construction defaults:
        # Conv1d reset_parameters and A_log=log(U[1,16]).
        module.dt_bias.data.zero_()
        module.A_log.data.uniform_(1.0, 16.0).log_()
        for _conv in (module.q_conv1d, module.k_conv1d, module.v_conv1d):
            _conv.reset_parameters()
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            _zero_embedding_padding_row(module.weight.data, module.padding_idx)
    elif isinstance(module, FusedRMSNormGated):
        module.reset_parameters()
    elif isinstance(module, Learnable2DInterpPosEmbDivided_fixed):
        module.reset_parameters()


@config.override_method(
    "KimiK3PreTrainedModel._init_weights",
    description="Init fused expert weights (in-place, NPU-safe) and zero the KDA dt_bias",
)
def kimi_k3_pretrained_init_weights_patched(self, module):
    # important: this ported version of Llava isn't meant for training from scratch - only
    # inference and fine-tuning - so the proper init weights code was removed - the original codebase
    # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
    std = (
        self.config.initializer_range
        if hasattr(self.config, "initializer_range")
        else self.config.text_config.initializer_range
    )

    if isinstance(module, KimiBlockSparseMLP):
        # Modification: fused expert params are plain nn.Parameters (not
        # nn.Linear), so the default branches below never touch them.
        # In-place normal_ keeps this NPU-safe under fully_shard (no
        # host-side copy into device/DTensor params).
        module.gate_up_proj.data.normal_(mean=0.0, std=std)
        module.down_proj.data.normal_(mean=0.0, std=std)
        return
    if isinstance(module, KimiMoEGate):
        # Modification: the router is missed on every init path --
        # ``reset_parameters`` re-inits only ``weight`` (and never runs under
        # the FSDP2 meta flow anyway) while ``e_score_correction_bias`` is
        # left as uninitialized ``torch.empty`` even on the eager path.
        # Restore the construction default for the weight (kaiming, matching
        # reset_parameters) and start the bias at zero (no auxiliary-loss-free
        # correction at init).
        nn.init.kaiming_uniform_(module.weight, a=sqrt(5))
        module.e_score_correction_bias.data.zero_()
        return
    if isinstance(module, KimiRMSNorm):
        # Modification: FSDP2 meta flow materializes with to_empty (all
        # zeros) and relies on init_weights() to re-initialize; the
        # construction-time torch.ones in KimiRMSNorm.__init__ never runs
        # on meta. Without this branch every norm weight stays zero ->
        # dead forward (logits==0, loss==ln(V)) and zero upstream grads.
        module.weight.data.fill_(1.0)
        return
    if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
        # Modification: vision tower (MoonViT / PatchMerger) norms hit the
        # same to_empty zeroing; restore construction defaults (ones, zero
        # bias).
        module.weight.data.fill_(1.0)
        if getattr(module, "bias", None) is not None:
            module.bias.data.zero_()
        return
    if isinstance(module, KimiDeltaAttention):
        # Modification: HF leaves `dt_bias` as uninitialized `torch.empty`;
        # zero it so the KDA gate stays finite under random init.
        # Modification (KDA zero-grad root cause): same to_empty loss as
        # the text-model copy -- restore ShortConvolution (nn.Conv1d)
        # weights and A_log construction defaults.
        module.dt_bias.data.zero_()
        module.A_log.data.uniform_(1.0, 16.0).log_()
        for _conv in (module.q_conv1d, module.k_conv1d, module.v_conv1d):
            _conv.reset_parameters()

    if hasattr(module, "class_embedding"):
        module.class_embedding.data.normal_(mean=0.0, std=std)

    if isinstance(module, (nn.Linear, nn.Conv2d)):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            _zero_embedding_padding_row(module.weight.data, module.padding_idx)
    elif isinstance(module, FusedRMSNormGated):
        module.reset_parameters()
    elif isinstance(module, Learnable2DInterpPosEmbDivided_fixed):
        module.reset_parameters()


@config.override_method(
    "MoonViT3dPretrainedModel._init_weights",
    description="Vision-tower weight init under the FSDP2 meta flow: composite-model "
    "smart_apply dispatches sub-models to their own _init_weights and the "
    "transformers default chain has no branch for the learnable 2D pos-emb "
    "(or the KDA o_norm / to_empty-zeroed tower norms), so to_empty leaves raw "
    "device memory in the pos-emb -- zeroed NPU pages silently drop visual "
    "position information and fsdp offload leaves uninitialized CPU memory "
    "that poisons image features with NaN from the first step",
)
def moonvit3d_pretrained_init_weights_patched(self, module):
    # important: this ported version of Llava isn't meant for training from scratch - only
    # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
    # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
    # Modification: the tower's VisionTowerConfig carries no initializer_range /
    # text_config (unlike the root KimiK3Config), so mirror the transformers
    # default std ladder instead of its composite-config fallback.
    if hasattr(self.config, "initializer_range"):
        std = self.config.initializer_range or 0.02
    elif hasattr(self.config, "init_std"):
        std = self.config.init_std
    elif hasattr(self.config, "initializer_factor"):
        std = self.config.initializer_factor
    else:
        # 0.02 is the standard default value across the library
        std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

    if hasattr(module, "class_embedding"):
        module.class_embedding.data.normal_(mean=0.0, std=std)

    if isinstance(module, (nn.Linear, nn.Conv2d)):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            _zero_embedding_padding_row(module.weight.data, module.padding_idx)
    elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
        # Modification: FSDP2 meta flow materializes with to_empty (all
        # zeros); restore the construction defaults (ones, zero bias).
        module.weight.data.fill_(1.0)
        if getattr(module, "bias", None) is not None:
            module.bias.data.zero_()
    elif isinstance(module, FusedRMSNormGated):
        # KDA o_norm affine scale: restore the ones default after to_empty
        # materialization (checkpoint-missing keys re-init here as well).
        module.reset_parameters()
    elif isinstance(module, Learnable2DInterpPosEmbDivided_fixed):
        # Vision pos-emb is a raw nn.Parameter filled only by
        # reset_parameters(); without this branch to_empty materialization
        # leaves allocator garbage in it.
        module.reset_parameters()


# The position-id stub must live at module level: VLMTrainer captures it
# in the dataset transform and dataloader workers pickle that transform
# (spawn start method on NPU) -- a closure would fail with
# "Can't pickle local object".


@config.add_helper
def _kimi_k3_position_ids(input_ids=None, **kwargs):
    """Arange position-id stub for the VLM trainer data hook.

    K3 derives positions inside the modeling (vision divided-fixed
    pos-emb from grid_thws, NoPE MLA, recurrent KDA), so generic
    consumers get an arange over the sequence; module level keeps the
    reference picklable for dataloader workers.
    """
    if input_ids is None:
        return None
    return torch.arange(input_ids.shape[-1], device=input_ids.device).expand_as(input_ids)


@config.add_helper
def _kimi_k3_decoder_attn_implementation(model):
    """The attention implementation the decoder layers actually run.

    The packed-batch guards in both entry points must consult the decoder's
    own selection (round-6 review): the VLM root config permits per-child
    attention selections that rank above the root value (the round-3
    mapping form), so the text child is the authority; on the text-only
    entry the config IS the text config and the same resolution applies.
    Reading the root first let a root varlen / text eager pair slip a
    packed row past the guard (the dense eager mask would attend across
    samples) and, symmetrically, rejected a valid varlen-configured pack.
    """
    decoder_config = getattr(model.config, "text_config", None)
    if decoder_config is None:
        decoder_config = model.config
    attn_impl = getattr(decoder_config, "_attn_implementation", None)
    if attn_impl is None:
        attn_impl = getattr(model.config, "_attn_implementation", None)
    return attn_impl


@config.override_method(
    "KimiK3ForConditionalGeneration.get_position_id_func",
    description="VLM trainer data hook: K3 derives positions inside the modeling "
    "(vision divided-fixed pos-emb from grid_thws, NoPE MLA, recurrent KDA); "
    "return the module-level arange stub (picklable for dataloader workers)",
)
def kimi_k3_get_position_id_func_patched(self):
    return _kimi_k3_position_ids


@config.override_method(
    "KimiK3ForConditionalGeneration.get_extra_collate_infos",
    description="K3 SP contract: text tokens are seq-sharded (ulysses) but the vision tower runs "
    "redundantly on every SP rank -- pixel_values/grid_thws stay full (no sp-slice). "
    "grid_thws (num_images, 3) cats along dim 0 like image_grid_thw. Tuple form; "
    "MainCollator expands it to DataCollateInfo.",
)
def kimi_k3_get_extra_collate_infos_patched(self):
    return {
        "pixel_values": (0, False, None, None),
        "grid_thws": (0, False, None, None),
    }


# Parallel plans (EP for the fused expert weights)


@config.override_method(
    "KimiLinearForCausalLM.get_parallel_plan",
    description="Register KimiLinear expert parallel plan for v5 generated modeling",
)
def kimi_linear_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan("model")


@config.override_method(
    "KimiK3ForConditionalGeneration.get_parallel_plan",
    description="Register KimiK3 expert parallel plan for v5 generated modeling",
)
def kimi_k3_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()


# Dummy references
# Names referenced by the patch bodies above that live in the *generated*
# module (vendored from the source module). They are defined as dummies here
# so this config module stays importable; patchgen resolves the real references
# from the generated module at codegen time.
SituAndMul = None  # noqa: E305
_get_situ_activation_params = None  # noqa: E305
ShortConvolution = None  # noqa: E305
FusedRMSNormGated = None  # noqa: E305
KimiMoEGate = None  # noqa: E305
KimiMLP = None  # noqa: E305
KimiRMSNorm = None  # noqa: E305
KimiDecoderLayer = None  # noqa: E305
KimiBlockSparseMLP = None  # noqa: E305
KimiDeltaAttention = None  # noqa: E305
KimiLinearModel = None  # noqa: E305
Learnable2DInterpPosEmbDivided_fixed = None  # noqa: E305
MLP2 = None  # noqa: E305
apply_rope = None  # noqa: E305
