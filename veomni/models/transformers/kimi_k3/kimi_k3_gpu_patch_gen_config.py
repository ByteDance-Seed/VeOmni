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
    instead of forcing ``flash_attention_2`` (MLA already dispatches through
    ``ALL_ATTENTION_FUNCTIONS`` with an eager fallback).
6.  ``KimiK3ForConditionalGeneration.__init__`` --propagate the top-level
    ``_attn_implementation`` to ``text_config`` / ``vision_config`` (full
    ``__init__`` override; sdpa falls back to eager for the vision tower) and
    keep the linear-attention decoder layers in the root FSDP2 unit by
    clearing the aggregated ``_no_split_modules`` after ``post_init``.
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
    (``kimi_k3`` variant) rebound onto the source's per-instance handle: the
    fused AscendC kernel when configured, the modeling reference otherwise.
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
16. ``KimiK3ForConditionalGeneration.forward`` --SP label alignment: undo the
    SP collator's causal pre-shift before the multimodal merge scatters the
    labels (pre-shifted labels would strand the first post-image text token's
    supervision inside the IGNORE-filled vision span), so the post-merge shift
    in the SP slicing block becomes the single causal shift.
17. ``KimiK3ForConditionalGeneration.forward`` --packed varlen metadata: the
    multimodal merge recomputes ``position_ids`` as a continuous arange, which
    erases the collator's ``position_ids == 0`` segment markers. Map the
    pre-merge boundaries to merged coordinates through the merge's occupation
    table, rebuild segment-aware ``position_ids``, refresh the FlashAttention
    cu_seq_lens kwargs, and forward them (plus the coalesced
    ``cu_seqlens`` the KDA/MLA layers read) into the inner model; without
    them softmax attention and the recurrent KDA state leak across packed
    samples. ``KimiLinearForCausalLM.forward`` forwards the collator's
    (unexpanded, still-valid) kwargs the same way for text-only batches.
"""

from functools import partial
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from veomni.models.loss_utils import ForCausalLMLoss
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import (
    CausalLMOutputWithLogProbs,
)
from veomni.utils.seqlen_pos_transform_utils import (
    coalesce_tail_padding_cu_seqlens,
    prepare_fa_kwargs_from_position_ids,
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
config.add_import("veomni.models.loss_utils", names=["ForCausalLMLoss"])
config.add_import("veomni.ops", names=["VeomniOp"])
config.add_import("veomni.ops.config", names=["resolve_op_impl"])
config.add_import(
    "veomni.utils.seqlen_pos_transform_utils",
    names=["coalesce_tail_padding_cu_seqlens", "prepare_fa_kwargs_from_position_ids"],
)


# SituGLU (instance-local VeomniOp, fused Triton kernel)


@config.override_method(
    "SituAndMul.__init__",
    description="Construct a local situ_and_mul VeomniOp",
)
def situ_and_mul_init_patched(self, beta: float = 1.0, linear_beta: float | None = None):
    nn.Module.__init__(self)
    self.beta = beta
    self.linear_beta = linear_beta
    self.veomni_situ_and_mul = VeomniOp("situ_and_mul", "kimi_k3", resolve_op_impl("situ_and_mul_implementation"))


@config.override_method(
    "SituAndMul.forward",
    description="Always call the local situ_and_mul VeomniOp",
)
def situ_and_mul_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    return self.veomni_situ_and_mul(x, beta=self.beta, linear_beta=self.linear_beta)


# KDA chunked linear attention (instance-local VeomniOp, fused AscendC kernel)

# Patch 12: the source binds the vendored eager ``chunk_kda`` reference onto
# each ``KimiDeltaAttention`` instance (``self.veomni_chunk_kda``) and calls it
# from both training paths. The override below rebinds that handle to a
# ``chunk_kda`` ``VeomniOp`` (``kimi_k3`` variant) resolved from the ops
# selection installed at model-build time, mirroring the rms_norm / situ_and_mul
# instance-local pattern. Signatures are kwarg-compatible; extra K3 args
# (``A_log``/``dt_bias``/``transpose_state_layout``) pass through.


@config.override_method(
    "KimiDeltaAttention.__init__",
    description="Bind the instance-local chunk_kda VeomniOp (replaces the vendored eager reference)",
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
    # Modification: rebind the source's eager ``chunk_kda`` reference to the
    # ops-configured ``VeomniOp`` (instance-local kernel handle; the fused
    # AscendC kernel when configured, the modeling reference otherwise).
    self.veomni_chunk_kda = VeomniOp("chunk_kda", "kimi_k3", resolve_op_impl("chunk_kda_implementation"))


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


# RMSNorm (instance-local VeomniOp, functional fused kernel)


@config.override_method(
    "KimiRMSNorm.__init__",
    description="Construct a local rms_norm VeomniOp",
)
def kimi_rmsnorm_init_patched(self, hidden_size, eps=1e-6):
    nn.Module.__init__(self)
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.veomni_rms_norm = VeomniOp("rms_norm", "standard", resolve_op_impl("rms_norm_implementation"))


@config.override_method(
    "KimiRMSNorm.forward",
    description="Always call the local rms_norm VeomniOp",
)
def kimi_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)


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
        if self.is_situ:
            beta, linear_beta = _get_situ_activation_params(config)
            self.act_fn = SituAndMul(beta=beta, linear_beta=linear_beta)
        else:
            self.act_fn = ACT2FN[config.hidden_act]
        self.veomni_moe = VeomniOp(
            "moe_experts",
            "kimi_k3" if self.is_situ else "standard",
            resolve_op_impl("moe_implementation"),
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
    self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    self.use_attn_residuals = getattr(config, "attn_res_block_size", None) is not None
    if self.use_attn_residuals:
        self.output_attn_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    # Modification: respect the caller-requested attention implementation
    # (eager / veomni_* NPU kernels) instead of forcing flash_attention_2.
    # KimiMLAAttention already dispatches through ALL_ATTENTION_FUNCTIONS with
    # an eager fallback, so any registered implementation works.
    if getattr(config, "_attn_implementation", None) is None:
        config._attn_implementation = "flash_attention_2"
    self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
    self.gradient_checkpointing = False
    # Initialize weights and apply final processing
    self.post_init()


# KimiK3ForConditionalGeneration.__init__ (propagate _attn_implementation) --
# NOTE: patchgen's codegen has no INIT_MODIFICATION applicator (modify_init is
# a silent no-op beyond the header listing), so this must be a full
# override_method replacement of __init__.


@config.override_method(
    "KimiK3ForConditionalGeneration.__init__",
    description="Propagate the top-level `_attn_implementation` to text_config and "
    "vision_config before construction (the MoonViT3d tower registers "
    "'eager'/'flash_attention_2'/'veomni_flash_attention_2' in "
    "VL_VISION_ATTENTION_FUNCTIONS -- HF K3 has no vision 'sdpa', so an "
    "'sdpa' request falls back to 'eager' for vision); also override "
    "_no_split_modules to keep vision/projector in the root FSDP2 unit "
    "(asymmetric multimodal batches deadlock otherwise)",
)
def kimi_k3_for_conditional_generation_init_patched(self, config):
    # Modification: propagate the attention implementation to the sub-configs
    # before submodule construction. The MoonViT3d vision tower has no 'sdpa'
    # implementation (VL_VISION_ATTENTION_FUNCTIONS registers only
    # 'eager'/'flash_attention_2'), so an 'sdpa' request falls back to 'eager'
    # for the vision tower.
    if getattr(config, "_attn_implementation", None) is not None:
        config.text_config._attn_implementation = config._attn_implementation
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            vision_impl = config._attn_implementation
            if vision_impl == "sdpa":
                vision_impl = "eager"
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
    impl = resolve_op_impl("cross_entropy_loss_implementation")
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
    impl = resolve_op_impl("cross_entropy_loss_implementation")
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
        sp_group = get_parallel_state().sp_group
        input_ids = gather_outputs(input_ids, gather_dim=1, group=sp_group)
        if attention_mask is not None:
            attention_mask = gather_outputs(attention_mask, gather_dim=1, group=sp_group)
        if labels is not None:
            labels = gather_outputs(labels, gather_dim=1, group=sp_group)
        if position_ids is not None:
            position_ids = gather_outputs(position_ids, gather_dim=1, group=sp_group)

    # Modification (SP label alignment): the SP collator pre-shifts labels
    # (``[..., 1:]`` + tail pad) and ForCausalLMLoss then consumes them as-is
    # under SP. K3's merge, however, scatters labels position-wise onto text
    # slots only: a pre-shifted label sitting on a media placeholder slot is
    # dropped and the first post-image text token's supervision would strand
    # inside the IGNORE-filled vision span. Undo the pre-shift here so the
    # merge scatters raw labels; the single causal shift is re-applied to the
    # merged labels in the SP slicing block below (and stays inside the loss
    # helper on the non-SP path), so every path applies exactly one shift.
    if get_parallel_state().sp_enabled and labels is not None:
        head = labels.new_full((labels.shape[0], 1), self.config.ignore_index)
        labels = torch.cat([head, labels[..., :-1]], dim=1).contiguous()

    pre_merge_position_ids = None
    merged_first_positions = None
    if inputs_embeds is None:
        # 1. Extra the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text and images
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
            pre_merge_position_ids = position_ids
            inputs_embeds, attention_mask, labels, position_ids, merged_first_positions = (
                self._merge_input_ids_with_image_features(
                    image_features,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    labels,
                )
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

    # Modification (packed varlen): the merge recomputes position_ids as a
    # continuous arange over the merged row, erasing the collator's
    # ``position_ids == 0`` segment markers. Map every pre-merge boundary
    # (packed-sample starts and SP tail pads) to its merged coordinate through
    # the merge's occupation table, rebuild segment-aware position_ids, and
    # refresh the FlashAttention cu_seq_lens kwargs: on an all-ones packed
    # mask the softmax attention and the recurrent KDA state would otherwise
    # run as one long sequence and leak across packed samples. The
    # collator's pre-merge cu_seq_lens cannot be reused because the merge
    # expands media placeholders. Padded batches (non-all-ones masks) keep
    # the attention-mask path, matching the collator contract of publishing
    # cu_seq_lens only for the all-ones packed form.
    fa_kwargs = {}
    # The collator precomputes the all-ones check host-side and publishes it
    # as ``attention_mask_is_all_ones``; reading the flag avoids the per-step
    # host-device sync of ``bool(attention_mask.all())``. The flag is None
    # only when the mask sits on device without a host copy (or the caller
    # hand-built the batch, e.g. tests) -- fall back to the device check.
    _mask_all_ones = kwargs.get("attention_mask_is_all_ones")
    if _mask_all_ones is None and attention_mask is not None:
        _mask_all_ones = bool(attention_mask.all())
    if (
        merged_first_positions is not None
        and pre_merge_position_ids is not None
        and attention_mask is not None
        and _mask_all_ones
    ):
        batch_size, merged_len = inputs_embeds.shape[:2]
        boundary = pre_merge_position_ids == 0
        row_offsets = torch.arange(batch_size, device=inputs_embeds.device) * merged_len
        starts = (merged_first_positions + row_offsets.unsqueeze(1))[boundary]
        is_start = torch.zeros(batch_size * merged_len, dtype=torch.bool, device=starts.device)
        is_start[starts] = True
        is_start[row_offsets] = True
        coords = torch.arange(batch_size * merged_len, device=starts.device)
        segment_start = coords.masked_fill(~is_start, 0).cummax(-1).values
        position_ids = (coords - segment_start).view(batch_size, merged_len)
        (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
            position_ids
        )
        # Coalesce the trailing pad segments (one boundary per pad token)
        # into a single tail segment: npu_fusion_attention rejects the
        # uncoalesced per-token form (mirrors the collator-side helper). The
        # pad count comes from the collator's published ``tail_padding_length``
        # (set whenever it appends SP / pad-to-length tail pads) -- deriving
        # it from trailing ``position_ids == 0`` would conflate real 1-token
        # packed samples with pads and merge a genuine sample boundary.
        tail_pads = int(kwargs.get("tail_padding_length", 0))
        if tail_pads > 0:
            cu_seq_lens_q = coalesce_tail_padding_cu_seqlens(cu_seq_lens_q, tail_pads)
            cu_seq_lens_k = cu_seq_lens_q
            seqlens = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
            max_length_q = int(seqlens.max().item()) if seqlens.numel() else 0
            max_length_k = max_length_q
        fa_kwargs["cu_seq_lens_q"] = cu_seq_lens_q
        fa_kwargs["cu_seq_lens_k"] = cu_seq_lens_k
        fa_kwargs["max_length_q"] = max_length_q
        fa_kwargs["max_length_k"] = max_length_k
        # The KDA layers and the veomni attention adapters read the varlen
        # layout from this kwarg (full-sequence coordinates, matching the
        # gathered sequences inside the layers).
        fa_kwargs["cu_seqlens"] = cu_seq_lens_q
    elif merged_first_positions is None:
        # No merge ran (text-only batch): the collator's cu_seq_lens are
        # still valid -- forward them (plus the coalesced linear-attention
        # alias) exactly like KimiLinearForCausalLM.forward does.
        for _fa_key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
            if _fa_key in kwargs:
                fa_kwargs[_fa_key] = kwargs[_fa_key]
        _linear_cu = kwargs.get("linear_attn_cu_seq_lens_q")
        if _linear_cu is None:
            _linear_cu = kwargs.get("cu_seq_lens_q")
        if _linear_cu is not None:
            fa_kwargs["cu_seqlens"] = _linear_cu

    # Modification (SP): slice the merged embeds / labels back to
    # the local shard and hand the LM a full-sequence cache_position so
    # position ids and causal masks are built at full length (matching the
    # gathered sequences inside the attention layers).
    if get_parallel_state().sp_enabled:
        sp_group = get_parallel_state().sp_group
        full_seq_len = inputs_embeds.shape[1]
        inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, group=sp_group)
        if labels is not None:
            # Modification: VeOmni's ForCausalLMLoss consumes labels as-is
            # under SP (dataloader-side pre-shift contract). The pre-shift
            # was undone before the merge (see the SP label alignment block
            # above), so this shift is the single causal shift over the
            # merged labels, applied before slicing to shards. The slice pads
            # the label row to an even shard split; those pad slots must read
            # as ignored, not as token 0.
            labels = nn.functional.pad(labels, (0, 1), value=self.config.ignore_index)[..., 1:].contiguous()
            labels = slice_input_tensor(labels, dim=1, group=sp_group, padding_value=self.config.ignore_index)
        cache_position = torch.arange(full_seq_len, device=inputs_embeds.device)
        # Modification (SP, odd merged rows): slice_input_tensor pads the merged
        # row to a multiple of sp_size so the shards split evenly. The collator
        # cannot predict that padding target -- the merge expands media
        # placeholders (+3 per image here), so an even pre-merge row can merge
        # to an odd length. Those slice pads ride through the layers' SP
        # re-gathers as trailing tokens past every cu boundary: the KDA short
        # convs honor cu and drop them from q/k/v while the unconvolved gate
        # keeps them, a shape mismatch the chunk kernel rejects. Extend the
        # last cu segment to cover the slice pads (never append a new boundary)
        # so every varlen consumer sees cu[-1] == padded length. Merging --
        # rather than publishing a separate pad segment like the collator's
        # tail-pad contract -- is deliberate: two adjacent trailing 1-token
        # segments (collator tail pad + slice pad) make the AscendC chunk_kda
        # kernel return non-finite values for earlier *real* segments, while a
        # single trailing 1-token segment is fine. The pads sit after every
        # real token, so causality keeps real outputs unchanged and the pad
        # slots' outputs are discarded (their labels are ignore_index).
        slice_pad = (-full_seq_len) % get_parallel_state().sp_size
        if slice_pad:
            for _cu_key in ("cu_seq_lens_q", "cu_seq_lens_k", "cu_seqlens"):
                _cu = fa_kwargs.get(_cu_key)
                if _cu is not None:
                    fa_kwargs[_cu_key] = torch.cat([_cu[:-1], _cu[-1:] + slice_pad])
            # The extended last segment can now be the longest one; bump the
            # FA launch bound by the pad count (an upper bound -- the previous
            # max may or may not have been the last segment -- which is always
            # safe for a launch grid).
            for _max_key in ("max_length_q", "max_length_k"):
                _max_len = fa_kwargs.get(_max_key)
                if _max_len is not None:
                    fa_kwargs[_max_key] = _max_len + slice_pad

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
    "KimiK3ForConditionalGeneration.get_position_id_func",
    description="VLM trainer data hook: K3 derives positions inside the modeling "
    "(vision divided-fixed pos-emb from grid_thws, NoPE MLA, recurrent KDA); "
    "return an arange stub for generic consumers",
)
def kimi_k3_get_position_id_func_patched(self):
    def _kimi_k3_position_ids(input_ids=None, **kwargs):
        if input_ids is None:
            return None
        return torch.arange(input_ids.shape[-1], device=input_ids.device).expand_as(input_ids)

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


@config.override_method(
    "KimiK3ForConditionalGeneration.get_dyn_bsz_length_fn",
    description="Dynamic-batching length hook: K3 expands one media_pad placeholder per image "
    "to grid_t*h*w/(mh*mw) merged features inside the model (occupation table), so the "
    "packer's attention_mask-only budget must add the expanded vision length. "
    "Closes over merge ints only -- picklable for "
    "worker-side dynamic batching.",
)
def kimi_k3_get_dyn_bsz_length_fn_patched(self):
    # Return a picklable partial of the module-level length fn in
    # veomni.data.data_transform: trainer-state checkpointing pickles the
    # dynamic-batching strategy (incl. this fn) by reference, so closures
    # defined here (method locals) would break torch.save.
    from functools import partial

    from veomni.data.data_transform import kimi_k3_expanded_length

    # Modification: ``merge_kernel_size`` lives on the *vision* config (the
    # MoonViT patch merger), not the top-level KimiK3Config -- reading it
    # there always fell back to the (2, 2) default.
    merge = getattr(self.config.vision_config, "merge_kernel_size", (2, 2)) or (2, 2)
    return partial(kimi_k3_expanded_length, int(merge[0]), int(merge[1]))


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
