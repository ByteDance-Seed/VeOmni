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

Patches (stage-2 skeleton):
1.  ``KimiRMSNorm.forward``  --OpSlot guard for fused RMSNorm.
2.  ``KimiMoEGate.forward``  --drop the inference-only ``assert not self.training``.
3.  ``KimiBlockSparseMLP``   --replaced by ``PatchedKimiK3Experts``: fused
    ``[E, 2I, H]`` / ``[E, H, I]`` expert weights plus the VeOmni fused-MoE
    OpSlot dispatch. SituGLU experts dispatch to the situ fused-MoE kernel
    (``moe_experts`` op, ``kimi_k3`` variant: NPU group-gemm + SituGLU, EP
    alltoall-aware); other activations may take the standard fused path once
    an OpSlot is bound.
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
7.  ``KimiLinearForCausalLM.forward`` --fused CE path via
    ``veomni_causal_lm_loss`` / the patched ``loss_function``; returns
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
11. ``SituAndMul.forward`` --OpSlot guard for the fused NPU Triton SituGLU
    kernel (``situ_and_mul`` op, ``kimi_k3`` variant).
12. ``chunk_kda`` --name hijack (drop-import + post-import redefinition) with
    an OpSlot guard: routes the KDA chunked linear attention to the fused
    Triton kernel from the triton-ascend-kernels package (``chunk_kda`` op,
    ``kimi_k3`` variant) when bound; eager ``chunk_kda_naive`` stays as the
    fallback.
13. ``KimiK3ForConditionalGeneration.forward`` --ulysses sequence-parallel
    data flow: gather the per-rank sequence shards (input_ids / mask /
    labels / position_ids) to the full sequence before the multimodal merge
    (vision tower runs redundantly per rank), slice the merged embeds and
    labels back to the local shard, and hand the LM a full-sequence
    ``cache_position`` so causal masks / position ids match the gathered
    sequences inside the attention layers.
14. ``KimiDeltaAttention.forward`` / ``KimiMLAAttention.forward`` (source
    fixes #13/#14) --SP gather at entry / narrow at exit: linear attention
    is sequential across tokens and MLA is full-sequence, so both run on the
    full (gathered) sequence on every SP rank and narrow the output back to
    the local shard. The MLA shard-sized causal mask is dropped (all-ones
    training packs); the KDA cu_seqlens derive from the full-sequence mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

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

config.add_post_import_block(
    """
    # OpSlot declarations
    # These are bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_rms_norm = OpSlot("rms_norm", "standard")
    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    veomni_moe_experts_situ = OpSlot("moe_experts", "kimi_k3")
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_situ_and_mul = OpSlot("situ_and_mul", "kimi_k3")
    veomni_chunk_kda = OpSlot("chunk_kda", "kimi_k3")
    """
)


# SituGLU (OpSlot guard, fused Triton kernel)


@config.override_method(
    "SituAndMul.forward",
    description="OpSlot guard: route SituGLU to the fused NPU Triton kernel "
    "(situ_and_mul op, kimi_k3 variant) when bound",
)
def situ_and_mul_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    # Modification: OpSlot guard --use the fused Triton kernel when bound.
    # Signature matches the eager SituAndMul: x is [..., 2d] with gate on the
    # left half and up on the right half.
    if veomni_situ_and_mul.use_non_eager_impl:
        return veomni_situ_and_mul(x, beta=self.beta, linear_beta=self.linear_beta)
    # Original HF code below, unchanged.
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
    if self.linear_beta is not None:
        up = self.linear_beta * torch.tanh(up / self.linear_beta)
    return (situ_a * up).to(x.dtype)


# KDA chunked linear attention (OpSlot guard, fused triton-ascend-kernels kernel)

# Patch 12: hijack the ``chunk_kda`` name. ``chunk_kda`` is *imported* from
# ``kda_fallback`` (not defined in the source module), so FUNCTION_REPLACEMENT
# cannot touch it. Drop the import and re-define the name in a post-import
# block: the guard dispatches to the fused triton-ascend-kernels chunk_kda
# when the ``veomni_chunk_kda`` OpSlot is bound, and falls back to the eager
# ``chunk_kda_naive`` otherwise. Signatures are kwarg-compatible; extra K3
# args (``A_log``/``dt_bias``/``transpose_state_layout``) pass through.

config.drop_import_names("chunk_kda")
config.add_post_import_block(
    """
    # --- Patch 12: chunk_kda OpSlot guard (KDA fused Triton kernel) ---
    from veomni.models.transformers.kimi_k3.source.kda_fallback import (
        chunk_kda as _kda_chunk_kda_eager,
    )

    def chunk_kda(q, k, v, g, beta, **kwargs):
        # OpSlot guard: use the fused triton-ascend-kernels implementation
        # when bound; otherwise the eager naive implementation above.
        if veomni_chunk_kda.use_non_eager_impl:
            return veomni_chunk_kda(q=q, k=k, v=v, g=g, beta=beta, **kwargs)
        return _kda_chunk_kda_eager(q=q, k=k, v=v, g=g, beta=beta, **kwargs)
    """
)


# RMSNorm (OpSlot guard, functional fused kernel)


@config.override_method(
    "KimiRMSNorm.forward",
    description="OpSlot guard for fused RMSNorm (standard formulation)",
)
def kimi_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # Modification: OpSlot guard --use fused RMSNorm kernel when bound.
    if veomni_rms_norm.use_non_eager_impl:
        return veomni_rms_norm(hidden_states, self.weight, self.variance_epsilon)
    # Original HF code below, unchanged.
    dtype = hidden_states.dtype
    x = hidden_states.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    return self.weight * x.to(dtype)


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
        "MoE path. SituGLU experts dispatch through the situ OpSlot (kimi_k3 variant, NPU "
        "group-gemm + SituGLU activation, EP-aware); other activations dispatch through the "
        "standard OpSlot when bound."
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        # Modification: OpSlot guards for the fused MoE kernels. SituGLU
        # experts dispatch to the situ variant (``moe_experts`` op,
        # ``kimi_k3`` variant: NPU group-gemm with the SituGLU activation and
        # alltoall dispatch/combine under EP); other activations take the
        # standard fused path. Eager expert loop stays the fallback.
        if self.is_situ:
            if veomni_moe_experts_situ.use_non_eager_impl:
                return veomni_moe_experts_situ(self, hidden_states, top_k_index, top_k_weights)
        elif veomni_moe_experts_forward.use_non_eager_impl:
            return veomni_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx])
            if self.is_situ:
                # SituAndMul splits the trailing dim in half itself:
                # situ(beta * tanh(gate/beta) * sigmoid(gate) * up).
                current_hidden_states = self.act_fn(gate_up)
            else:
                gate, up = gate_up.chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


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
    "vision_config before construction (the MoonViT3d tower registers only "
    "'eager'/'flash_attention_2' in VL_VISION_ATTENTION_FUNCTIONS -- HF K3 has "
    "no vision 'sdpa', so an 'sdpa' request falls back to 'eager' for vision); "
    "also override _no_split_modules to keep vision/projector in the root FSDP2 "
    "unit (asymmetric multimodal batches deadlock otherwise)",
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


# KimiLinearForCausalLM.forward (fused CE)


@config.override_method(
    "KimiLinearForCausalLM.forward",
    description="Fused cross-entropy path via veomni_causal_lm_loss / patched loss_function",
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
    )

    hidden_states = outputs[0]
    if generation_mode:
        hidden_states = hidden_states[:, -1:]

    # Modification: fused cross-entropy path.
    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        # Modification: OpSlot guard for cross-entropy loss.
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            # Modification: VeOmni's patched `loss_function` (via LOSS_MAPPING)
            # returns (loss, logits, fused_linear_aux); unpack to match the
            # OpSlot branch above.
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            if fused_linear_aux is not None:
                # fused_linear_aux path empties loss/logits slots; clear the local 3D
                # logits so output mirrors the OpSlot branch's contract.
                logits = None
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

    # Modification (stage 5, SP): ulysses sequence parallel -- the sequence
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

    if inputs_embeds is None:
        # 1. Extra the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        if pixel_values is not None and len(pixel_values) > 0 and input_ids.shape[1] != 1:
            image_features = self._extract_image_features(pixel_values, grid_thws)
            if self.mm_projector:
                image_features = self.mm_projector(image_features)

            inputs_embeds = inputs_embeds.to(image_features[0].dtype)  # num_tokens, embed_dim
            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                image_features,
                inputs_embeds,
                input_ids,
                attention_mask,
                labels,
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

    # Modification (stage 5, SP): slice the merged embeds / labels back to
    # the local shard and hand the LM a full-sequence cache_position so
    # position ids and causal masks are built at full length (matching the
    # gathered sequences inside the attention layers).
    if get_parallel_state().sp_enabled:
        sp_group = get_parallel_state().sp_group
        full_seq_len = inputs_embeds.shape[1]
        inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, group=sp_group)
        if labels is not None:
            # Modification (fix #18): VeOmni's ForCausalLMLoss skips its internal
            # causal shift when sp_enabled (it expects dataloader-side pre-shift).
            # K3 expands labels inside the merge, so the shift must happen here,
            # on the full post-expansion labels, before slicing to shards.
            labels = nn.functional.pad(labels, (0, 1), value=self.config.ignore_index)[..., 1:].contiguous()
            labels = slice_input_tensor(labels, dim=1, group=sp_group)
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
        # Modification: OpSlot guard for cross-entropy loss.
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.language_model.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.language_model.lm_head(hidden_states)
            # Modification: VeOmni's patched `loss_function` (via LOSS_MAPPING)
            # returns (loss, logits, fused_linear_aux); unpack to match the
            # OpSlot branch above.
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.language_model.lm_head.weight,
                **kwargs,
            )
            if fused_linear_aux is not None:
                logits = None
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
    if isinstance(module, KimiDeltaAttention):
        # Modification: HF leaves `dt_bias` as uninitialized `torch.empty`;
        # zero it so the KDA gate stays finite under random init (checked-in
        # weights overwrite this). A_log and the ShortConvolution weights are
        # already initialized at construction.
        module.dt_bias.data.zero_()
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


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
    if isinstance(module, KimiDeltaAttention):
        # Modification: HF leaves `dt_bias` as uninitialized `torch.empty`;
        # zero it so the KDA gate stays finite under random init.
        module.dt_bias.data.zero_()

    if hasattr(module, "class_embedding"):
        module.class_embedding.data.normal_(mean=0.0, std=std)

    if isinstance(module, (nn.Linear, nn.Conv2d)):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


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
KimiMoEGate = None  # noqa: E305
KimiMLP = None  # noqa: E305
KimiRMSNorm = None  # noqa: E305
KimiDecoderLayer = None  # noqa: E305
KimiBlockSparseMLP = None  # noqa: E305
KimiDeltaAttention = None  # noqa: E305
