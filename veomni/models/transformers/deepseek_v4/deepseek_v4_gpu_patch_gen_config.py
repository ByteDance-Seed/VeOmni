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
- ``apply_rotary_pos_emb`` — the generated definition stays the eager
  reference. DeepSeek-V4 uses a *partial* RoPE (the trailing
  ``qk_rope_head_dim`` slice only, with the leading nope channels untouched)
  plus an interleaved ``repeat_interleave(2)`` cos/sin layout that
  ``liger_rotary_pos_emb`` does not implement — SKILL.md flags this exact
  case (partial_rotary -> liger NaN) — so ``device_patch.py`` disables the
  registry-default Liger backend and offers a model-specific fused Triton
  kernel instead, selected by ``rotary_pos_emb_implementation=triton``.
  Because every call site (Q, MQA KV, the inverse rotation on the attention
  output, the indexer Q, and the compressors) resolves the module global,
  that one rebind covers all of them; ``compress_packed_windows`` takes it
  as an ``apply_rope`` argument for the same reason.
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
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACache,
    DeepseekV4HCACache,
    apply_rotary_pos_emb,
    load_balancing_loss_func,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.distributed.sequence_parallel import reduce_sequence_parallel_loss
from veomni.models.transformers.deepseek_v4.packed_utils import (
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
from veomni.ops import fused_moe_forward
from veomni.ops.dispatch import OpsConfigSlot, OpSlot
from veomni.ops.kernels.deepseek_v4 import sparse_attn_tilelang, sparse_mqa_target_fwd, v4_lighting_indexer
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import MoeCausalLMOutputWithLogProbs, MoeModelOutputWithIndexerKL
from veomni.utils.moe_router_replay import get_active_replay, maybe_replay_indices


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
    names=[
        "FusedLinearAuxOutput",
        "FusedLinearAuxOutputMixin",
        "MoeCausalLMOutputWithLogProbs",
        "MoeModelOutputWithIndexerKL",
    ],
)
config.drop_import_names("MoeCausalLMOutputWithPast")

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


@config.add_helper
def _indexer_loss_enabled(module) -> bool:
    """Whether to build the indexer KL, refusing loudly on unsupported setups.

    Silence is the failure mode worth designing against here: every unsupported
    configuration below would otherwise train the indexer on a wrong signal, or
    on none, while the loss curve looked entirely reasonable.

    Read off ``module.config``, where the objective and its weight are declared,
    beside the ``output_router_logits`` / ``router_aux_loss_coef`` pair this model's
    other auxiliary objective is configured through and folded in from. ``module`` is
    therefore anything holding the model config -- a ``DeepseekV4Attention`` or the
    ``DeepseekV4Model`` itself. Per-instance rather than module-global, so two models
    built from this one generated module (a DPO policy and its reference) can differ.

    ``getattr`` with a default, not attribute access: ``MODELING_BACKEND=hf`` skips
    ``MODEL_CONFIG_REGISTRY`` and hands the patched classes an upstream
    ``DeepseekV4Config``, which declares neither field. Undeclared is not the same as
    absent, though, and the difference is exactly what the subclass buys. Keys found
    in a ``config.json`` are ``setattr``ed by ``from_dict`` whether the class declared
    them or not, so a flag-on checkpoint carries the objective into an upstream config
    unaided; a *kwarg* is applied only if the attribute already exists, so enabling the
    objective from YAML on a base checkpoint that never had the key is the one path
    that silently drops without the declaration. What the default here covers is the
    remaining case -- neither declared nor on disk -- where off is the only safe
    reading of a config that cannot express the objective.

    A non-positive coefficient counts as off, matching Megatron's
    ``coeff is not None and coeff > 0`` (``training/training.py:3317``). It is read
    here rather than only at the fold-in because this predicate is what decides the
    teacher recompute as well: ``loss + 0.0 * kl`` is the right *value* while still
    building the graph, so the backward writes a zero ``p.grad`` onto every indexer
    parameter -- and Muon skips only ``p.grad is None`` (``muon.py:902``) while
    ``_apply_ortho`` decays whatever it steps (``:1005-1006``), which is weight decay
    on 226M otherwise-frozen parameters, at the full cost of the teacher kernel.
    Gating here makes ``dsa_indexer_loss_coef: 0.0`` cost exactly what
    ``dsa_indexer_loss: false`` costs.

    Before the refusals below, not after: a user who switched the term off with the
    coefficient has not asked for a TileLang indexer, and refusing their run over the
    configuration of a feature they just disabled would be advice about the wrong
    thing.

    ``DeepseekV4Config.validate_build_prerequisites`` refuses the two implementation fields at
    model-build time, before any rank reads a weight, so a launched run is told there
    rather than here. These stay because this predicate also covers the paths that
    never pass through ``build_foundation_model`` -- a model constructed straight from
    ``_from_config`` -- and because the parallel-state refusals below have no earlier
    home: the state is installed by then, but a model-agnostic gate cannot know that
    *this* model has no context-parallel indexer path.
    """
    if not getattr(module.config, "dsa_indexer_loss", False):
        return False
    if getattr(module.config, "dsa_indexer_loss_coef", 1.0) <= 0:
        return False
    if veomni_dsa_indexer_implementation.value != "tilelang":
        raise ValueError(
            "dsa_indexer_loss requires dsa_indexer_implementation='tilelang'; the eager "
            "indexer discards its scores, so the loss would have nothing to train against"
        )
    if veomni_dsa_attention_implementation.value != "tilelang":
        raise ValueError(
            "dsa_indexer_loss requires dsa_attention_implementation='tilelang'; the teacher "
            "distribution is derived from the TileLang attention LSE"
        )
    state = get_parallel_state()
    if state.ulysses_size > 1:
        raise ValueError(
            f"dsa_indexer_loss requires ulysses_size=1, got ulysses_size={state.ulysses_size}: under "
            "Ulysses each rank holds a head shard, so the head sum in the teacher would be partial."
        )
    if state.cp_size > 1:
        raise ValueError(
            f"dsa_indexer_loss requires cp_size=1, got cp_size={state.cp_size}: DeepSeek-V4's forward "
            "has no context-parallel path, so each rank would treat its sequence shard as a whole "
            "sequence and the teacher would be built from the resulting attention."
        )
    return True


@config.add_helper
def _builds_indexer_kl(module) -> bool:
    """Whether *this attention layer* builds a KL, and so returns four values.

    ``module`` is a ``DeepseekV4Attention``. Three call sites act on this answer --
    the attention forward that returns the extra values, the decoder layer that
    unpacks them, and the model loop that accumulates them -- and they are in three
    different functions. They read this predicate rather than each re-deriving the
    condition, because a copy that goes stale in any one of them is an arity
    mismatch: gating the decoder layer on ``_indexer_loss_enabled`` alone would
    four-unpack the two-tuple every sliding and HCA layer returns, which is three
    of the four layers of the reference checkpoint.

    The attention forward also hands its answer *down*, to the compressor and the
    indexer, whose returns change arity by the same decision (``_split_indexer_output``).
    They are passed it rather than calling this because neither keeps the model config
    the predicate reads, and because one evaluation per layer per forward is one fewer
    thing that can disagree with itself mid-call.

    ``_indexer_loss_enabled`` comes first so that its refusals fire on every layer
    type rather than only on the ones carrying an indexer: a model configured for
    the loss but built without a single CSA layer would otherwise accept the flag
    and train nothing. The layer type is what then keeps HCA and sliding layers on
    their two-value return -- only a CSA layer carries a Lightning Indexer, so only
    it has a student to train, and the others' compressors hand back a perfectly
    ordinary ``CompressedCandidates`` carrying causal ranges instead of scores.

    The test is on the layer type rather than on ``module.compressor.indexer``
    existing, because the two fail in opposite directions. ``layer_type`` comes
    from the checkpoint's ``layer_types``, so a rename of the compressor's
    attribute breaks the KL loudly at the attribute access in the attention
    forward; keying the gate on that attribute's *name* would instead turn the
    whole auxiliary objective into a no-op, with no error and no change of arity --
    a plausible loss curve training nothing, which is the failure class this
    feature exists to prevent.
    """
    return _indexer_loss_enabled(module) and module.layer_type == "compressed_sparse_attention"


@config.add_helper
def _split_indexer_output(indexer_output, build_indexer_loss: bool):
    """Unpack ``DeepseekV4Indexer.forward``'s return, whose arity follows the flag.

    The indexer returns ``(top_k_indices, index_score)`` only when the loss is on, so
    that a flag-off forward keeps exactly the arity every existing caller unpacks. The
    two compressor call sites read it through here rather than through an
    ``isinstance(..., tuple)`` test, so that a disagreement surfaces as an unpacking
    error at the call rather than as a silently missing student distribution much
    later.

    ``build_indexer_loss`` is the same value the compressor passed *into* the indexer
    on the line above, which is the point: neither the indexer nor the compressor
    evaluates the predicate. ``DeepseekV4Attention.forward`` evaluates it once per
    layer per forward and hands it down, so the producer's arity and the consumer's
    unpacking cannot disagree -- there is only one answer, and it arrives by argument.
    Neither module holds the model config to re-derive it from in any case: the
    compressor and the indexer take a config in ``__init__`` and keep only scalars off
    it.
    """
    if not build_indexer_loss:
        return indexer_output, None
    top_k_indices, index_score = indexer_output
    return top_k_indices, index_score


@config.add_helper
def indexer_kl_terms(index_score: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query ``KL(target || softmax(index_score))`` for DeepSeek-V3.2 eq. (4), and
    the zero-information reference to read it against.

    Args:
        index_score: [B, S, C] indexer scores at the selected slots, -inf at misses
        target:      [B, S, C] fp32, zero at misses, and per row either L1-normalised
                     or identically zero where the teacher had no mass to give

    Returns:
        ``(kl, uniform_kl)``, both [B, S] fp32. ``uniform_kl`` is detached: it is a
        metric only and must never reach the objective.
    """
    # A query whose compressed slots are *all* misses scores every one of them
    # ``-inf``, and ``log_softmax`` of such a row is NaN. Masking that after the
    # fact is not enough: the mask below hides the NaN from the returned value, but
    # ``log_softmax``'s backward computes ``g - softmax * g.sum(-1)`` with
    # ``softmax = exp(NaN)``, so even the zero gradient such a row receives comes
    # back NaN -- and the indexer's own backward propagates it, because it forms
    # ``grad * relu(logits)`` and ``NaN * 0`` is NaN. The row is therefore
    # neutralised on the way *in*. It is the common case, not a corner one: the
    # first ``compress_rate - 1`` positions of every packed sample have no complete
    # compression window behind them.
    scoreable = torch.isfinite(index_score)
    # Two ways a row has nothing to teach, and both have to be excluded from *both*
    # returned terms rather than only from the KL. A row scored entirely ``-inf`` is
    # the NaN case above. A row the teacher gave no mass -- every slot a miss, or
    # every selected logit so far below the LSE that ``exp`` underflowed -- would
    # otherwise contribute 0 to the KL and a full ``log(n_candidates)`` to the
    # reference, which is not a student that captured everything; it is a row with
    # nothing to capture, and leaving it in the denominator alone flatters the
    # captured fraction by exactly the rows where the objective did no work.
    nothing_to_learn = ~scoreable.any(-1, keepdim=True) | (target.sum(-1, keepdim=True) <= 0)
    # Scalar zeros rather than ``torch.zeros_like``: the operand is only a zero, and
    # a materialised one is a full [B, S, C] fp32 tensor -- 50 MB each at S=24576,
    # C=512, about a third of the ~300 MB transient this function costs per CSA layer
    # call. ``torch.where`` promotes a Python float as a weak scalar, so the result
    # dtype is the fp32 of the other operand either way.
    scores = torch.where(nothing_to_learn, 0.0, index_score.float())
    log_q = torch.log_softmax(scores, dim=-1)
    log_target = torch.log(target.clamp_min(torch.finfo(torch.float32).tiny))
    # ``log_q`` is -inf exactly where ``target`` is 0, and 0 * -inf is NaN, so the
    # zero-mass slots have to be masked rather than merely multiplied out.
    contributions = torch.where(target > 0, target * (log_target - log_q), 0.0)
    # The scale the KL has to be read against. ``log(n_candidates) - H(target)`` is
    # the KL a student would pay knowing the candidate set and nothing whatever about
    # which slot matters, so the KL alone says nothing until it is divided by this:
    # a plateau of 0.021 means one thing against a reference of 0.374 and another
    # against 0.02. ``n_candidates`` is the number of slots the student can score at
    # all -- the finite entries of ``index_score`` -- so both quantities are over the
    # same support and a row with one candidate correctly contributes 0.
    #
    # No mask on the entropy: ``clamp_min`` keeps ``log_target`` finite, so a
    # zero-mass slot contributes ``0 * log(tiny) == 0`` rather than the ``0 * -inf``
    # the KL above has to guard against.
    #
    # Detached, and nothing here could carry a graph in any case: ``target`` comes
    # from a forward-only TileLang interface with no ``autograd.Function``, and the
    # only tensor derived from ``index_score`` is an integer count. The ``detach``
    # is the contract rather than the mechanism -- this must not perturb a gradient
    # even if a future teacher becomes differentiable.
    neg_entropy = (target * log_target).sum(-1)
    uniform_kl = torch.where(
        nothing_to_learn.squeeze(-1),
        0.0,
        torch.log(scoreable.sum(-1).clamp_min(1).to(torch.float32)) + neg_entropy,
    )
    return contributions.sum(-1), uniform_kl.detach()


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
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = hidden_states.dtype
    post, comb, collapsed = self.attn_hc(hidden_states)
    # --- Patch.3 ---
    # The attention returns its KL and that KL's zero-information reference as third
    # and fourth values on exactly the layers ``_builds_indexer_kl`` selects, so this
    # reads the same predicate rather than restating the condition or testing the
    # length of what came back: a length test would read a stale two-tuple from a
    # broken gate as "no KL here" and train nothing, whereas an arity mismatch against
    # the predicate raises.
    builds_indexer_kl = _builds_indexer_kl(self.self_attn)
    if builds_indexer_kl:
        attn_output, _, indexer_kl, indexer_uniform = self.self_attn(self.input_layernorm(collapsed), **kwargs)
    else:
        attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
    # --- Patch.3 ---
    if veomni_mhc_post.use_non_eager_impl:
        hidden_states = veomni_mhc_post(attn_output, hidden_states, post, comb)
    else:
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

    post, comb, collapsed = self.ffn_hc(hidden_states)
    mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
    if veomni_mhc_post.use_non_eager_impl:
        output = veomni_mhc_post(mlp_output, hidden_states, post, comb)
    else:
        output = post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )
    # --- Patch.3 ---
    # The KL leaves as an element of the return value, never as an attribute on
    # ``self`` or on the hidden states. Gradient checkpointing wraps this call, and
    # a tensor that reaches the model loop by any route other than the checkpointed
    # function's return value carries no graph: under the reentrant implementation
    # the first forward runs inside ``torch.no_grad()``, and under either one the
    # recomputed forward's tensors are the ones the backward is built from. The
    # indexer would then receive no gradient at all while the logged KL fell.
    #
    # A bare tensor when there is no KL, rather than ``(output, None)``: that is
    # what every existing caller of a DeepSeek-V4 decoder layer unpacks, and the
    # flag-off path has to stay exactly what it was. It also keeps the checkpointed
    # return free of non-tensor leaves.
    if builds_indexer_kl:
        return output, indexer_kl, indexer_uniform
    return output
    # --- Patch.3 ---


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
    # --- Patch.3 ---
    # Accepted and ignored. ``DeepseekV4Attention.forward`` holds one compressor whose
    # class is chosen by layer type and calls it through a single call site, so the two
    # compressors have to take the same arguments; only the CSA one owns a Lightning
    # Indexer and so only it has anything to do with this. Defaulted, so an HCA
    # compressor called directly is unaffected.
    build_indexer_loss: bool = False,
    # --- Patch.3 ---
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
    build_indexer_loss: bool = False,
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
        indexer_output = self.indexer(
            hidden_states,
            q_residual,
            position_ids,
            past_key_values,
            layer_idx,
            packed_sequence_slices=packed_sequence_slices,
            packed_compression_metadata=packed_compression_metadata,
            build_indexer_loss=build_indexer_loss,
        )
        top_k_indices, indexer_scores = _split_indexer_output(indexer_output, build_indexer_loss)
        candidates = CompressedCandidates(topk_indices=top_k_indices, indexer_scores=indexer_scores)
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
    indexer_output = self.indexer(
        hidden_states,
        q_residual,
        position_ids,
        past_key_values,
        layer_idx,
        build_indexer_loss=build_indexer_loss,
    )
    top_k_indices, indexer_scores = _split_indexer_output(indexer_output, build_indexer_loss)
    candidates = CompressedCandidates(topk_indices=top_k_indices, indexer_scores=indexer_scores)
    block_bias = scatter_topk_block_bias(compressed_kv, top_k_indices, batch, seq_len) if build_block_bias else None
    return (compressed_kv, block_bias, candidates) if return_topk_indices else (compressed_kv, block_bias)


# ================================================================
# Patch: DeepseekV4Indexer.forward
# 1. Dispatch CUDA prefill/training index scoring to the TileLang Lightning
#    Indexer when ``dsa_indexer_implementation=tilelang``. Cache/decode and unusual
#    position layouts fall outside what that kernel accepts, and having been asked
#    for it explicitly this refuses rather than demoting to the eager scorer.
# 2. Under ``dsa_indexer_loss``, hand the per-slot index scores back next to the
#    selection so the auxiliary KL has a student to train, and detach the inputs so
#    that KL cannot reach the main model. The eager scorer discards those scores and
#    so cannot serve the objective, but needs no refusal of its own here: the gate
#    admits the objective only under ``tilelang``, and the dispatch refusal in (1)
#    then covers every call that TileLang cannot take.
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

        index_score, top_k_indices = v4_lighting_indexer(
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
# 3. Under ``dsa_indexer_loss`` on a CSA layer, return the indexer KL and its
#    zero-information reference as third and fourth values. The return
#    annotation states that arity, so a caller reads the contract off the
#    signature rather than off a comment -- keep the two in step.
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
) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
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
        veomni_dsa_attention_implementation.value == "tilelang"
        and past_key_values is None
        and q.is_cuda
        and q.dtype == torch.bfloat16
    )
    # ``DeepseekV4Model.forward`` withholds the dense mask exactly when the packed
    # metadata is sufficient to validate candidates on its own, so its absence is
    # the signal to take the mask-free path and skip every O(S^2) intermediate.
    mask_free_sparse = use_compact_sparse_indices and attention_mask is None
    # --- Patch.3 ---
    # Evaluated before the compressor rather than beside its consumer below, because
    # the compressor and the indexer under it change return arity on this same answer
    # and are handed it rather than deriving it. It is also where the gate's refusals
    # come from, so an unsupported configuration is rejected before this layer does
    # any work. The decoder layer above and the model loop above that read the same
    # predicate to decide how many values to unpack; see its docstring.
    build_indexer_loss = _builds_indexer_kl(self)
    # --- Patch.3 ---
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
            # --- Patch.3 ---
            build_indexer_loss=build_indexer_loss,
            # --- Patch.3 ---
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
    # --- Patch.3 ---
    if build_indexer_loss:
        index_score = compressed_candidates.indexer_scores if compressed_candidates is not None else None
        if index_score is None:
            raise RuntimeError(
                "dsa_indexer_loss is enabled but the CSA compressor produced no indexer scores, so the "
                "KL would have no student distribution to train. Every path that can drop them raises "
                "before here, so this is a wiring regression rather than a configuration problem."
            )
        # The width of the compressed slice the teacher is asked for, read off the
        # *scores* so that the KL pairs slot ``j`` of the teacher with the score
        # ``index_score[..., j]``.
        #
        # The check below claims exactly one thing: that the two tensors the KL pairs
        # are the same width. It compares two widths, so it cannot see a reordering of
        # ``torch.cat((sliding_indices, compressed_indices))`` -- that leaves both
        # widths unchanged while ``[:, :, -width:]`` starts reading window slots. The
        # reordering guard is a test, not this line:
        # ``test_target_reads_the_full_window_lse_and_the_trailing_compressed_slice``
        # compares the teacher's slot tensor against the indexer's own selection
        # lifted past the full-resolution KV rows.
        #
        # ``raise`` rather than ``assert``, matching its siblings above and below:
        # ``python -O`` strips an ``assert``, and this is the only thing standing
        # between the teacher's ``[:, :, -width:]`` and the sliding-window slots. A
        # width mismatch under -O would not crash -- it would silently train the
        # indexer against the wrong distribution.
        kwargs["indexer_target_width"] = index_score.shape[-1]
        if kwargs["indexer_target_width"] != compressed_candidates.topk_indices.shape[-1]:
            raise RuntimeError(
                f"the indexer scored {kwargs['indexer_target_width']} slots while the compressor selected "
                f"{compressed_candidates.topk_indices.shape[-1]}: the KL pairs slot j of the teacher with "
                "index_score[..., j], so the two must be the same width"
            )
    # --- Patch.3 ---
    attention_outputs = attention_interface(
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
    # --- Patch.3 ---
    # The three-value return is only reachable through the patched
    # ``eager_attention_forward`` above: ``_indexer_loss_enabled`` requires
    # ``dsa_attention_implementation == "tilelang"``, and DeepSeek-V4 declares no
    # support for any registry interface (``_supports_flash_attn`` /
    # ``_supports_sdpa`` / ``_supports_flex_attn`` are all False), so
    # ``_attn_implementation`` is "eager" and ``get_interface`` falls back to the
    # module-level function this file replaces.
    if build_indexer_loss:
        attn_output, attn_weights, target = attention_outputs
        kl_terms, uniform_terms = indexer_kl_terms(index_score, target)
        indexer_kl = kl_terms.sum()
        # Summed over exactly the rows the KL is summed over, so the two travel the
        # whole way to the metric through the same denominators and the ratio taken at
        # the end is a ratio of means. A per-row ``kl / uniform`` averaged instead
        # would be dominated by the rows with the smallest reference -- wrong, and
        # wrong in a way that still lands in [0, 1] and looks entirely plausible.
        indexer_uniform = uniform_terms.sum()
    else:
        attn_output, attn_weights = attention_outputs
    # --- Patch.3 ---

    if ulysses_enabled:
        # eager/TileLang return [B, S_full, H_local, D]; restore local seq + full heads.
        attn_output = gather_heads_scatter_seq(
            attn_output, head_dim=2, seq_dim=1, group=get_parallel_state().ulysses_group
        )

    attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)
    grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
    grouped = self.o_a_proj(grouped).flatten(2)
    output = self.o_b_proj(grouped)
    # --- Patch.3 ---
    # 0-d sums rather than the [B, S] terms: the decoder layer above only has to
    # add these together, and summing here keeps the reduction over the query rows
    # this rank holds, so a future sequence-parallel mode reduces a plain sum of
    # per-rank contributions rather than having to re-derive the row weighting.
    if build_indexer_loss:
        return output, attn_weights, indexer_kl, indexer_uniform
    # --- Patch.3 ---
    return output, attn_weights


# ================================================================
# Patch: eager_attention_forward
# 1. Dispatch DeepSeek-V4 attention to the TileLang sparse MQA kernel when
#    ``dsa_attention_implementation=tilelang``. The existing additive mask is
#    converted to a compact fixed-width index list, preserving sliding-window,
#    compressor, causal, and invalid-index semantics.
# 2. Preserve the upstream eager implementation as the default fallback.
# 3. Return the indexer loss's teacher distribution as a third value when the
#    caller sets ``indexer_target_width``. The return annotation states that
#    arity, so a caller reads the contract off the signature rather than off a
#    comment; ``indexer_target_width`` is the only thing that selects it.
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
) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    # --- Patch.1 ---
    attention_implementation = veomni_dsa_attention_implementation.value
    if attention_implementation not in {"eager", "tilelang"}:
        raise ValueError(
            "DeepSeek-V4 does not support "
            f"dsa_attention_implementation={attention_implementation!r}; expected 'eager' or 'tilelang'"
        )
    # Operand dtypes are the kernel's contract and are enforced by
    # ``sparse_attn_tilelang`` itself, which reports the offending dtype. Only
    # structural conditions belong here.
    use_tilelang = (
        attention_implementation == "tilelang"
        and query.is_cuda
        and query.shape[-1] == 1 << (query.shape[-1] - 1).bit_length()
        and (isinstance(attention_mask, torch.Tensor) or kwargs.get("sparse_topk_indices") is not None)
        and dropout == 0
        and key.shape[1] == 1
    )
    # --- Patch.3 ---
    # The indexer loss's teacher is a TileLang kernel, so a declined dispatch cannot
    # produce one. Refusing ahead of the general refusal below turns that into a
    # legible error rather than the caller's unpack of a two-value return.
    if not use_tilelang and kwargs.get("indexer_target_width") is not None:
        raise RuntimeError(
            "dsa_indexer_loss needs the TileLang sparse attention dispatch to obtain the teacher's "
            "log-sum-exp, but the dispatch was declined at runtime. Check that query/key/value are "
            "bf16 CUDA tensors."
        )
    # --- Patch.3 ---
    # Mask-free callers rely on this refusal for correctness, not just for
    # diagnostics: they withheld the dense mask, so an eager fallback would have
    # nothing left to enforce causality with.
    if attention_implementation == "tilelang" and not use_tilelang:
        raise ValueError(
            "dsa_attention_implementation='tilelang' was requested but the TileLang sparse attention "
            f"does not support this call: is_cuda={query.is_cuda}, head_dim={query.shape[-1]}, "
            f"mask={type(attention_mask).__name__}, dropout={dropout}, kv_heads={key.shape[1]}"
        )
    if use_tilelang:
        topk_indices = kwargs.get("sparse_topk_indices")
        if topk_indices is None:
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
        elif attention_mask is not None:
            topk_indices = mask_sparse_attention_indices(attention_mask, topk_indices)
        sinks = kwargs.get("s_aux", module.sinks)
        # --- Patch.3 ---
        # ``indexer_target_width`` is how ``DeepseekV4Attention.forward`` asks for the
        # indexer loss's teacher distribution: the width of the compressed slice it
        # wants scored, and the signal that this call returns three values instead of
        # two. Only that forward sets it, and only when its own gate is on.
        target_width = kwargs.get("indexer_target_width")
        if target_width is not None:
            query_rows = query.transpose(1, 2).contiguous()
            kv_rows = key[:, 0].contiguous()
            # One forward, and the teacher reads *its* LSE. That LSE is the true CSA
            # denominator only because ``topk_indices`` spans the sliding window as
            # well as the compressed entries and the kernel folds the sink into the
            # same sumexp. A second forward over the compressed slice alone would
            # produce a plausible, decreasing loss that trains the indexer toward the
            # wrong distribution (NVIDIA/Megatron-LM#5776).
            attn_output, lse = sparse_attn_tilelang(
                query_rows,
                kv_rows,
                sinks.float().contiguous(),
                topk_indices,
                scaling,
                return_lse=True,
            )
            # The compressed entries are the *trailing* range of the index tensor:
            # both ``build_sparse_attention_indices`` and
            # ``build_packed_sparse_attention_indices`` end at
            # ``torch.cat((sliding_indices, compressed_indices), dim=-1)``, and the
            # caller asserts that this width is the selection's own.
            target = sparse_mqa_target_fwd(
                query_rows,
                kv_rows,
                topk_indices[:, :, -target_width:].contiguous(),
                lse,
                scaling,
            )
            # A row the teacher gave no mass at all goes out as exactly zero rather
            # than as ``0 / tiny``. The two differ: dividing by the clamp raises the
            # denominator instead of the numerator, so a row whose mass is denormal
            # rather than zero comes back summing to something in (0, 1) -- neither a
            # distribution nor an absence of one, and ``indexer_kl_terms`` weights it
            # as though it were the former. Zero is the case that says "nothing to
            # learn from this row", and the KL excludes it from both of its terms.
            #
            # Reachable two ways: every slot of the row was a miss, which is the
            # common one; or every selected compressed logit sat so far below the LSE
            # that ``exp`` underflowed, i.e. attention put essentially all of this
            # query's mass on its sliding window and sink.
            target_mass = target.sum(-1, keepdim=True)
            tiny = torch.finfo(torch.float32).tiny
            target = torch.where(target_mass > tiny, target / target_mass.clamp_min(tiny), 0.0)
            return attn_output, None, target
        # --- Patch.3 ---
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
) -> MoeModelOutputWithIndexerKL:
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
            veomni_dsa_attention_implementation.value == "tilelang"
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

    # --- Patch.3 ---
    indexer_kl_total = None
    indexer_uniform_total = None
    indexer_kl_layers = 0
    # --- Patch.3 ---
    for layer in self.layers:
        # --- Patch.3 ---
        # The same predicate the decoder layer and the attention forward read, so
        # the arity of ``layer_output`` is decided in one place rather than three.
        # Branching on ``isinstance(layer_output, tuple)`` instead would *absorb* a
        # regression rather than surface it: a decoder layer that returned
        # ``(hidden_states, None)`` on the flag-off path would read here as "this
        # layer built a KL", every test would still pass, and nothing would enforce
        # the bare-tensor contract the layer's own comment spells out.
        builds_indexer_kl = _builds_indexer_kl(layer.self_attn)
        # --- Patch.3 ---
        layer_output = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=causal_mask,
            input_ids=input_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        # --- Patch.3 ---
        # ``isinstance`` *verifies* the predicate here; it does not stand in for it.
        # The unpacking below is not self-checking: ``a, b = tensor`` succeeds for
        # any tensor whose leading dimension is 2, so on a two-sample batch a bare
        # tensor from a regressed layer would be taken apart into hidden states and
        # a "KL" without a word — the batch size deciding whether the bug is loud.
        if builds_indexer_kl is not isinstance(layer_output, tuple):
            raise RuntimeError(
                f"decoder layer {layer.layer_idx} returned "
                f"{'a tuple' if isinstance(layer_output, tuple) else type(layer_output).__name__} while "
                f"_builds_indexer_kl says builds_indexer_kl={builds_indexer_kl}: the layer and the model "
                "loop disagree about the indexer-KL return arity"
            )
        # Only the CSA layers return a tuple; the rest return the bare tensor they
        # always returned. The KL is summed rather than averaged over the layers,
        # which is deliberate and matches the MoE router aux loss this sits beside;
        # ``indexer_kl_layers`` carries the count so the *metric* can be a per-layer
        # mean while the objective keeps the sum. The uniform reference is summed
        # over the same layers by the same rule, which is what makes the ratio of the
        # two independent of that divisor.
        if builds_indexer_kl:
            hidden_states, layer_kl, layer_uniform = layer_output
            indexer_kl_total = layer_kl if indexer_kl_total is None else indexer_kl_total + layer_kl
            indexer_uniform_total = (
                layer_uniform if indexer_uniform_total is None else indexer_uniform_total + layer_uniform
            )
            indexer_kl_layers += 1
        else:
            hidden_states = layer_output
        # --- Patch.3 ---

    # --- Patch.3 ---
    # A model configured for the loss whose ``layer_types`` has no CSA entry would
    # otherwise accept the flag and train nothing: with no layer carrying a
    # Lightning Indexer there is no student, ``indexer_kl_total`` stays ``None``,
    # and both the metric and the fold-in in ``ForCausalLM.forward`` are skipped in
    # silence -- a plausible loss curve training nothing, which is the failure class
    # this feature exists to prevent. It is the same class the refusals in
    # ``_indexer_loss_enabled`` cover, and the analogous case one layer down -- a CSA
    # compressor that produced no scores -- already raises.
    #
    # Here rather than beside the fold-in in ``ForCausalLM.forward``, so that it
    # fires identically with and without ``labels``: the fold-in is the half that is
    # conditional on labels, the refusal must not be. And on the first forward rather
    # than at construction, because the only construction-time hook is a full-body
    # ``override_method`` on ``DeepseekV4Model.__init__``: patchgen replaces methods
    # whole, so that would fork the constructor from upstream and silently drop any
    # field a future transformers adds to it. That is a worse instance of this very
    # failure class than the one it would close, and every other refusal in this
    # feature fires on the first forward too.
    if indexer_kl_layers == 0 and _indexer_loss_enabled(self):
        raise RuntimeError(
            "dsa_indexer_loss is enabled but no layer of this model builds an indexer KL: "
            f"layer_types={list(self.config.layer_types)} contains no 'compressed_sparse_attention' "
            "entry, and only a CSA layer carries a Lightning Indexer to train. The flag would "
            "otherwise be accepted and train nothing."
        )
    # --- Patch.3 ---

    hidden_states = self.norm(self.hc_head(hidden_states))
    # --- Patch.3 ---
    # ``MoeModelOutputWithIndexerKL`` declares the four fields below; assigning them
    # onto a ``MoeModelOutputWithPast`` instead would make them invisible to
    # ``keys()`` and to pytree flattening, and any consumer that reconstructs the
    # output would drop them without a word. All are ``None`` with the loss off,
    # and ``keys()`` skips ``None``, so the flag-off output is unchanged -- which is
    # why the layer count goes out as ``None`` rather than as the 0 it holds there.
    #
    # The token count is every local query row, padding included (~0.1% of a packed
    # row on the reference run). Excluding them would need packed-metadata plumbing
    # for a correction far below the scale this auxiliary objective is tuned at.
    return MoeModelOutputWithIndexerKL(
        last_hidden_state=hidden_states,
        past_key_values=return_cache,
        indexer_kl_total=indexer_kl_total,
        indexer_uniform_total=indexer_uniform_total,
        indexer_query_tokens=hidden_states.shape[0] * hidden_states.shape[1] if indexer_kl_total is not None else None,
        indexer_kl_layers=indexer_kl_layers if indexer_kl_total is not None else None,
    )
    # --- Patch.3 ---


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

    outputs: MoeModelOutputWithIndexerKL = self.model(
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

    # --- Patch.3 ---
    aux_metrics = None
    if outputs.indexer_kl_total is not None:
        local_query_tokens = torch.tensor(
            outputs.indexer_query_tokens, device=outputs.indexer_kl_total.device, dtype=torch.float32
        )
        # The model body summed over this rank's query rows; the mean is taken here
        # because ``reduce_sequence_parallel_loss`` wants a local *mean* and the
        # local count, and re-weights by that count before dividing by the global
        # one. Handing it the sum would train perfectly well on a single rank and
        # converge to the wrong cross-rank weighting -- a discrepancy invisible to
        # any single-process test of the value.
        local_mean = outputs.indexer_kl_total / local_query_tokens.clamp_min(1)
        # The zero-information reference the KL has to be read against, carried
        # through byte-for-byte the same denominators: the same per-rank token count,
        # the same SP reduction, the same layer sum. The ratio taken below is then a
        # ratio of means over identical supports, which is the only form of it that
        # is right -- averaging a per-row or per-rank ``kl / uniform`` instead gives a
        # number that still lands in [0, 1] and is quietly wrong.
        #
        # ``.clone()`` on the token count for each call, not a shared tensor:
        # ``ReduceLoss.forward`` all-reduces ``num_valid_tokens`` *in place*, so a
        # second call handed the same tensor would divide by an SP-world-size-times
        # inflated count -- correct on one rank, wrong on two, and so invisible to
        # every test this feature has.
        local_uniform_mean = outputs.indexer_uniform_total / local_query_tokens.clamp_min(1)
        # Unreachable today, and deliberately still here: ``_indexer_loss_enabled``
        # refuses both sequence-parallel modes, so ``sp_enabled`` is False by the time
        # control reaches this line. Whichever change lifts one of those refusals
        # inherits a reduction whose weighting is already right, rather than growing
        # one next to a fold-in that reads correct on a single rank either way.
        if get_parallel_state().sp_enabled:
            indexer_kl = reduce_sequence_parallel_loss(local_mean, local_query_tokens.clone())
            indexer_uniform = reduce_sequence_parallel_loss(local_uniform_mean, local_query_tokens.clone())
        else:
            indexer_kl = local_mean
            indexer_uniform = local_uniform_mean
        # The *loss* keeps the layer sum -- a settled decision, and the reason the
        # fold-in below reads ``indexer_kl`` rather than the mean. The *metric* is a
        # per-layer mean, matching Megatron's
        # ``avg_indexer_loss = values.sum() / max(num_indexer_layers, 1)``
        # (``dsa.py:427``): summed, ``training/indexer_kl`` is ~21x larger on
        # DeepSeek-V4-Flash (21 CSA layers) than on the 1-CSA-layer smoke checkpoint
        # at identical per-layer quality, so no two runs with different layer counts
        # -- and no comparison against an upstream number -- mean anything.
        #
        # ``max(..., 1)`` guards nothing reachable: this block is entered only when at
        # least one layer contributed. It is there because the divisor is the sort of
        # thing that becomes reachable later, and a division by zero here would be a
        # NaN in a metric rather than an error.
        indexer_kl_layers = max(outputs.indexer_kl_layers or 0, 1)
        # ``indexer_kl`` alone says nothing: the reference run's plateau of 0.021
        # means one thing against a zero-information reference of 0.374 and another
        # against 0.02. ``indexer_kl_captured`` is the reading -- 1.0 is a student
        # that reproduces the teacher, 0.0 is one that knows only the candidate set --
        # and both terms go out beside it, because a reader given only the fraction
        # can reconstruct neither.
        #
        # The layer divisor cancels in the ratio, both terms being summed over the
        # same layers; it is applied to each anyway so the two reported numbers are
        # per-layer means on the same scale as each other and as ``indexer_kl``.
        #
        # ``clamp_min`` on the denominator: the reference is zero exactly when every
        # query row has at most one candidate, in which case the KL is zero too and
        # nothing was there to capture. 1.0 -- "captured everything" -- is the honest
        # reading of that, and a NaN in a metric would propagate into the logger.
        #
        # The ratio is formed here, over this micro-batch, and what the logger shows
        # is therefore a mean of per-micro-batch ratios once ``mean_aux_metrics``
        # divides by the micro-step count and ``EnvironMeterCallback`` averages over
        # the data-parallel group -- *not* one minus the ratio of the two numbers
        # logged beside it. That is the aggregation the comments above rule out for
        # rows and for sequence-parallel ranks, and the reason it is fine here is a
        # property of what is being averaged rather than a change of principle: the
        # rows those comments are about differ by orders of magnitude in their
        # reference, so the small ones dominate a mean of ratios, while these terms
        # are each already a mean over a whole micro-batch of rows and land within a
        # few percent of one another. An exact global ratio would need the aux-metric
        # path to carry a numerator and a denominator instead of a value, which is a
        # trainer-wide change for a third decimal place.
        indexer_kl_metric = indexer_kl.detach() / indexer_kl_layers
        indexer_uniform_metric = indexer_uniform.detach() / indexer_kl_layers
        aux_metrics = {
            "indexer_kl": indexer_kl_metric,
            "indexer_kl_uniform": indexer_uniform_metric,
            "indexer_kl_captured": 1.0
            - indexer_kl_metric / indexer_uniform_metric.clamp_min(torch.finfo(torch.float32).tiny),
        }
        # No labels means no loss to fold into -- ``loss`` is ``None`` and the
        # addition would raise. The metric is still reported: an inference forward
        # that computed the KL may as well say what it was.
        #
        # The coefficient is positive by the time control reaches here:
        # ``_indexer_loss_enabled`` gates on it, so a non-positive one leaves
        # ``indexer_kl_total`` ``None`` and this whole block unentered. It is not
        # re-checked, because two places deciding "is the objective on" is exactly the
        # staleness this feature's single-predicate discipline exists to prevent.
        if labels is not None:
            # The language-model objective as it stood before the KL joined it, so a
            # flag-on run still has a curve comparable to a flag-off baseline. The
            # fold-in below stays exactly as it was -- it is what makes the indexer's
            # gradient scale right by construction, riding
            # ``reduce_sequence_parallel_loss`` and ``mean_global_loss`` on the same
            # chain as the LM loss -- so this is an extra *metric*, not Megatron's
            # ``DSAIndexerLossAutoScaler``, which leaves the forward value untouched at
            # the price of reproducing that chain by hand.
            #
            # Subtracting the reported KL from ``training/foundation_loss`` is not the
            # same number: ``mean_global_loss`` weights the total by the micro-batch's
            # label-token share while an aux metric gets a plain ``1/N``, so the
            # subtraction is exact only when the micro-batches carry equal label
            # counts. This entry rides the aux-metric path, so it needs no such
            # assumption.
            aux_metrics["lm_loss_before_indexer_kl"] = loss.detach()
            # Read off ``self.config``, four lines below where ``self.router_aux_loss_coef``
            # weights this model's other auxiliary objective, and from the same place.
            loss = loss + self.config.dsa_indexer_loss_coef * indexer_kl.to(loss.device)
    # --- Patch.3 ---

    return MoeCausalLMOutputWithLogProbs(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
        fused_linear_aux=fused_linear_aux,
        # --- Patch.3 ---
        aux_metrics=aux_metrics,
        # --- Patch.3 ---
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
