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

"""Lightning Indexer KL helpers shared by DeepSeek-V4 GPU and NPU generated modeling."""

from __future__ import annotations

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models_kernel.utils.kernel_utils import resolve_kernel_impl


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

    ``DeepseekV4Config.validate_build_prerequisites`` refuses the two implementation
    fields at model-build time, before any rank reads a weight, so a launched run is
    told there rather than here. These stay because this predicate also covers the
    paths that never pass through ``build_foundation_model`` -- a model constructed
    straight from ``_from_config`` -- and because the parallel-state refusals below
    have no earlier home: the state is installed by then, but a model-agnostic gate
    cannot know that this model has no context-parallel indexer path.
    """
    if not getattr(module.config, "dsa_indexer_loss", False):
        return False
    if getattr(module.config, "dsa_indexer_loss_coef", 1.0) <= 0:
        return False
    if resolve_kernel_impl("dsa_indexer_implementation") != "tilelang":
        raise ValueError(
            "dsa_indexer_loss requires dsa_indexer_implementation='tilelang'; the eager "
            "indexer discards its scores, so the loss would have nothing to train against"
        )
    if resolve_kernel_impl("dsa_attention_implementation") != "tilelang":
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
            "has no context-parallel indexer path, so each rank would treat its sequence shard as a "
            "whole sequence and the teacher would be built from the resulting attention."
        )
    return True


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


__all__ = [
    "_builds_indexer_kl",
    "_indexer_loss_enabled",
    "_split_indexer_output",
    "indexer_kl_terms",
]
