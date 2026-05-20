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

"""Chunked fused linear top-k forward-KL **distillation** loss.

Companion to ``chunk_logprobs.py``. Extends the same chunked-fused-linear
pattern (custom ``torch.autograd.Function`` that streams the lm_head
projection chunk-by-chunk, never materializing ``[T, V]`` logits) with
the three tensors verl's top-k distillation path consumes:

    distillation_losses[b, l] = Σ_k exp(log_p_t,b,l,k)
                                · (log_p_t,b,l,k - log_q_s,b,l,k)
    student_mass[b, l]        = Σ_k exp(log_q_s,b,l,k)
    teacher_mass[b, l]        = Σ_k exp(log_p_t,b,l,k)

where ``log_q_s = log_softmax(student_logits).gather(teacher_topk_ids)``
and ``log_p_t = teacher_topk_log_probs`` (passed in dense, ``[B, L, K]``).

The output schema matches verl's
``verl/trainer/distillation/fsdp/losses.py::compute_forward_kl_topk`` —
verl's ``compute_topk_loss`` already routes the ``"veomni"`` strategy
through the FSDP loss path, so an external verl branch that wires the
VeOmni engine to read these fields off ``model_output`` lands on the
fused-linear, no-``[T, V]``-logits version of the same numerics.

Why a new file rather than extending ``chunk_logprobs.py``:

- ``chunk_logprobs.py`` is on the hot path for DPO / PPO and is
  bitwise-validated against verl's ``FusedLinearForPPOFunction``.
  Adding the distillation branch in-line would expand its backward to
  three additive grad paths even on the no-teacher case (or grow a
  per-chunk Python branch), so keeping the new kernel sibling avoids
  any regression risk on the existing path.
- The forward op order on the **shared** outputs (``log_probs``,
  ``entropy``) is copied verbatim from ``chunk_logprobs.py`` (same
  helpers, same fp32 cast site, same fa_ce per-token NLL), so the new
  kernel inherits the same fp32 rounding boundary as verl's
  ``FusedLinearForPPOFunction`` for those two slots.

Sign convention recap (matches verl):

- ``distillation_losses``: non-negative (forward KL ``KL(p_t || q_s)``,
  computed on the top-k support; bitwise-equal to
  ``compute_forward_kl_topk`` under the same inputs).
- ``student_mass`` / ``teacher_mass``: non-negative; returned with
  ``requires_grad=False`` — they are reported as metrics by verl and
  not used for backprop.
- ``log_probs``: non-positive — actual log-probabilities (sign already
  flipped relative to NLL).
- ``entropy``: non-negative — softmax entropy ``H[p] = -Σ p log p``.

All five outputs are masked to ``0.0`` at positions where
``labels == ignore_index``; gradient is zero at those positions on
every path.
"""

from typing import Optional

import torch

from ....distributed.parallel_state import get_parallel_state
from .chunk_logprobs import (
    _per_token_entropy_from_logits,
    _per_token_log_probs_from_logits,
)


class _ChunkedLinearTopkDistill(torch.autograd.Function):
    """Chunked linear projection + top-k forward-KL + log-probs + entropy.

    Same skeleton as ``_ChunkedLinearLogProbs``. Forward saves
    ``(hidden_states, weight, labels, teacher_topk_ids,
    teacher_topk_log_probs)`` and the scalars needed to drive the
    backward; backward chunks again and recomputes logits to derive
    ``dhidden = dlogits @ weight``, ``dweight = dlogits.t() @ hidden_states``.

    Gradients reach the lm_head via three additive ``dlogits``
    contributions:

    1. ``log_probs`` grad — identical to ``_ChunkedLinearLogProbs``:
       ``dlp · (one_hot[labels] - probs)``.
    2. ``entropy`` grad — identical to ``_ChunkedLinearLogProbs``:
       ``-dent · p · (log p + H)``.
    3. ``distillation_losses`` grad (new). With
       ``L_l = Σ_k exp(log_p_t,k) · (log_p_t,k - log_q_s,k)`` and
       ``log_q_s,k = log_softmax(logits)[ids_k]``,
       ``∂L_l/∂logits[v] = ddist · (-p_t[v] + teacher_mass · softmax[v])``,
       where ``p_t[v] = scatter_add(exp(log_p_t), ids)[v]`` is the dense
       teacher top-k probability (zero outside the top-k support).

    ``teacher_mass`` and ``student_mass`` are detached on the output
    side so callers can ``backward`` through ``distillation_losses``
    (and optionally the other two grad paths) without surprise.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        teacher_topk_ids: torch.Tensor,
        teacher_topk_log_probs: torch.Tensor,
        temperature: float,
        chunk_size: int,
        ignore_index: int,
        log_prob_min_clamp: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)

        orig_shape = labels.shape
        orig_hidden_shape = hidden_states.shape
        K = teacher_topk_ids.shape[-1]

        h_2d = hidden_states.reshape(-1, hidden_states.size(-1))
        l_1d = labels.reshape(-1)
        ids_2d = teacher_topk_ids.reshape(-1, K)
        tlp_2d = teacher_topk_log_probs.reshape(-1, K)
        T = l_1d.shape[0]

        out_requires_grad = h_2d.requires_grad or weight.requires_grad
        log_probs = torch.zeros(T, device=h_2d.device, dtype=torch.float32, requires_grad=out_requires_grad)
        entropy = torch.zeros(T, device=h_2d.device, dtype=torch.float32, requires_grad=out_requires_grad)
        distill = torch.zeros(T, device=h_2d.device, dtype=torch.float32, requires_grad=out_requires_grad)
        # ``student_mass`` / ``teacher_mass`` are returned as metrics; they
        # must not be part of the backward graph. Allocate with
        # ``requires_grad=False`` and emit them detached at the end.
        student_mass = torch.zeros(T, device=h_2d.device, dtype=torch.float32)
        teacher_mass = torch.zeros(T, device=h_2d.device, dtype=torch.float32)

        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            h_chunk = h_2d[chunk_start:chunk_end]
            l_chunk = l_1d[chunk_start:chunk_end]
            ids_chunk = ids_2d[chunk_start:chunk_end]
            tlp_chunk = tlp_2d[chunk_start:chunk_end]

            # Op order mirrors ``chunk_logprobs.py`` exactly so the shared
            # (log_probs, entropy) outputs hit the same fp32 rounding
            # boundary as verl's ``FusedLinearForPPOFunction``.
            logits = h_chunk @ weight.t()
            if temperature != 1.0:
                logits = logits / temperature
            logits = logits.float()

            log_probs[chunk_start:chunk_end] = _per_token_log_probs_from_logits(logits, l_chunk, ignore_index)
            chunk_entropy = _per_token_entropy_from_logits(logits)
            mask = l_chunk != ignore_index
            entropy[chunk_start:chunk_end] = torch.where(mask, chunk_entropy, torch.zeros_like(chunk_entropy))

            # ── top-k distillation outputs ───────────────────────────────
            # Use ``log_softmax`` on logits (already fp32) and gather at the
            # teacher's top-k indices. ``student_mass`` / ``teacher_mass`` are
            # *before* any clamp so they faithfully report the mass on the
            # teacher's top-k support.
            student_log_probs = logits.log_softmax(dim=-1)
            student_topk_lp = student_log_probs.gather(dim=-1, index=ids_chunk)  # [chunk, K]

            student_mass_chunk = student_topk_lp.exp().sum(dim=-1)  # [chunk]
            teacher_mass_chunk = tlp_chunk.exp().sum(dim=-1)  # [chunk]

            if log_prob_min_clamp is not None:
                student_topk_lp = student_topk_lp.clamp_min(log_prob_min_clamp)
                tlp_clamped = tlp_chunk.clamp_min(log_prob_min_clamp)
            else:
                tlp_clamped = tlp_chunk

            # Forward KL on top-k: Σ_k exp(log_p_t,k) · (log_p_t,k - log_q_s,k).
            # Cast teacher tensors to fp32 once for the reduction (the input
            # may arrive as bf16 from verl's nested-tensor path).
            tlp_f32 = tlp_clamped.float()
            student_topk_lp_f32 = student_topk_lp.float()
            kl_chunk = (tlp_f32.exp() * (tlp_f32 - student_topk_lp_f32)).sum(dim=-1)

            # Mask all three top-k outputs at IGN positions.
            zero = torch.zeros_like(kl_chunk)
            distill[chunk_start:chunk_end] = torch.where(mask, kl_chunk, zero)
            student_mass[chunk_start:chunk_end] = torch.where(mask, student_mass_chunk, zero)
            teacher_mass[chunk_start:chunk_end] = torch.where(mask, teacher_mass_chunk, zero)

        ctx.save_for_backward(h_2d, weight, l_1d, ids_2d, tlp_2d)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.log_prob_min_clamp = log_prob_min_clamp
        ctx.orig_hidden_shape = orig_hidden_shape

        # student_mass / teacher_mass are emitted detached; they do not
        # propagate gradients (no consumer in verl needs them to). The
        # backward unpacks five grad_output slots but only walks the first
        # three (log_probs, entropy, distill).
        return (
            log_probs.view(orig_shape),
            entropy.view(orig_shape),
            distill.view(orig_shape),
            student_mass.view(orig_shape).detach(),
            teacher_mass.view(orig_shape).detach(),
        )

    @staticmethod
    def backward(
        ctx,
        dlog_probs: Optional[torch.Tensor],
        dentropy: Optional[torch.Tensor],
        ddistill: Optional[torch.Tensor],
        dstudent_mass: Optional[torch.Tensor],  # noqa: ARG004 - student_mass is detached
        dteacher_mass: Optional[torch.Tensor],  # noqa: ARG004 - teacher_mass is detached
    ):
        if dlog_probs is None and dentropy is None and ddistill is None:
            return None, None, None, None, None, None, None, None, None

        h_2d, weight, l_1d, ids_2d, tlp_2d = ctx.saved_tensors
        T = l_1d.shape[0]
        dlog_probs_1d = dlog_probs.reshape(-1).float() if dlog_probs is not None else None
        dentropy_1d = dentropy.reshape(-1).float() if dentropy is not None else None
        ddistill_1d = ddistill.reshape(-1).float() if ddistill is not None else None

        dhidden = torch.zeros_like(h_2d) if h_2d.requires_grad else None
        dweight = torch.zeros_like(weight) if weight.requires_grad else None

        for chunk_start in range(0, T, ctx.chunk_size):
            chunk_end = min(chunk_start + ctx.chunk_size, T)
            h_chunk = h_2d[chunk_start:chunk_end]
            l_chunk = l_1d[chunk_start:chunk_end]
            ids_chunk = ids_2d[chunk_start:chunk_end]
            tlp_chunk = tlp_2d[chunk_start:chunk_end]
            dlp_chunk = dlog_probs_1d[chunk_start:chunk_end] if dlog_probs_1d is not None else None
            dent_chunk = dentropy_1d[chunk_start:chunk_end] if dentropy_1d is not None else None
            ddist_chunk = ddistill_1d[chunk_start:chunk_end] if ddistill_1d is not None else None

            logits = h_chunk @ weight.t()
            if ctx.temperature != 1.0:
                logits = logits / ctx.temperature
            logits = logits.float()

            probs = logits.softmax(dim=-1)
            mask = (l_chunk != ctx.ignore_index).float()

            dlogits = torch.zeros_like(probs)

            # ── log_probs gradient path ──────────────────────────────────
            if dlp_chunk is not None:
                safe_labels = l_chunk.clamp(min=0).unsqueeze(-1)
                one_hot = torch.zeros_like(probs).scatter_(-1, safe_labels, 1.0)
                masked_dlp = (dlp_chunk * mask).unsqueeze(-1)
                dlogits = dlogits + masked_dlp * (one_hot - probs)

            # ── entropy gradient path ────────────────────────────────────
            if dent_chunk is not None:
                log_probs_full = logits.log_softmax(dim=-1)
                entropy_full = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
                masked_dent = (dent_chunk * mask).unsqueeze(-1)
                dlogits = dlogits + probs * (log_probs_full + entropy_full.unsqueeze(-1)) * (-masked_dent)

            # ── top-k distillation gradient path ─────────────────────────
            # L_l = Σ_k exp(clamp(log_p_t,k)) · (clamp(log_p_t,k) - clamp(log_q_s,k)).
            # Only log_q_s,k depends on logits (through log_softmax); the
            # clamp on log_p_t,k is a no-op wrt logits. So
            # ∂L_l/∂logits[v] = -Σ_k exp(clamp(log_p_t,k)) · ∂clamp(log_q_s,k)/∂logits[v]
            # where ∂clamp(log_q_s,k)/∂logits[v] is 0 when the clamp is
            # active at k (log_q_s,k <= floor) and otherwise equals
            # δ(v == ids_k) - softmax[v]. This collapses to:
            # ∂L_l/∂logits[v] = -p_t_sparse[v] + teacher_mass_eff · softmax[v]
            # where p_t_sparse[v] = Σ_k exp(clamp(log_p_t,k)) · δ(v == ids_k)
            # gated by ``active`` so positions with active student clamp
            # contribute zero. The teacher coefficient must use the
            # **clamped** tlp to stay consistent with the forward.
            if ddist_chunk is not None:
                if ctx.log_prob_min_clamp is not None:
                    # Recompute the same student top-k log-probs as forward
                    # so we know which entries the clamp masked off.
                    student_log_probs = logits.log_softmax(dim=-1)
                    student_topk_lp = student_log_probs.gather(dim=-1, index=ids_chunk)
                    active = (student_topk_lp >= ctx.log_prob_min_clamp).float()
                    # Teacher coefficient: ``exp(clamp(log_p_t,k))`` — must
                    # match the forward's ``tlp_clamped`` so the gradient
                    # is consistent. Floor-clamping log-probs floors the
                    # exponentiated mass at ``exp(log_prob_min_clamp)``.
                    tlp_f32 = tlp_chunk.clamp_min(ctx.log_prob_min_clamp).float()
                else:
                    active = None
                    tlp_f32 = tlp_chunk.float()

                # Dense ``p_t[v]`` via scatter_add (handles duplicate ids
                # by summing — the chance verl emits duplicates is zero
                # in practice but the math is well-defined either way).
                p_t_k = tlp_f32.exp()  # [chunk, K]
                if active is not None:
                    p_t_k_eff = p_t_k * active
                else:
                    p_t_k_eff = p_t_k
                p_t_dense = torch.zeros_like(probs).scatter_add_(-1, ids_chunk, p_t_k_eff)
                teacher_mass_eff = p_t_k_eff.sum(dim=-1, keepdim=True)  # [chunk, 1]
                masked_ddist = (ddist_chunk * mask).unsqueeze(-1)
                dlogits = dlogits + masked_ddist * (-p_t_dense + teacher_mass_eff * probs)

            # Op order mirrors ``chunk_logprobs.py``: cast to input dtype
            # first, then divide by temperature, then matmul. Critical for
            # bitwise compatibility on the bf16 path.
            dlogits = dlogits.to(h_chunk.dtype)
            if ctx.temperature != 1.0:
                dlogits = dlogits / ctx.temperature

            if dhidden is not None:
                dhidden[chunk_start:chunk_end] = dlogits @ weight
            if dweight is not None:
                dweight += dlogits.t() @ h_chunk

        if dhidden is not None:
            dhidden = dhidden.view(ctx.orig_hidden_shape)

        # Signature is (hidden_states, weight, labels, teacher_topk_ids,
        # teacher_topk_log_probs, temperature, chunk_size, ignore_index,
        # log_prob_min_clamp) — 9 inputs, only the first two carry grads.
        return dhidden, dweight, None, None, None, None, None, None, None


def chunk_topk_distill_function(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    log_prob_min_clamp: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked fused-linear top-k forward-KL distillation + log-probs + entropy.

    Args:
        hidden_states: ``[B, L, H]`` (or ``[L, H]`` for packed inputs).
        weights: lm_head weight ``[V, H]``. Bias is not supported.
        labels: integer label tensor with shape matching the leading dims of
            ``hidden_states``. Positions equal to ``ignore_index`` produce
            ``0.0`` on every output (no gradient flows through them).
        teacher_topk_ids: ``[B, L, K]`` int64 — teacher's top-k vocabulary
            indices per position. Caller is expected to densify any nested
            tensors upstream (verl's FSDP path does
            ``.values().unsqueeze(0)`` before this function is reached).
        teacher_topk_log_probs: ``[B, L, K]`` — teacher's log-probabilities
            at those indices. May be bf16 from a nested storage; cast to
            fp32 internally for the KL reduction.
        chunk_size: token-dim chunk for the streamed projection.
        ignore_index: label value to mask (default ``-100``).
        shift_labels: pre-shifted labels (SP path). Same semantics as
            ``chunk_logprobs_function``. When the caller (verl) is already
            in remove-padding + use_fused_kernels mode, it passes the
            already-rolled labels as ``shift_labels`` and the kernel skips
            its internal causal shift; teacher tensors must align with
            those shifted labels.
        temperature: divides logits before log_softmax. Defaults to 1.0.
        log_prob_min_clamp: if not None, clamp both teacher and student
            top-k log-probabilities to this minimum before the KL
            reduction. Matches verl's
            ``DistillationLossConfig.log_prob_min_clamp`` semantics.

    Returns:
        ``(log_probs, entropy, distillation_losses, student_mass, teacher_mass)``
        — all five tensors share ``labels``' shape, all are fp32. The
        last two are ``detach()``-ed (metrics only); the first three
        flow gradients into ``hidden_states`` and ``weights`` through
        the closed-form backward.
    """
    sp_enabled = get_parallel_state().sp_enabled

    used_explicit_shift = shift_labels is not None
    if used_explicit_shift:
        labels_shifted = shift_labels
    elif sp_enabled:
        labels_shifted = labels
    else:
        labels_shifted = labels[..., 1:].contiguous()
        hidden_states = hidden_states[..., :-1, :].contiguous()
        # Teacher tensors are aligned to the label time-step, so apply the
        # same causal shift on them. (The reason we slice ``[..., 1:, :]`` on
        # hidden_states is that hidden_state[t-1] predicts label[t]; teacher
        # logprobs at position t come pre-aligned with label[t], so we drop
        # the leading-most slot to match the post-shift label length.)
        teacher_topk_ids = teacher_topk_ids[..., 1:, :].contiguous()
        teacher_topk_log_probs = teacher_topk_log_probs[..., 1:, :].contiguous()

    log_probs, entropy, distill, student_mass, teacher_mass = _ChunkedLinearTopkDistill.apply(
        hidden_states,
        weights,
        labels_shifted,
        teacher_topk_ids,
        teacher_topk_log_probs,
        float(temperature),
        int(chunk_size),
        int(ignore_index),
        log_prob_min_clamp,
    )

    if not sp_enabled and not used_explicit_shift:
        log_probs = torch.nn.functional.pad(log_probs, (0, 1), value=0.0)
        entropy = torch.nn.functional.pad(entropy, (0, 1), value=0.0)
        distill = torch.nn.functional.pad(distill, (0, 1), value=0.0)
        # ``F.pad`` produces a fresh tensor that re-enters the autograd graph;
        # re-detach the mass outputs so callers can rely on
        # ``requires_grad=False`` (metrics-only contract).
        student_mass = torch.nn.functional.pad(student_mass, (0, 1), value=0.0).detach()
        teacher_mass = torch.nn.functional.pad(teacher_mass, (0, 1), value=0.0).detach()
    return log_probs, entropy, distill, student_mass, teacher_mass
