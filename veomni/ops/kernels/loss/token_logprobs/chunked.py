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

"""Chunked fused linear log-probs and top-k distillation tensor kernels.

Contains the tensor-level ``_ChunkedLinearLogProbs`` and
``_ChunkedLinearTopkDistill`` custom ``torch.autograd.Function`` kernels.
Per-token log-probability computation dispatches through the ``token_logprobs``
operation family on ``OP_REGISTRY`` so device-specific kernels (GPU
``flash_attn`` Triton cross-entropy, Ascend NPU ``torch_npu`` cross-entropy,
and the PyTorch ``eager`` fallback) remain isolated in their respective row
modules. Model-facing causal label shifting, sequence-parallel policy, and
trailing padding stay in ``veomni.models.loss_utils``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
from torch import Tensor

from ....registry import VeomniOp
from .standard.eager import token_entropy_from_logits


def _verify_token_logprobs_backend(impl: str) -> None:
    """Import-check the underlying vendor kernel for ``impl``.

    ``OP_REGISTRY`` checks module discoverability via ``find_spec`` without
    importing vendor packages. This helper verifies that the compiled extension
    can actually be imported before committing to a non-eager row.
    """
    if impl == "flash_attn":
        from flash_attn.ops.triton.cross_entropy import cross_entropy_loss  # noqa: F401
    elif impl == "torch_npu":
        import torch_npu

        if not hasattr(torch_npu, "npu_cross_entropy_loss"):
            raise ImportError("torch_npu.npu_cross_entropy_loss is not available")


def resolve_token_logprobs_op(
    impl: str | None = None,
    *,
    device_type: str | None = None,
) -> VeomniOp:
    """Resolve the active ``token_logprobs`` ``VeomniOp`` handle.

    When ``impl`` is ``None``:
    - Selects ``"eager"`` directly when ``device_type == "cpu"`` so CPU tensors
      on a GPU or NPU host do not dispatch into an accelerator-only kernel.
    - Resolves the host default from ``OP_REGISTRY`` otherwise, falling back to
      ``"eager"`` if the discoverable vendor module fails to import at runtime
      (for example due to a binary extension ABI mismatch).

    When ``impl`` is explicitly specified, resolution is strict and never falls
    back on unmet requirements or runtime ``ImportError``.
    """
    if impl is None and device_type == "cpu":
        return VeomniOp("token_logprobs", "standard", "eager")
    op = VeomniOp("token_logprobs", "standard", impl)
    if impl is None:
        try:
            _verify_token_logprobs_backend(op.impl)
        except ImportError:
            return VeomniOp("token_logprobs", "standard", "eager")
    else:
        _verify_token_logprobs_backend(op.impl)
    return op


def resolve_token_logprobs_fn(
    impl: str | None = None,
    *,
    device_type: str | None = None,
) -> Callable[..., Tensor]:
    """Resolve the active ``token_logprobs`` implementation callable via ``VeomniOp``."""
    return resolve_token_logprobs_op(impl, device_type=device_type)


class _ChunkedLinearLogProbs(torch.autograd.Function):
    """Custom autograd Function: chunked linear projection + log-softmax + gather + entropy.

    Mirrors verl's ``FusedLinearForPPOFunction`` (custom autograd
    Function with explicit chunked forward + backward) so the FSDP2
    correctness story is the same. Returns ``(log_probs, entropy)`` as
    a 2-tuple; the entropy gradient path in backward is only walked
    when the caller actually backprops through the entropy output.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        weight: Tensor,
        labels: Tensor,
        temperature: float,
        chunk_size: int,
        ignore_index: int,
        token_logprobs_fn: Optional[Callable[..., Tensor]] = None,
    ) -> tuple[Tensor, Tensor]:
        """Stream the lm_head projection chunk-by-chunk to compute log-probs and entropy."""
        ctx.set_materialize_grads(False)
        if token_logprobs_fn is None:
            token_logprobs_fn = resolve_token_logprobs_fn(device_type=hidden_states.device.type)

        orig_shape = labels.shape
        orig_hidden_shape = hidden_states.shape
        # ``autograd.Function.forward`` runs with grad mode disabled. If
        # ``reshape`` must copy a non-contiguous input, the copy does not retain
        # ``requires_grad``. Capture the output requirement before flattening.
        out_requires_grad = hidden_states.requires_grad or weight.requires_grad
        h_2d = hidden_states.reshape(-1, hidden_states.size(-1))
        l_1d = labels.reshape(-1)
        T = l_1d.shape[0]

        log_probs = torch.zeros(T, device=h_2d.device, dtype=torch.float32, requires_grad=out_requires_grad)
        # Entropy is returned in fp32; verl downcasts to ``hidden_states.dtype``
        # but downstream RL code reads it in fp32 so we keep the higher
        # precision here. The mask-to-zero at IGN positions is exact in either
        # dtype, so this only affects the valid slots.
        entropy = torch.zeros(T, device=h_2d.device, dtype=torch.float32, requires_grad=out_requires_grad)

        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            h_chunk = h_2d[chunk_start:chunk_end]
            l_chunk = l_1d[chunk_start:chunk_end]

            # Op order mirrors verl's ``_fused_linear_for_ppo_fwd``:
            # ``(h @ w.t()) / T`` in the input dtype, then upcast to
            # fp32 before computing log-probs. For bf16 inputs this
            # rounds the temperature scale at bf16 precision (matches
            # verl bit-for-bit); for fp32 inputs the cast is a no-op
            # so behaviour is unchanged.
            logits = h_chunk @ weight.t()
            if temperature != 1.0:
                logits = logits / temperature
            logits = logits.float()

            log_probs[chunk_start:chunk_end] = token_logprobs_fn(logits, l_chunk, ignore_index=ignore_index)
            chunk_entropy = token_entropy_from_logits(logits)
            # Mask entropy at IGN positions (mirrors log_probs masking).
            # Entropy is well-defined at every position (it depends only on
            # the softmax distribution, not the label), but downstream
            # consumers expect IGN slots to be exactly 0 so they can sum
            # without applying a separate mask.
            mask = l_chunk != ignore_index
            entropy[chunk_start:chunk_end] = torch.where(mask, chunk_entropy, torch.zeros_like(chunk_entropy))

        ctx.save_for_backward(h_2d, weight, l_1d)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.orig_hidden_shape = orig_hidden_shape

        return log_probs.view(orig_shape), entropy.view(orig_shape)

    @staticmethod
    def backward(ctx, dlog_probs: Optional[Tensor], dentropy: Optional[Tensor]):
        """Recompute chunked logits and accumulate gradients into ``hidden_states`` and ``weight``."""
        if dlog_probs is None and dentropy is None:
            return None, None, None, None, None, None, None

        h_2d, weight, l_1d = ctx.saved_tensors
        T = l_1d.shape[0]
        dlog_probs_1d = dlog_probs.reshape(-1).float() if dlog_probs is not None else None
        dentropy_1d = dentropy.reshape(-1).float() if dentropy is not None else None

        # Saved tensors may have different ``requires_grad`` flags after a
        # reshape copy. Gate gradients on the original Function inputs.
        dhidden = torch.zeros_like(h_2d) if ctx.needs_input_grad[0] else None
        dweight = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None

        for chunk_start in range(0, T, ctx.chunk_size):
            chunk_end = min(chunk_start + ctx.chunk_size, T)
            h_chunk = h_2d[chunk_start:chunk_end]
            l_chunk = l_1d[chunk_start:chunk_end]
            dlp_chunk = dlog_probs_1d[chunk_start:chunk_end] if dlog_probs_1d is not None else None
            dent_chunk = dentropy_1d[chunk_start:chunk_end] if dentropy_1d is not None else None

            # Recompute logits with the same op order as forward so the
            # saved-weight reference (which FSDP2 has unsharded by now
            # via its pre-backward hook) lands the same matmul +
            # rounding boundary as the kernel forward / verl forward.
            logits = h_chunk @ weight.t()
            if ctx.temperature != 1.0:
                logits = logits / ctx.temperature
            logits = logits.float()

            probs = logits.softmax(dim=-1)
            mask = (l_chunk != ctx.ignore_index).float()

            dlogits = torch.zeros_like(probs)

            # ── log_probs gradient path ──────────────────────────────────
            # ∂(gather(log_softmax(logits), labels)) / ∂logits[i, j] =
            #     δ(j == labels[i]) - softmax(logits)[i, j]
            # so dlogits[i, j] = dlog_probs[i] * (one_hot[i, j] - probs[i, j]).
            if dlp_chunk is not None:
                safe_labels = l_chunk.clamp(min=0).unsqueeze(-1)
                one_hot = torch.zeros_like(probs).scatter_(-1, safe_labels, 1.0)
                masked_dlp = (dlp_chunk * mask).unsqueeze(-1)
                dlogits = dlogits + masked_dlp * (one_hot - probs)

            # ── entropy gradient path (mirrors verl exactly) ─────────────
            # ∂H[p]/∂logits[i, j] = p_j (log p_j + H[p]) so
            # dlogits[i, j] = -dentropy[i] * p_j * (log p_j + H[p_i])
            # The ``-`` sign is folded into the ``-dentropy.unsqueeze(-1)``
            # multiplier (matches verl's expression).
            if dent_chunk is not None:
                log_probs_full = logits.log_softmax(dim=-1)
                entropy_full = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
                masked_dent = (dent_chunk * mask).unsqueeze(-1)
                dlogits = dlogits + probs * (log_probs_full + entropy_full.unsqueeze(-1)) * (-masked_dent)

            # Op order mirrors verl's ``_fused_linear_for_ppo_bwd``:
            # cast back to the input dtype first, then divide by
            # temperature, *then* matmul. The cast-before-divide order
            # changes the bf16 rounding, so swapping it is what makes
            # this kernel's gradients bitwise identical to verl.
            dlogits = dlogits.to(h_chunk.dtype)
            if ctx.temperature != 1.0:
                dlogits = dlogits / ctx.temperature

            if dhidden is not None:
                dhidden[chunk_start:chunk_end] = dlogits @ weight
            if dweight is not None:
                dweight += dlogits.t() @ h_chunk

        if dhidden is not None:
            dhidden = dhidden.view(ctx.orig_hidden_shape)

        return dhidden, dweight, None, None, None, None, None


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
    3. ``distillation_losses`` grad. With
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
        hidden_states: Tensor,
        weight: Tensor,
        labels: Tensor,
        teacher_topk_ids: Tensor,
        teacher_topk_log_probs: Tensor,
        temperature: float,
        chunk_size: int,
        ignore_index: int,
        log_prob_min_clamp: Optional[float],
        token_logprobs_fn: Optional[Callable[..., Tensor]] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Stream the lm_head projection chunk-by-chunk for top-k forward-KL distillation."""
        ctx.set_materialize_grads(False)
        if token_logprobs_fn is None:
            token_logprobs_fn = resolve_token_logprobs_fn(device_type=hidden_states.device.type)

        orig_shape = labels.shape
        orig_hidden_shape = hidden_states.shape
        K = teacher_topk_ids.shape[-1]

        # ``autograd.Function.forward`` runs with grad mode disabled. If
        # ``reshape`` must copy a non-contiguous input, the copy does not retain
        # ``requires_grad``. Capture the output requirement before flattening.
        out_requires_grad = hidden_states.requires_grad or weight.requires_grad
        h_2d = hidden_states.reshape(-1, hidden_states.size(-1))
        l_1d = labels.reshape(-1)
        ids_2d = teacher_topk_ids.reshape(-1, K)
        tlp_2d = teacher_topk_log_probs.reshape(-1, K)
        T = l_1d.shape[0]

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

            # Op order mirrors ``_ChunkedLinearLogProbs`` exactly so the shared
            # (log_probs, entropy) outputs hit the same fp32 rounding
            # boundary as verl's ``FusedLinearForPPOFunction``.
            logits = h_chunk @ weight.t()
            if temperature != 1.0:
                logits = logits / temperature
            logits = logits.float()

            log_probs[chunk_start:chunk_end] = token_logprobs_fn(logits, l_chunk, ignore_index=ignore_index)
            chunk_entropy = token_entropy_from_logits(logits)
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
        dlog_probs: Optional[Tensor],
        dentropy: Optional[Tensor],
        ddistill: Optional[Tensor],
        dstudent_mass: Optional[Tensor],  # noqa: ARG004 - student_mass is detached
        dteacher_mass: Optional[Tensor],  # noqa: ARG004 - teacher_mass is detached
    ):
        """Recompute chunked logits and accumulate gradients for log-probs, entropy, and top-k KL."""
        if dlog_probs is None and dentropy is None and ddistill is None:
            return None, None, None, None, None, None, None, None, None, None

        h_2d, weight, l_1d, ids_2d, tlp_2d = ctx.saved_tensors
        T = l_1d.shape[0]
        dlog_probs_1d = dlog_probs.reshape(-1).float() if dlog_probs is not None else None
        dentropy_1d = dentropy.reshape(-1).float() if dentropy is not None else None
        ddistill_1d = ddistill.reshape(-1).float() if ddistill is not None else None

        # Saved tensors may have different ``requires_grad`` flags after a
        # reshape copy. Gate gradients on the original Function inputs.
        dhidden = torch.zeros_like(h_2d) if ctx.needs_input_grad[0] else None
        dweight = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None

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
            if ddist_chunk is not None:
                if ctx.log_prob_min_clamp is not None:
                    student_log_probs = logits.log_softmax(dim=-1)
                    student_topk_lp = student_log_probs.gather(dim=-1, index=ids_chunk)
                    active = (student_topk_lp >= ctx.log_prob_min_clamp).float()
                    tlp_f32 = tlp_chunk.clamp_min(ctx.log_prob_min_clamp).float()
                else:
                    active = None
                    tlp_f32 = tlp_chunk.float()

                p_t_k = tlp_f32.exp()  # [chunk, K]
                if active is not None:
                    p_t_k_eff = p_t_k * active
                else:
                    p_t_k_eff = p_t_k
                p_t_dense = torch.zeros_like(probs).scatter_add_(-1, ids_chunk, p_t_k_eff)
                teacher_mass_eff = p_t_k_eff.sum(dim=-1, keepdim=True)  # [chunk, 1]
                masked_ddist = (ddist_chunk * mask).unsqueeze(-1)
                dlogits = dlogits + masked_ddist * (-p_t_dense + teacher_mass_eff * probs)

            # Op order mirrors ``_ChunkedLinearLogProbs``: cast to input dtype
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

        return dhidden, dweight, None, None, None, None, None, None, None, None


def chunk_linear_logprobs(
    hidden_states: Tensor,
    weights: Tensor,
    labels: Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    token_logprobs_fn: Optional[Callable[..., Tensor]] = None,
) -> tuple[Tensor, Tensor]:
    """Run the tensor-level chunked linear log-probs and entropy autograd Function.

    Assumes ``hidden_states`` and ``labels`` are already aligned along the
    sequence dimension (no causal shift or trailing padding is applied here).
    """
    return _ChunkedLinearLogProbs.apply(
        hidden_states,
        weights,
        labels,
        float(temperature),
        int(chunk_size),
        int(ignore_index),
        token_logprobs_fn,
    )


def chunk_linear_topk_distill(
    hidden_states: Tensor,
    weights: Tensor,
    labels: Tensor,
    teacher_topk_ids: Tensor,
    teacher_topk_log_probs: Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    log_prob_min_clamp: Optional[float] = None,
    token_logprobs_fn: Optional[Callable[..., Tensor]] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run the tensor-level chunked linear top-k forward-KL distillation autograd Function.

    Assumes ``hidden_states``, ``labels``, and ``teacher_topk_*`` are already
    aligned along the sequence dimension (no causal shift or trailing padding
    is applied here).
    """
    log_probs, entropy, distill, student_mass, teacher_mass = _ChunkedLinearTopkDistill.apply(
        hidden_states,
        weights,
        labels,
        teacher_topk_ids,
        teacher_topk_log_probs,
        float(temperature),
        int(chunk_size),
        int(ignore_index),
        log_prob_min_clamp,
        token_logprobs_fn,
    )
    return log_probs, entropy, distill, student_mass.detach(), teacher_mass.detach()
