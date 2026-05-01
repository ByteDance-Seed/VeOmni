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

"""Chunked fused linear log-probs for PPO-style RL.

Returns per-token actual log-probabilities ``log p(y_t | <t)`` while
streaming the lm_head projection chunk-by-chunk — never materializes
the full ``[T, V]`` logits tensor. The output is the building block
PPO needs for policy / reference logprob recompute at long context +
large vocab.

The ``LAST_LOG_PROBS`` ``ContextVar`` lets the loss-function dispatch
hand off per-token log-probs to a post-forward hook on the model
without abusing the ``output.logits`` slot. The kernel sets it; the
forward hook installed by ``build_foundation_model`` reads it and
attaches to ``output.log_probs``.

Implementation pattern follows
``verl/utils/experimental/torch_functional.py::FusedLinearForPPOFunction``
so VeOmni-built models drop into verl's existing fused-kernel flow
without behavioural surprises:

- Custom ``torch.autograd.Function`` with **explicit chunked
  backward**. Forward saves ``(hidden_states, weight, labels)``;
  backward chunks again and recomputes logits to derive
  ``dhidden_states = dlogits @ weight`` and ``dweight = dlogits.t()
  @ hidden_states``.
- ``hidden_states`` is reshaped to ``[T, H]`` internally; the output
  is reshaped back to the input's leading dims.
- ``temperature`` is applied as ``logits / T`` (chain rule
  divides ``dlogits`` by ``T`` in backward).

FSDP2 contract: the saved ``weight`` reference is the lm_head
parameter; FSDP2's pre-backward hook (installed by
``fully_shard()`` on the parent module) unshards the parameter
before this Function's backward fires, so ``weight @ ...`` and
``weight.t() @ ...`` see the unsharded data. This is the same
contract verl already validates in production with
``FusedLinearForPPOFunction``.

VeOmni-specific extensions on top of verl's pattern:

- ``ignore_index`` masking: ``log_probs == 0`` and zero gradient at
  positions where ``labels == ignore_index``. Needed because
  VeOmni's data pipeline (chat templates, packing) sets
  IGNORE_INDEX boundaries; verl's data pipeline filters them
  upstream so its kernel doesn't.
- Causal label shift (``labels[..., 1:]`` / ``hidden[..., :-1, :]``)
  applied internally when SP is disabled, matching the convention
  of the sibling ``chunk_loss_function``. SP-enabled callers pass
  pre-shifted labels via the dataloader and the shift here is
  skipped.

When ``flash_attn.ops.triton.cross_entropy.cross_entropy_loss`` is
importable we route the per-token NLL through it (verl's preferred
path); without flash_attn we fall back to ``log_softmax + gather``.
The fa_ce path is what makes per-token log-probs **bitwise** identical
to verl's ``FusedLinearForPPOFunction`` under deterministic + batch-
invariant mode — both stacks call the same triton kernel on the same
fp32 logits.
"""

from typing import Optional

import torch

from ....distributed.parallel_state import get_parallel_state


try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss as _fa_cross_entropy_loss

    _FA_CE_AVAILABLE = True
except ImportError:
    _FA_CE_AVAILABLE = False


def _per_token_log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Per-token ``log p(y_t)`` from fp32 logits.

    Uses ``flash_attn.ops.triton.cross_entropy.cross_entropy_loss`` when
    available — same op verl's ``FusedLinearForPPOFunction`` calls, so
    we inherit the same fp32 rounding and stay bitwise with verl's
    output. Falls back to ``log_softmax + gather`` with manual
    ``ignore_index`` masking otherwise. Both paths return zero at
    ``labels == ignore_index`` (fa_ce sets per-token loss to 0 at IGN
    positions internally, the negation is then ``-0.0`` which compares
    equal to ``0.0``).
    """
    if _FA_CE_AVAILABLE:
        per_token_nll = _fa_cross_entropy_loss(logits, labels, ignore_index=ignore_index)[0]
        return -per_token_nll

    mask = labels != ignore_index
    safe_labels = labels.clamp(min=0).unsqueeze(-1)
    log_probs = logits.log_softmax(dim=-1).gather(-1, safe_labels).squeeze(-1)
    return torch.where(mask, log_probs, torch.zeros_like(log_probs))


class _ChunkedLinearLogProbs(torch.autograd.Function):
    """Custom autograd Function: chunked linear projection + log-softmax + gather.

    Mirrors verl's ``FusedLinearForPPOFunction`` (custom autograd
    Function with explicit chunked forward + backward) so the FSDP2
    correctness story is the same.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        chunk_size: int,
        ignore_index: int,
    ) -> torch.Tensor:
        ctx.set_materialize_grads(False)

        orig_shape = labels.shape
        orig_hidden_shape = hidden_states.shape
        h_2d = hidden_states.reshape(-1, hidden_states.size(-1))
        l_1d = labels.reshape(-1)
        T = l_1d.shape[0]

        out_requires_grad = h_2d.requires_grad or weight.requires_grad
        log_probs = torch.zeros(T, device=h_2d.device, dtype=torch.float32, requires_grad=out_requires_grad)

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

            log_probs[chunk_start:chunk_end] = _per_token_log_probs_from_logits(logits, l_chunk, ignore_index)

        ctx.save_for_backward(h_2d, weight, l_1d)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.orig_hidden_shape = orig_hidden_shape

        return log_probs.view(orig_shape)

    @staticmethod
    def backward(ctx, dlog_probs: Optional[torch.Tensor]):
        if dlog_probs is None:
            return None, None, None, None, None, None

        h_2d, weight, l_1d = ctx.saved_tensors
        T = l_1d.shape[0]
        dlog_probs_1d = dlog_probs.reshape(-1).float()

        dhidden = torch.zeros_like(h_2d) if h_2d.requires_grad else None
        dweight = torch.zeros_like(weight) if weight.requires_grad else None

        for chunk_start in range(0, T, ctx.chunk_size):
            chunk_end = min(chunk_start + ctx.chunk_size, T)
            h_chunk = h_2d[chunk_start:chunk_end]
            l_chunk = l_1d[chunk_start:chunk_end]
            dlp_chunk = dlog_probs_1d[chunk_start:chunk_end]

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
            safe_labels = l_chunk.clamp(min=0).unsqueeze(-1)

            # ∂(gather(log_softmax(logits), labels)) / ∂logits[i, j] =
            #     δ(j == labels[i]) - softmax(logits)[i, j]
            # so dlogits[i, j] = dlog_probs[i] * (one_hot[i, j] - probs[i, j]).
            one_hot = torch.zeros_like(probs).scatter_(-1, safe_labels, 1.0)
            masked_dlp = (dlp_chunk * mask).unsqueeze(-1)
            dlogits = masked_dlp * (one_hot - probs)

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

        return dhidden, dweight, None, None, None, None


def chunk_logprobs_function(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute per-token actual log-probabilities via a chunked fused linear.

    Args:
        hidden_states: ``[B, L, H]`` (or ``[L, H]`` for already-packed
            inputs).
        weights: lm_head weight, ``[V, H]``. Bias is not supported.
        labels: integer label tensor with shape matching the leading
            dims of ``hidden_states``. Positions equal to
            ``ignore_index`` produce ``0.0`` in the output (no
            gradient flows through them either).
        chunk_size: token-dim chunk for the streamed projection.
        ignore_index: label value to mask (default ``-100`` /
            ``IGNORE_INDEX``).
        shift_labels: pre-shifted target labels. When provided, the
            kernel uses them as-is and skips the internal causal
            shift, so callers can supply custom alignment (e.g. the
            ``ForCausalLMLoss`` SP path that pre-shifts via padding).
            The output tensor's seq length matches ``shift_labels``
            (no trailing pad). When ``None``, this function applies
            the causal ``labels[..., 1:]`` /
            ``hidden_states[..., :-1, :]`` shift internally and pads
            the trailing seq slot with ``0.0`` so the output shape
            matches the input ``labels``.
        temperature: divides logits before log_softmax (PPO actor
            path). Defaults to 1.0 (no-op).

    Returns:
        Per-token log-probabilities with the same shape as the input
        ``labels``. **Sign: non-positive** (``log p(y_t)``), matches
        HF / verl conventions — no negation needed at the call site.
    """
    sp_enabled = get_parallel_state().sp_enabled

    # Three modes for choosing the per-position target (matches the
    # ``ForCausalLMLoss`` contract):
    # 1. Caller passes pre-shifted labels via ``shift_labels`` -> trust them
    #    and run hidden_states unchanged. Output keeps the input's seq length.
    # 2. SP enabled, ``shift_labels`` not provided -> SequenceParallelCollator
    #    has already globally shifted ``labels``; don't shift again.
    # 3. SP disabled, ``shift_labels`` not provided -> apply the causal
    #    ``labels[..., 1:]`` / ``hidden[..., :-1, :]`` shift here and pad
    #    the trailing seq slot with 0 so the returned shape matches input
    #    ``labels``.
    used_explicit_shift = shift_labels is not None
    if used_explicit_shift:
        labels_shifted = shift_labels
    elif sp_enabled:
        labels_shifted = labels
    else:
        labels_shifted = labels[..., 1:].contiguous()
        hidden_states = hidden_states[..., :-1, :].contiguous()

    log_probs = _ChunkedLinearLogProbs.apply(
        hidden_states, weights, labels_shifted, float(temperature), int(chunk_size), int(ignore_index)
    )

    if not sp_enabled and not used_explicit_shift:
        # Pad with one zero at the right of the (last) seq dim so the
        # returned tensor matches the input ``labels`` shape. The
        # padded slot corresponds to the final input token (no
        # next-token target) — a no-op under any sane downstream mask.
        log_probs = torch.nn.functional.pad(log_probs, (0, 1), value=0.0)
    return log_probs
