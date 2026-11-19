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
"""dLLM masked cross-entropy loss.

This is the diffusion-LM training objective: a *masked* cross-entropy that is
fundamentally different from the autoregressive ``ForCausalLMLoss`` and so does
NOT reuse that wrapper:

- **No label shift** — the mask predictor predicts the original token *in place*
  at each masked position, not the next token.
- **Only masked positions contribute** — ``labels`` carries the original token id
  at masked positions and ``ignore_index`` everywhere else (prompt / unmasked).
- **Per-token weighting** — each token carries a weight ``w_i = 1 / (t_s · N_s)``
  where ``t_s`` is the per-sample mask ratio and ``N_s`` the sample's answer
  length. Equivalent to::

      token_loss = CE(logits[mask], target[mask], reduction='none') / p_mask[mask]
      ce_loss    = sum(token_loss / answer_lengths[mask]) / B

  i.e. ``w_i = 1 / (p_mask_i · answer_length_i)`` and the final reduction divides
  by the number of samples ``B`` (not the number of tokens). The collator
  precomputes ``loss_weight`` (see the dLLM packing collator); this op only
  consumes it.

Reduction policy (kept deliberately minimal — global DP/SP normalization by the
*global* sample count is the trainer's job, mirroring how ``mean_global_loss``
all-reduces token counts for the causal path):

- ``"sum"``  → returns the local weighted sum ``Σ_i w_i · CE_i``. The trainer
  divides by the all-reduced global sample count.
- ``"mean"`` → returns ``Σ_i w_i · CE_i / num_samples`` (requires ``num_samples``).
  With ``num_samples == B`` this equals the official ``ce_loss`` above; used for
  single-process / unit-test parity.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def dllm_masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weight: torch.Tensor,
    num_samples: Optional[int] = None,
    ignore_index: int = -100,
    reduction: str = "sum",
) -> torch.Tensor:
    """Weighted masked cross-entropy for dLLM (mask-predictor) training.

    Args:
        logits: ``[..., vocab_size]`` prediction logits. Flattened internally and
            upcast to float32 for a numerically stable loss.
        labels: ``[...]`` long tensor; the original token id at masked positions
            and ``ignore_index`` at every non-loss position (prompt / unmasked).
        loss_weight: ``[...]`` float tensor of per-token weights ``1/(t_s·N_s)``.
            Values at ``ignore_index`` positions are never read.
        num_samples: number of samples in this (micro-)batch. Required for
            ``reduction="mean"``; ignored for ``reduction="sum"``.
        ignore_index: label value marking non-loss positions (default ``-100``).
        reduction: ``"sum"`` (local weighted sum, for distributed training) or
            ``"mean"`` (weighted sum / ``num_samples``, for single-process parity).

    Returns:
        A scalar loss tensor (connected to ``logits`` even when no token is
        masked, so the backward graph stays intact).
    """
    if reduction not in ("sum", "mean"):
        raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")

    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size).float()
    labels = labels.reshape(-1)
    loss_weight = loss_weight.reshape(-1).to(logits.dtype)

    mask = labels != ignore_index
    if not torch.any(mask):
        # No masked token in this shard — return a zero loss that still depends
        # on ``logits`` so the autograd graph (and FSDP reduce) is well-formed.
        zero = logits.sum() * 0.0
        return zero

    sel_logits = logits[mask]
    sel_labels = labels[mask].to(logits.device)
    sel_weight = loss_weight[mask].to(logits.device)

    # Per-token CE at masked positions, then apply the 1/(t·N) weighting.
    token_ce = F.cross_entropy(sel_logits, sel_labels, reduction="none")
    weighted_sum = (token_ce * sel_weight).sum()

    if reduction == "sum":
        return weighted_sum

    if num_samples is None:
        raise ValueError("reduction='mean' requires num_samples (the per-batch sample count).")
    num_samples = float(num_samples)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    return weighted_sum / num_samples
