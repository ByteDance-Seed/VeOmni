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

"""Candidate-scoring objectives and their shared decision-batch adapter."""

from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as functional


LOSS_TYPES = {"soft_ce", "smooth_brier", "distill_rps"}


@dataclass(frozen=True)
class DecisionScoringLoss:
    """Adapt common supervision tensors without reading model-specific metadata."""

    loss_type: str = "soft_ce"
    smoothing: float = 0.1
    brier: float = 1.0
    rps: float = 0.5
    target_sum_atol: float = 1e-6

    def __post_init__(self):
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(f"Unknown decision loss {self.loss_type!r}")

    @property
    def config(self):
        return asdict(self)

    def __call__(self, output, batch, device):
        return decision_loss(
            output.logits,
            batch["decision_supervision"]["counts"].to(device),
            batch["decision_supervision"]["targets"].to(device),
            batch["decision_supervision"]["weights"].to(device),
            ordinal_mask=batch["decision_supervision"]["ordinal_mask"].to(device)
            if self.loss_type == "distill_rps"
            else None,
            **self.config,
        )


def option_mask(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Return the valid-candidate mask for logits [questions, options]."""
    if logits.ndim != 2 or counts.shape != logits.shape[:1] or counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected candidate logits [questions, options] and integer option counts")
    if counts.device != logits.device or torch.any((counts < 1) | (counts > logits.shape[1])):
        raise ValueError("Option counts must be on the logits device and within the padded width")
    return torch.arange(logits.shape[1], device=counts.device)[None] < counts[:, None]


def choice_log_probs(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Compute FP32 log probabilities, excluding padded candidates from softmax."""
    valid = option_mask(logits, counts)
    if not torch.isfinite(logits.masked_select(valid)).all():
        raise ValueError("Active candidate logits must be finite")
    return functional.log_softmax(logits.float().masked_fill(~valid, -torch.inf), dim=-1)


def decision_loss(
    logits: torch.Tensor,
    counts: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    loss_type: str = "soft_ce",
    ordinal_mask: torch.Tensor | None = None,
    smoothing: float = 0.1,
    brier: float = 1.0,
    rps: float = 0.5,
    target_sum_atol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a weighted loss numerator and effective weight, without averaging.

    ``targets`` contains soft distributions shaped like ``logits``; ``counts``
    and ``weights`` have one entry per question. Supervised targets sum to one
    within ``target_sum_atol`` and assign zero mass to padded candidates.
    ``distill_rps`` requires a boolean ``ordinal_mask`` selecting questions whose
    candidates are ordered. Zero-weight and single-candidate questions contribute
    no effective weight. Distributed reduction belongs to the caller.
    """
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"Unknown decision loss {loss_type!r}")
    log_probs = choice_log_probs(logits, counts)
    if targets.shape != logits.shape or weights.shape != counts.shape:
        raise ValueError("Targets must match candidate logits; weights must have one entry per question")
    if targets.device != logits.device or weights.device != logits.device:
        raise ValueError("All objective inputs must be on the same device")
    valid = option_mask(logits, counts)
    if not torch.isfinite(targets).all() or not torch.isfinite(weights).all():
        raise ValueError("Targets and weights must be finite")
    if torch.any(targets < 0) or torch.any(weights < 0) or torch.any(targets.masked_select(~valid) != 0):
        raise ValueError("Targets/weights must be nonnegative with no mass on padding")
    if not 0 <= target_sum_atol < float("inf"):
        raise ValueError("Target sum tolerance must be finite and nonnegative")
    supervised_sums = targets.float().sum(-1)[weights > 0]
    if not torch.allclose(supervised_sums, torch.ones_like(supervised_sums), atol=target_sum_atol, rtol=0):
        raise ValueError("Each supervised target must sum to one")
    if not 0 <= smoothing <= 1 or min(brier, rps) < 0:
        raise ValueError("Invalid label-smoothing or calibration loss weights")
    target = targets.float()
    if loss_type == "smooth_brier":
        target = (1 - smoothing) * target + smoothing * valid / counts[:, None]
    per_question = -(target * log_probs.masked_fill(~valid, 0)).sum(-1)
    probabilities = log_probs.exp()
    if loss_type == "smooth_brier":
        per_question = per_question + brier * ((probabilities - targets).square() * valid).sum(-1) / counts
    if loss_type == "distill_rps":
        if (
            ordinal_mask is None
            or ordinal_mask.shape != counts.shape
            or ordinal_mask.dtype != torch.bool
            or ordinal_mask.device != logits.device
        ):
            raise ValueError("distill_rps requires a boolean ordinal mask per question on the logits device")
        cumulative_gap = (probabilities.cumsum(-1) - targets.cumsum(-1)).square()
        thresholds = torch.arange(logits.shape[1], device=logits.device)[None] < (counts - 1)[:, None]
        ordinal = (cumulative_gap * thresholds).sum(-1) / (counts - 1).clamp(min=1)
        per_question = per_question + rps * ordinal * ordinal_mask
    effective_weights = weights.float() * (counts > 1)
    return (per_question * effective_weights).sum(), effective_weights.sum()
