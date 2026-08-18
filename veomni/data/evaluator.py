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

"""Evaluation metrics for training-time validation.

Evaluators are registered via :data:`EVALUATOR_REGISTRY` so that users can
add custom metrics without modifying trainer code::

    @EVALUATOR_REGISTRY.register("my_metric")
    class MyEvaluator(Evaluator):
        def compute(self, logits, labels, **kwargs):
            ...

All evaluators are distributed-aware: per-rank partial sums are aggregated
via ``all_reduce`` so the final metric reflects the global validation set.
"""

import math
from typing import Dict, List

import torch
import torch.distributed as dist

from ..utils import logging
from ..utils.registry import Registry


logger = logging.get_logger(__name__)

EVALUATOR_REGISTRY = Registry("evaluator")


class Evaluator:
    """Base class for evaluation metrics.

    Subclasses implement :meth:`compute` which receives raw tensors from the
    model forward pass and returns a dict of metric_name -> scalar value.

    The base class handles distributed aggregation: each rank computes a
    partial sum and count, then ``all_reduce`` combines them so the returned
    metric is the global average across all DP ranks.
    """

    def compute(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute per-rank partial metrics.

        Args:
            logits: Model output logits of shape ``(batch, seq, vocab)`` or ``(batch, vocab)``.
            labels: Ground-truth labels of shape ``(batch, seq)`` or ``(batch,)``.
            **kwargs: Additional forward outputs (e.g. ``loss``).

        Returns:
            Dict with keys ``"<metric_name>"`` (per-rank sum) and
            ``"<metric_name>_count"`` (per-rank sample/token count).
        """
        raise NotImplementedError

    def aggregate(self, partial: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """All-reduce partial sums across DP ranks and compute final values.

        Args:
            partial: Output of :meth:`compute` from the local rank.

        Returns:
            Dict of metric_name -> float (global average).
        """
        result: Dict[str, float] = {}
        metric_names = [k for k in partial if not k.endswith("_count")]

        for name in metric_names:
            count_key = f"{name}_count"
            total = partial[name].detach().clone()
            count = partial[count_key].detach().clone()

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)

            if count.item() > 0:
                result[name] = (total / count).item()
            else:
                result[name] = float("nan")

        return result


@EVALUATOR_REGISTRY.register("perplexity")
class PerplexityEvaluator(Evaluator):
    """Token-weighted perplexity.

    Accumulates total negative log-likelihood and token count across all
    validation batches, then computes ``exp(total_nll / total_tokens)``.
    This correctly handles variable-length sequences and distributed shards.
    """

    def compute(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Handle both (batch, seq, vocab) and (batch, vocab) shapes
        if logits.dim() == 3:
            # Shift for causal LM: predict token t+1 from token t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels

        # Ignore index -100 (HuggingFace convention)
        ignore_index = -100
        mask = shift_labels != ignore_index

        # Compute per-token NLL using log_softmax for numerical stability
        log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
        # Gather the log-prob of the correct token
        gathered = log_probs.gather(dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        # Zero out ignored positions
        gathered = gathered * mask
        total_nll = -gathered.sum()
        total_tokens = mask.sum()

        return {
            "perplexity": total_nll,
            "perplexity_count": total_tokens,
        }

    def aggregate(self, partial: Dict[str, torch.Tensor]) -> Dict[str, float]:
        result = super().aggregate(partial)
        # Convert average NLL to perplexity
        val = result.get("perplexity")
        if val is not None and not math.isnan(val):
            result["perplexity"] = math.exp(val)
        return result


@EVALUATOR_REGISTRY.register("accuracy")
class AccuracyEvaluator(Evaluator):
    """Classification accuracy.

    Compares ``argmax(logits)`` to ``labels`` and averages over all valid
    (non-ignored) positions. Works for both sequence classification and
    token-level classification.
    """

    def compute(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        if logits.dim() == 3:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels

        ignore_index = -100
        mask = shift_labels != ignore_index

        predictions = shift_logits.argmax(dim=-1)
        correct = (predictions == shift_labels) & mask

        return {
            "accuracy": correct.sum().float(),
            "accuracy_count": mask.sum().float(),
        }


@EVALUATOR_REGISTRY.register("token_accuracy")
class TokenAccuracyEvaluator(Evaluator):
    """Token-level accuracy (alias for accuracy with a different name).

    Useful when both sequence-level and token-level metrics are needed
    in the same validation run.
    """

    def compute(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        if logits.dim() == 3:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels

        ignore_index = -100
        mask = shift_labels != ignore_index

        predictions = shift_logits.argmax(dim=-1)
        correct = (predictions == shift_labels) & mask

        return {
            "token_accuracy": correct.sum().float(),
            "token_accuracy_count": mask.sum().float(),
        }


@EVALUATOR_REGISTRY.register("loss")
class LossEvaluator(Evaluator):
    """Average validation loss.

    Uses the model's native loss when available (``kwargs["loss"]``),
    otherwise computes cross-entropy from logits and labels.
    """

    def compute(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        loss = kwargs.get("loss")
        if loss is not None:
            # Model already computed the loss; use it directly
            if logits.dim() == 3:
                shift_labels = labels[..., 1:].contiguous()
            else:
                shift_labels = labels
            ignore_index = -100
            token_count = (shift_labels != ignore_index).sum()
            return {
                "val_loss": loss * token_count,
                "val_loss_count": token_count.float(),
            }

        # Fallback: compute CE loss from logits
        if logits.dim() == 3:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels

        ignore_index = -100
        mask = shift_labels != ignore_index
        ce_loss = torch.nn.functional.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )
        token_count = mask.sum()

        return {
            "val_loss": ce_loss,
            "val_loss_count": token_count.float(),
        }


def build_evaluator(name: str) -> Evaluator:
    """Look up an evaluator by its registry name.

    Args:
        name: Registry key, e.g. ``"perplexity"``, ``"accuracy"``.

    Returns:
        An :class:`Evaluator` instance.
    """
    return EVALUATOR_REGISTRY[name]()


def compute_metrics(
    evaluators: List[Evaluator],
    logits: torch.Tensor,
    labels: torch.Tensor,
    **kwargs,
) -> Dict[str, float]:
    """Run multiple evaluators and return aggregated metrics.

    Each evaluator computes a per-rank partial, then all partials are
    all-reduced to produce global averages.

    Args:
        evaluators: List of :class:`Evaluator` instances.
        logits: Model output logits.
        labels: Ground-truth labels.
        **kwargs: Extra forward outputs passed to each evaluator.

    Returns:
        Dict mapping metric names to float values.
    """
    results: Dict[str, float] = {}
    for evaluator in evaluators:
        partial = evaluator.compute(logits, labels, **kwargs)
        results.update(evaluator.aggregate(partial))
    return results
