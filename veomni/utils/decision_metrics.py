"""Weighted decision metrics, accumulated before distributed reduction."""

import torch

from .decision_loss import choice_log_probs


METRIC_NAMES = ("cross_entropy", "kl", "brier", "accuracy")


def metric_totals(output, batch):
    logits = output.logits
    counts = batch["decision_supervision"]["counts"].to(logits.device)
    targets = batch["decision_supervision"]["targets"].to(logits.device).float()
    weights = batch["decision_supervision"]["weights"].to(logits.device).float() * (counts > 1)
    log_probs = choice_log_probs(logits, counts)
    probabilities = log_probs.exp()
    valid = torch.arange(logits.shape[1], device=logits.device)[None] < counts[:, None]
    ce = -(targets * log_probs.masked_fill(~valid, 0)).sum(-1)
    entropy = -(targets * targets.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)
    brier = (probabilities - targets).square().sum(-1) / counts
    accuracy = (probabilities.argmax(-1) == targets.argmax(-1)).float()
    ordinal = batch["decision_supervision"]["ordinal_mask"].to(logits.device)
    thresholds = torch.arange(logits.shape[1], device=logits.device)[None] < (counts - 1)[:, None]
    rps = ((probabilities.cumsum(-1) - targets.cumsum(-1)).square() * thresholds).sum(-1)
    rps = rps / (counts - 1).clamp_min(1)
    return torch.stack(
        [
            weights.sum(),
            (ce * weights).sum(),
            ((ce - entropy) * weights).sum(),
            (brier * weights).sum(),
            (accuracy * weights).sum(),
            (rps * weights * ordinal).sum(),
            (weights * ordinal).sum(),
        ]
    ).detach()


def finalize_metrics(totals):
    values = totals.cpu().tolist()
    weight = values[0]
    result = {"question_weight": weight}
    result.update({name: values[index + 1] / weight if weight else None for index, name in enumerate(METRIC_NAMES)})
    result["rps"] = values[5] / values[6] if values[6] else None
    return result
