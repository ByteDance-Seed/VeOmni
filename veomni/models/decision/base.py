"""Decision-model factories used by the shared runtime and trainer."""

from dataclasses import dataclass
from typing import Callable

import torch
from transformers.utils import ModelOutput

from ...utils.decision_loss import choice_log_probs


@dataclass
class ScoringOutput(ModelOutput):
    logits: torch.Tensor | None = None


def decode_decisions(output, batch):
    """Decode candidate scores in the option order carried by the batch."""
    probabilities = (
        choice_log_probs(output.logits, batch["decision_supervision"]["counts"].to(output.logits.device))
        .exp()
        .detach()
        .cpu()
    )
    return [
        {
            "question_key": question_key,
            "option_keys": list(keys),
            "probabilities": probabilities[index, : len(keys)].tolist(),
            "decision": keys[int(probabilities[index, : len(keys)].argmax())],
        }
        for index, (question_key, keys) in enumerate(
            zip(batch["decision_supervision"]["question_keys"], batch["decision_supervision"]["keys"], strict=True)
        )
    ]


@dataclass(frozen=True)
class DecisionModelSpec:
    name: str
    build_loss: Callable
    default_loss: str
    build_model: Callable
    build_preprocessor: Callable
    export: Callable
    decode: Callable


DECISION_MODELS: dict[str, DecisionModelSpec] = {}


def get_decision_model(name: str) -> DecisionModelSpec:
    if name not in DECISION_MODELS:
        raise ValueError(f"Unknown decision model {name!r}; registered models: {sorted(DECISION_MODELS)}")
    return DECISION_MODELS[name]
