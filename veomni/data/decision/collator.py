"""Dict decision samples and request-aware supervision on shared token packing."""

from dataclasses import dataclass, field
from itertools import chain

import torch
import torch.nn.functional as F

from veomni.data.data_collator import MainCollator


DECISION_FIELDS = {"decision_meta", "decision_supervision", "decision_request_lengths"}


def model_features(sample):
    """Select backbone inputs without passing decision targets or metadata."""
    return {key: value for key, value in sample.items() if key not in DECISION_FIELDS and key != "padding_flag"}


@dataclass
class DecisionCollator:
    """Pack requests with the shared collator and align their question targets."""

    padding_case: dict | None = None
    feature_collator: object = field(default_factory=MainCollator)

    def __call__(self, cases):
        samples = []
        padding = []
        for case in cases:
            is_padding = case is None
            samples.append(self.padding_case if is_padding else case)
            padding.append(is_padding)
        batch = dict(self.feature_collator([model_features(sample) for sample in samples]))
        supervision = [sample["decision_supervision"] for sample in samples]
        max_options = max(item["targets"].shape[-1] for item in supervision)
        batch["decision_supervision"] = {
            "counts": torch.cat([item["counts"] for item in supervision]),
            "targets": torch.cat(
                [F.pad(item["targets"], (0, max_options - item["targets"].shape[-1])) for item in supervision]
            ),
            "weights": torch.cat(
                [
                    torch.zeros_like(item["weights"]) if is_padding else item["weights"]
                    for item, is_padding in zip(supervision, padding, strict=True)
                ]
            ),
            "ordinal_mask": torch.cat([item["ordinal_mask"] for item in supervision]),
            "keys": tuple(chain.from_iterable(item["keys"] for item in supervision)),
            "question_keys": tuple(chain.from_iterable(item["question_keys"] for item in supervision)),
            "padding": tuple(padding),
        }
        batch["decision_meta"] = [sample["decision_meta"] for sample in samples]
        batch["decision_request_lengths"] = tuple(sample["input_ids"].numel() for sample in samples)
        return batch


def encode_request(request, *, preprocessor, max_request_tokens=16384):
    """Return an ordinary sample dict; batch layout is constructed after grouping."""
    features, meta = preprocessor(request, max_length=max_request_tokens)
    option_ids = meta.get("option_ids") or [
        tuple(choice.key for choice in question.choices) for question in request.questions
    ]
    counts = torch.tensor([len(keys) for keys in option_ids])
    targets = torch.zeros(len(request.questions), int(counts.max()))
    weights = torch.zeros(len(request.questions))
    for index, question in enumerate(request.questions):
        if question.target is not None:
            target_by_key = dict(zip((choice.key for choice in question.choices), question.target))
            targets[index, : len(option_ids[index])] = torch.tensor([target_by_key[key] for key in option_ids[index]])
            weights[index] = question.weight if len(option_ids[index]) > 1 else 0
    return {
        **features,
        "decision_meta": meta,
        "decision_supervision": {
            "counts": counts,
            "targets": targets,
            "weights": weights,
            "ordinal_mask": torch.tensor(
                [question.question_type == "score" for question in request.questions], dtype=torch.bool
            ),
            "keys": tuple(tuple(keys) for keys in option_ids),
            "question_keys": tuple(question.key for question in request.questions),
        },
    }
