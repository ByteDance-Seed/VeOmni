"""Hugging Face adapter for ``LocalLLaMA/typed-decisions``."""

import json

from datasets import load_dataset

from .schema import parse_case, shuffled_request


def _decode_json(value):
    return json.loads(value) if isinstance(value, str) else value


def parse_typed_decisions_case(row):
    """Convert one upstream typed-decisions row to VeOmni's decision schema."""
    state = _decode_json(row["state"])
    questions = _decode_json(row["questions"])
    gold = _decode_json(row["gold"])
    converted_questions = []

    for key, question in questions.items():
        question_type = question["type"]
        if question_type == "score":
            choices = [(str(index), description) for index, description in enumerate(question["criteria"])]
        elif question_type == "noul":
            criteria = question.get("criteria") or {}
            choices = [
                ("true", criteria.get("true", "The proposition is true or the answer is yes.")),
                ("false", criteria.get("false", "The proposition is false or the answer is no.")),
            ]
        elif question_type == "choice":
            choices = list(question["criteria"].items())
        else:
            raise ValueError(f"Unknown typed-decisions question type: {question_type}")

        target = gold.get(key)
        converted = {
            "key": key,
            "question_type": question_type,
            "instructions": question.get("instructions") or key,
            "choices": [{"key": choice_key, "description": description} for choice_key, description in choices],
            "weight": float(target.get("weight", 1.0)) if target is not None else 0.0,
        }
        if target is not None:
            probabilities = target["probabilities"]
            total = sum(probabilities.values())
            if total <= 0:
                raise ValueError(f"Question {key!r} has nonpositive total target probability")
            converted["target"] = [probabilities[choice_key] / total for choice_key, _ in choices]
        converted_questions.append(converted)

    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return parse_case({"case_id": row["id"], "request": {"state": state, "questions": converted_questions}})


def load_cases(options, split="train"):
    dataset = load_dataset(
        options.dataset,
        options.dataset_version,
        split=split,
        revision=options.dataset_revision,
        cache_dir=options.cache_dir,
    )
    cases = [parse_typed_decisions_case(row) for row in dataset]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError(f"Duplicate typed-decisions case_id in {split} split")
    return cases


__all__ = [
    "load_cases",
    "parse_typed_decisions_case",
    "shuffled_request",
]
