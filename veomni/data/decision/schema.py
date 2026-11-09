"""Typed decision request schema and deterministic option augmentation."""

import math
import random
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Choice:
    key: str
    description: object = None


@dataclass(frozen=True)
class Question:
    key: str
    question_type: str
    instructions: object
    choices: tuple[Choice, ...]
    target: tuple[float, ...] | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class Request:
    state: object
    questions: tuple[Question, ...]
    choices_shuffled: bool = False


@dataclass(frozen=True)
class Case:
    case_id: str
    request: Request


def parse_case(row):
    request = row.get("request", row)
    questions = []
    for value in request["questions"]:
        choices = tuple(Choice(str(choice["key"]), choice.get("description")) for choice in value["choices"])
        target = value.get("target")
        keys = [choice.key for choice in choices]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Decision choices must have distinct, nonempty keys")
        if target is not None:
            target = tuple(float(item) for item in target)
            if len(target) != len(choices) or any(not math.isfinite(item) or item < 0 for item in target):
                raise ValueError("Decision target must match the options and contain finite nonnegative probabilities")
            if not math.isclose(sum(target), 1, rel_tol=0, abs_tol=1e-6):
                raise ValueError("Decision target probabilities must sum to one")
        weight = float(value.get("weight", 1))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Decision question weight must be finite and nonnegative")
        questions.append(
            Question(str(value["key"]), value["question_type"], value.get("instructions", ""), choices, target, weight)
        )
    if not questions or len({question.key for question in questions}) != len(questions):
        raise ValueError("Decision requests require distinct question keys")
    return Case(str(row["case_id"]), Request(request["state"], tuple(questions)))


def shuffled_request(case, seed, epoch):
    questions = []
    for question in case.request.questions:
        if question.question_type != "choice":
            questions.append(question)
            continue
        order = list(range(len(question.choices)))
        random.Random(f"{seed}:{epoch}:{case.case_id}:{question.key}").shuffle(order)
        questions.append(
            replace(
                question,
                choices=tuple(question.choices[index] for index in order),
                target=tuple(question.target[index] for index in order) if question.target is not None else None,
            )
        )
    return replace(case.request, questions=tuple(questions), choices_shuffled=True)
