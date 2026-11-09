"""CLEF request encoding, model options, and registered factories."""

import json
from dataclasses import dataclass

from .....utils.decision_loss import DecisionScoringLoss
from ...base import DecisionModelSpec, decode_decisions


QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"


def render(value):
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def encode_clef(request, tokenizer, *, max_length=16384, max_state_tokens=None):
    """Encode one text case with the CLEF serialization supplied for this integration."""
    if not isinstance(request.state, str):
        raise ValueError(
            "CLEF requires a string state; serialize objects before encoding. Image/video is unsupported."
        )
    schema_ids = []
    question_spans, option_spans, option_ids = [], [], []

    def append(destination, text):
        destination.extend(tokenizer.encode(text, add_special_tokens=False))

    append(schema_ids, "\n\nSCHEMA FIELDS:\n")
    for index, question in enumerate(request.questions):
        append(schema_ids, f"\nFIELD {index + 1}\nID: {question.key}\nTYPE: {question.question_type}\nINSTRUCTION: ")
        start = len(schema_ids)
        append(schema_ids, render(question.instructions))
        question_spans.append((start, len(schema_ids)))
        append(schema_ids, "\nALLOWED OPTIONS:\n")
        spans = []
        choices = question.choices
        if question.question_type == "choice" and not getattr(request, "choices_shuffled", False):
            choices = tuple(sorted(choices, key=lambda choice: choice.key))
        elif question.question_type == "noul":
            choices = tuple(sorted(choices, key=lambda choice: (choice.key != "true", choice.key)))
        option_ids.append(tuple(choice.key for choice in choices))
        for option_index, choice in enumerate(choices):
            append(schema_ids, f"OPTION {option_index + 1}: ")
            start = len(schema_ids)
            semantics = {"option_id": choice.key}
            if choice.description is not None:
                semantics["description"] = choice.description
            append(schema_ids, render(semantics))
            spans.append((start, len(schema_ids)))
            append(schema_ids, "\n")
        option_spans.append(spans)
        append(schema_ids, "END FIELD\n")
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    state_ids = tokenizer.encode(request.state, add_special_tokens=False)
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise ValueError(f"CLEF schema requires {fixed_length} tokens before state; maximum is {max_length}")
    state_ids = state_ids[: max_length - fixed_length]
    schema_offset = len(prefix_ids) + len(state_ids)
    question_spans = [(start + schema_offset, end + schema_offset) for start, end in question_spans]
    option_spans = [[(start + schema_offset, end + schema_offset) for start, end in spans] for spans in option_spans]
    input_ids = prefix_ids + state_ids + schema_ids + suffix_ids
    if any(end <= start for start, end in question_spans + [span for spans in option_spans for span in spans]):
        raise ValueError("CLEF question and option spans must contain at least one token")
    return input_ids, {
        "question_spans": question_spans,
        "option_spans": option_spans,
        "option_ids": option_ids,
        "kinds": [QUESTION_TYPES[question.question_type] for question in request.questions],
        "counts": [len(request.questions)],
    }


@dataclass(frozen=True)
class ClefModelOptions:
    head_config: dict | str | None = None
    head_path: str | None = None


def build_clef_loss(name=None, **kwargs):
    return DecisionScoringLoss(name or "smooth_brier", **kwargs)


def build_clef_model(*args, decision_model_config=None, **kwargs):
    from .modeling import load_local_model

    options = ClefModelOptions(**(decision_model_config or {}))
    return load_local_model(*args, head_config=options.head_config, head_path=options.head_path, **kwargs)


def build_clef_preprocessor(*args, **kwargs):
    from .processing import ClefPreprocessor

    return ClefPreprocessor.from_assets(*args, **kwargs)


def export_clef(*args, model, **kwargs):
    from .release import save_clef_release

    return save_clef_release(
        *args,
        head_config=model.head_config,
        scalar_head_parameters=model.scalar_head_parameters,
        **kwargs,
    )


CLEF_SPEC = DecisionModelSpec(
    name="clef",
    build_loss=build_clef_loss,
    default_loss="smooth_brier",
    build_model=build_clef_model,
    build_preprocessor=build_clef_preprocessor,
    export=export_clef,
    decode=decode_decisions,
)
