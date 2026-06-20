"""Suite-level V2 request context and conversation stimulus materialization."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import ParityCase, make_reference_image
from tests.seed_omni.parity_suite.core.stimulus import conversation_stimulus_to_batched_specs
from veomni.models.seed_omni.conversation import ConversationItem


_ALLOWED_ITEM_TYPES = frozenset({"image", "text", "output"})
_RANDOM_DISTRIBUTIONS = frozenset({"uniform", "normal", "zeros", "ones"})


@dataclass(frozen=True)
class V2RequestContext:
    """Inputs passed to model-specific V2 request handlers."""

    case: ParityCase
    kind: str
    canonical: Mapping[str, Any]
    stimulus: Mapping[str, Any]
    reference_output: Any | None
    device: torch.device


# Public request entry ---------------------------------------------------------


def conversation_request_from_conversation_list(ctx: V2RequestContext) -> dict[str, Any]:
    """Build the default V2 request from authored conversation stimulus only."""

    conversation_list = conversation_stimulus_to_batched_specs(ctx.stimulus)
    if conversation_list is None:
        raise KeyError(
            "Default V2 conversation request requires stimulus.conversation_list "
            "or stimulus.batched_conversation_list."
        )
    return {
        "conversation_list": conversation_list_from_specs(
            conversation_list,
            case=ctx.case,
            canonical=ctx.canonical,
            device=ctx.device,
        )
    }


# Conversation materialization ------------------------------------------------


def conversation_list_from_specs(
    specs: Sequence[Sequence[Mapping[str, Any]]],
    *,
    case: ParityCase | None = None,
    canonical: Mapping[str, Any],
    device: torch.device,
) -> list[list[ConversationItem]]:
    del canonical
    _validate_sequence(specs, "conversation_list")
    conversation_list: list[list[ConversationItem]] = []
    for sample_index, sample in enumerate(specs):
        _validate_sequence(sample, "conversation_list sample")
        conversation_list.append(
            [
                _conversation_item_from_spec(
                    spec,
                    device=device,
                    case=case,
                    sample_index=sample_index,
                    item_index=item_index,
                )
                for item_index, spec in enumerate(sample)
            ]
        )
    return conversation_list


# Item materialization ---------------------------------------------------------


def _conversation_item_from_spec(
    spec: Mapping[str, Any],
    *,
    case: ParityCase | None,
    device: torch.device,
    sample_index: int,
    item_index: int,
) -> ConversationItem:
    if not isinstance(spec, Mapping):
        raise TypeError(
            "Conversation item spec must be a mapping with type/role/value. "
            "Use stimulus.batched_conversation_list for an explicit batch; "
            f"got {type(spec).__name__}."
        )
    if "value" not in spec:
        raise KeyError("Conversation item spec must declare value.")
    item_type = str(spec.get("type", "text"))
    if item_type not in _ALLOWED_ITEM_TYPES:
        raise ValueError(
            f"Unsupported conversation item type {item_type!r}; expected one of {sorted(_ALLOWED_ITEM_TYPES)}."
        )
    meta = _materialize_meta(spec.get("meta", {}) or {}, device=device)
    return ConversationItem(
        type=item_type,
        value=_materialize_value_spec(
            spec["value"],
            device=device,
            case=case,
            sample_index=sample_index,
            item_index=item_index,
        ),
        role=str(spec.get("role", "user")),
        source=spec.get("source"),
        meta=meta,
    )


# Value and meta materialization ----------------------------------------------


def _materialize_value_spec(
    value: Any,
    *,
    device: torch.device,
    case: ParityCase | None,
    sample_index: int,
    item_index: int,
) -> Any:
    if not isinstance(value, Mapping):
        raise TypeError(f"Conversation item value must be a mapping with kind; got {type(value).__name__}.")
    kind = value.get("kind")
    if kind is None:
        legacy = sorted(key for key in ("tensor", "random", "path") if key in value)
        hint = f" Legacy key(s) are not supported: {legacy}." if legacy else ""
        raise ValueError(f"Conversation item value must declare kind.{hint}")
    kind = str(kind)
    if kind == "tensor":
        return _materialize_tensor_value(value, device=device)
    if kind == "random":
        return _materialize_random_value(
            value, case=case, device=device, sample_index=sample_index, item_index=item_index
        )
    if kind == "text":
        return str(value.get("text", ""))
    if kind == "image":
        return _materialize_image_value(value)
    raise ValueError(
        f"Unsupported conversation item value kind {kind!r}; expected 'tensor', 'random', 'text', or 'image'."
    )


def _materialize_tensor_value(spec: Mapping[str, Any], *, device: torch.device) -> torch.Tensor:
    if "tensor" not in spec:
        raise KeyError("kind: tensor value spec must declare tensor.")
    kwargs: dict[str, Any] = {}
    dtype = _resolve_dtype(spec.get("dtype"))
    if dtype is not None:
        kwargs["dtype"] = dtype
    if bool(spec.get("device", True)):
        kwargs["device"] = device
    return torch.tensor(spec["tensor"], **kwargs)


def _materialize_random_value(
    spec: Mapping[str, Any],
    *,
    case: ParityCase | None,
    device: torch.device,
    sample_index: int,
    item_index: int,
) -> torch.Tensor:
    if "shape" not in spec:
        raise KeyError("kind: random value spec must declare shape.")
    shape = _shape_tuple(spec["shape"])
    distribution = str(spec.get("distribution", "uniform"))
    if distribution not in _RANDOM_DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported random distribution {distribution!r}; expected one of {sorted(_RANDOM_DISTRIBUTIONS)}."
        )
    dtype = _resolve_dtype(spec.get("dtype", "float"))
    if distribution in {"uniform", "normal"} and dtype is not None and not dtype.is_floating_point:
        raise TypeError(f"random distribution {distribution!r} requires a floating point dtype; got {dtype}.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_random_seed(spec, case=case, sample_index=sample_index, item_index=item_index))
    if distribution == "uniform":
        low = float(spec.get("low", 0.0))
        high = float(spec.get("high", 1.0))
        tensor = torch.empty(shape, dtype=dtype).uniform_(low, high, generator=generator)
    elif distribution == "normal":
        mean = float(spec.get("mean", 0.0))
        std = float(spec.get("std", 1.0))
        tensor = torch.empty(shape, dtype=dtype).normal_(mean, std, generator=generator)
    elif distribution == "zeros":
        tensor = torch.zeros(shape, dtype=dtype)
    else:
        tensor = torch.ones(shape, dtype=dtype)
    return tensor.to(device=device if bool(spec.get("device", True)) else torch.device("cpu"))


def _materialize_image_value(spec: Mapping[str, Any]) -> Any:
    width = int(spec.get("width", 64))
    height = int(spec.get("height", width))
    if width <= 0 or height <= 0:
        raise ValueError(f"kind: image width/height must be positive; got {(width, height)}.")
    return make_reference_image(width, height)


def _materialize_meta(value: Any, *, device: torch.device) -> Any:
    if isinstance(value, Mapping):
        if "kind" in value:
            return _materialize_meta_value(value, device=device)
        return {key: _materialize_meta(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize_meta(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_meta(item, device=device) for item in value)
    return value


def _materialize_meta_value(value: Mapping[str, Any], *, device: torch.device) -> Any:
    kind = str(value["kind"])
    if kind == "tensor":
        return _materialize_tensor_value(value, device=device)
    raise ValueError(f"Unsupported meta value kind {kind!r}; expected 'tensor'.")


# Internal helpers -------------------------------------------------------------


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("kind: random shape must be a sequence of integers.")
    shape = tuple(int(dim) for dim in value)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"kind: random shape must contain positive dimensions; got {shape}.")
    return shape


def _validate_sequence(value: Any, name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a YAML sequence.")


def _random_seed(spec: Mapping[str, Any], *, case: ParityCase | None, sample_index: int, item_index: int) -> int:
    if "seed" in spec:
        return int(spec["seed"])
    model_seed = getattr(getattr(case, "model", None), "seed", 0)
    node_id = getattr(case, "node_id", "unknown")
    raw = f"{model_seed}:{node_id}:{sample_index}:{item_index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63)


def _resolve_dtype(value: Any) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    aliases = {
        "long": torch.long,
        "int64": torch.int64,
        "int": torch.int,
        "int32": torch.int32,
        "float": torch.float,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bool": torch.bool,
    }
    try:
        return aliases[str(value)]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor dtype in request spec: {value!r}") from exc


__all__ = [
    "V2RequestContext",
    "conversation_request_from_conversation_list",
    "conversation_stimulus_to_batched_specs",
    "conversation_list_from_specs",
]
