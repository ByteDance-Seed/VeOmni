"""Suite-level V2 request context and conversation request builders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import ParityCase, to_device
from veomni.models.seed_omni.conversation import ConversationItem


@dataclass(frozen=True)
class V2RequestContext:
    """Inputs passed to model-specific V2 request handlers."""

    case: ParityCase
    kind: str
    canonical: Mapping[str, Any]
    stimulus: Mapping[str, Any]
    reference_output: Any | None
    device: torch.device


@dataclass(frozen=True)
class _PathExpr:
    path: str
    dtype: torch.dtype | None
    device: bool
    detach: bool


class ConversationRequestBuilder:
    """Lightweight helpers for building V2 conversation requests from canonical data."""

    def __init__(self, canonical: Mapping[str, Any], *, device: torch.device) -> None:
        self._canonical = canonical
        self._device = device

    def path(
        self, path: str, *, dtype: torch.dtype | None = None, device: bool = True, detach: bool = True
    ) -> _PathExpr:
        return _PathExpr(path=path, dtype=dtype, device=device, detach=detach)

    def literal(self, value: Any) -> Any:
        return value

    def values(self, *items: Any) -> list[Any]:
        return list(items)

    def text(
        self,
        value: Any,
        *,
        role: str = "user",
        meta: Mapping[str, Any] | None = None,
        source: str | None = None,
    ) -> ConversationItem:
        return ConversationItem(
            type="text",
            value=_materialize(value, canonical=self._canonical, device=self._device),
            role=role,
            source=source,
            meta=_materialize(meta or {}, canonical=self._canonical, device=self._device),
        )

    def image(
        self,
        value: Any,
        *,
        role: str = "user",
        meta: Mapping[str, Any] | None = None,
        source: str | None = None,
    ) -> ConversationItem:
        return ConversationItem(
            type="image",
            value=_materialize(value, canonical=self._canonical, device=self._device),
            role=role,
            source=source,
            meta=_materialize(meta or {}, canonical=self._canonical, device=self._device),
        )

    def request(self, *items: Any) -> dict[str, Any]:
        sample = [_materialize(item, canonical=self._canonical, device=self._device) for item in items]
        return {"conversation_list": sample}

    def batched_request(self, *items: Any) -> dict[str, Any]:
        sample = [_materialize(item, canonical=self._canonical, device=self._device) for item in items]
        return {"conversation_list": [sample]}


def _materialize(value: Any, *, canonical: Mapping[str, Any], device: torch.device) -> Any:
    if isinstance(value, _PathExpr):
        return _resolve_path(value, canonical=canonical, device=device)
    if isinstance(value, Mapping):
        return {key: _materialize(item, canonical=canonical, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item, canonical=canonical, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize(item, canonical=canonical, device=device) for item in value)
    return value


def _resolve_path(expr: _PathExpr, *, canonical: Mapping[str, Any], device: torch.device) -> Any:
    value: Any = canonical
    for part in expr.path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Unable to resolve canonical path {expr.path!r}.")
        value = value[part]
    if torch.is_tensor(value):
        if expr.detach:
            value = value.detach()
        if expr.device:
            value = value.to(device=device)
        if expr.dtype is not None:
            value = value.to(dtype=expr.dtype)
        return value
    if isinstance(value, Mapping):
        if expr.device:
            return to_device(value, device)
        return dict(value)
    return value


__all__ = ["ConversationRequestBuilder", "V2RequestContext"]
