"""Reference capture specs and capture-plan builders."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from torch import nn


if TYPE_CHECKING:
    from tests.seed_omni.parity_suite.core import RefTapSpec


# Capture specs ----------------------------------------------------------------


@dataclass(frozen=True)
class HookTap:
    """A reference tap captured from a submodule ``forward`` output."""

    name: str
    module_path: str


@dataclass(frozen=True)
class ExtractorTap:
    """A reference tap extracted from run context instead of a forward hook."""

    name: str
    extractor: Callable[[ReferenceCaptureContext], Any]


@dataclass(frozen=True)
class FieldTap:
    """A reference tap read from the subject's returned observations."""

    name: str


@dataclass
class ReferenceCaptureContext:
    """Mutable context shared with reference runners and extractors."""

    ref_model: nn.Module | None
    inputs: Mapping[str, Any]
    hook_taps: MutableMapping[str, list[Any]]
    extras: dict[str, Any] = field(default_factory=dict)
    output: Any = None

    def record_extra(self, name: str, value: Any) -> None:
        self.extras[name] = value


@dataclass(frozen=True)
class ReferenceCapturePlan:
    field_taps: tuple[FieldTap, ...] = ()
    hook_taps: tuple[HookTap, ...] = ()
    extractor_taps: tuple[ExtractorTap, ...] = ()


# Capture plan builders --------------------------------------------------------


def build_reference_capture_plan(
    ref_taps: Iterable[tuple[str, RefTapSpec]],
) -> ReferenceCapturePlan:
    """Convert probe reference declarations into a runtime capture plan."""

    hook_taps: list[HookTap] = []
    extractor_taps: list[ExtractorTap] = []
    field_taps: list[FieldTap] = []
    for name, ref_tap in ref_taps:
        if ref_tap.kind == "field":
            field_taps.append(FieldTap(name=name))
            continue
        if ref_tap.kind == "hook":
            hook_taps.append(HookTap(name=name, module_path=ref_tap.target))
            continue
        if ref_tap.kind == "extractor":
            extractor_taps.append(ExtractorTap(name=name, extractor=_load_extractor(ref_tap.target)))
            continue
        raise ValueError(f"Unsupported ref_tap kind: {ref_tap.kind}")
    return ReferenceCapturePlan(
        field_taps=tuple(field_taps),
        hook_taps=tuple(hook_taps),
        extractor_taps=tuple(extractor_taps),
    )


# Internal helpers -------------------------------------------------------------


def _load_extractor(entrypoint: str) -> Callable[[ReferenceCaptureContext], Any]:
    module_name, symbol_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


__all__ = [
    "ExtractorTap",
    "FieldTap",
    "HookTap",
    "ReferenceCaptureContext",
    "ReferenceCapturePlan",
    "build_reference_capture_plan",
]
