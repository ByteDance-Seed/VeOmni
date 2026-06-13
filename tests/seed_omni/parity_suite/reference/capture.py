"""Online reference capture orchestration."""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from torch import nn

from .hooks import HookTap, capture_hook_taps
from .tensors import DEFAULT_MAX_CAPTURE_TENSOR_NUMEL, materialize_reference_value


TapDict = dict[str, list[Any]]
ReferenceFactory = Callable[[], nn.Module]
MemoryProbe = Callable[[], int]
EmptyCacheFn = Callable[[], None]


class ReferenceDriver(Protocol):
    """Protocol implemented by per-case reference drivers."""

    def run_reference(self, ref_model: nn.Module, inputs: Mapping[str, Any], context: ReferenceCaptureContext) -> Any:
        """Run the official reference recipe and return its result."""


@dataclass(frozen=True)
class ExtractorTap:
    """A reference tap extracted from run context instead of a forward hook."""

    name: str
    extractor: Callable[[ReferenceCaptureContext], Any]


@dataclass
class ReferenceCaptureContext:
    """Mutable context shared with driver recipes and extractors."""

    ref_model: nn.Module | None
    inputs: Mapping[str, Any]
    hook_taps: MutableMapping[str, list[Any]]
    extras: dict[str, Any] = field(default_factory=dict)
    output: Any = None

    def record_extra(self, name: str, value: Any) -> None:
        self.extras[name] = value


@dataclass(frozen=True)
class ReferenceCapturePlan:
    hook_taps: tuple[HookTap, ...] = ()
    extractor_taps: tuple[ExtractorTap, ...] = ()


@dataclass(frozen=True)
class ReferenceCaptureResult:
    taps: TapDict
    run_output: Any
    memory_before_release: int
    memory_after_release: int


def capture_reference_taps(
    *,
    reference_factory: ReferenceFactory,
    driver: ReferenceDriver,
    inputs: Mapping[str, Any],
    plan: ReferenceCapturePlan,
    max_tensor_numel: int = DEFAULT_MAX_CAPTURE_TENSOR_NUMEL,
    memory_probe: MemoryProbe | None = None,
    empty_cache_fn: EmptyCacheFn | None = None,
    require_memory_drop: bool | None = None,
) -> ReferenceCaptureResult:
    """Load, observe, run, extract, and release one reference model."""

    ref_model = reference_factory()
    taps: TapDict = {}
    run_output: Any = None
    memory_before_release = 0
    memory_after_release = 0
    try:
        context = ReferenceCaptureContext(ref_model=ref_model, inputs=inputs, hook_taps=taps)
        with capture_hook_taps(ref_model, plan.hook_taps, sink=taps, max_tensor_numel=max_tensor_numel):
            run_output = driver.run_reference(ref_model, inputs, context)
            context.output = run_output
        _capture_extractors(context, plan.extractor_taps, taps=taps, max_tensor_numel=max_tensor_numel)
        memory_before_release = _memory_allocated(memory_probe)
    finally:
        if "context" in locals():
            context.ref_model = None
            context.output = None
        _release_module_storage(ref_model)
        del ref_model
        gc.collect()
        _empty_cache(empty_cache_fn)
        memory_after_release = _memory_allocated(memory_probe)

    should_assert_drop = memory_before_release > 0 if require_memory_drop is None else require_memory_drop
    if should_assert_drop and memory_after_release >= memory_before_release:
        raise AssertionError(
            "Reference model memory was not released before V2 load: "
            f"before={memory_before_release}, after={memory_after_release}."
        )

    return ReferenceCaptureResult(
        taps=taps,
        run_output=run_output,
        memory_before_release=memory_before_release,
        memory_after_release=memory_after_release,
    )


def _capture_extractors(
    context: ReferenceCaptureContext,
    extractor_taps: tuple[ExtractorTap, ...],
    *,
    taps: TapDict,
    max_tensor_numel: int,
) -> None:
    for tap in extractor_taps:
        value = materialize_reference_value(
            tap.extractor(context),
            max_tensor_numel=max_tensor_numel,
            field_path=tap.name,
        )
        values = value if isinstance(value, list) else [value]
        taps.setdefault(tap.name, []).extend(values)


def _memory_allocated(memory_probe: MemoryProbe | None) -> int:
    if memory_probe is not None:
        return int(memory_probe())
    try:
        from veomni.utils.device import get_device_type, get_torch_device

        if get_device_type() == "cpu":
            return 0
        device_api = get_torch_device()
        if not hasattr(device_api, "memory_allocated"):
            return 0
        return int(device_api.memory_allocated())
    except Exception:
        return 0


def _empty_cache(empty_cache_fn: EmptyCacheFn | None) -> None:
    if empty_cache_fn is not None:
        empty_cache_fn()
        return
    try:
        from veomni.utils.device import empty_cache

        empty_cache()
    except Exception:
        return


def _release_module_storage(module: nn.Module) -> None:
    """Drop parameter storage even if a framework object still references the module."""

    try:
        module.to_empty(device="meta")
        return
    except Exception:
        pass
    try:
        module.to(device="cpu")
    except Exception:
        return


__all__ = [
    "ExtractorTap",
    "ReferenceCaptureContext",
    "ReferenceCapturePlan",
    "ReferenceCaptureResult",
    "ReferenceDriver",
    "capture_reference_taps",
]
