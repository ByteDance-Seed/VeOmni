"""Online reference capture orchestration and hook/tensor helpers."""

from __future__ import annotations

import gc
import importlib
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from torch import nn


if TYPE_CHECKING:
    from tests.seed_omni.parity_suite.core import RefTapSpec

from veomni.models.seed_omni.observer import (
    DEFAULT_MAX_CAPTURE_TENSOR_NUMEL,
    _materialize_observed_value,
)


TapDict = dict[str, list[Any]]
ReferenceFactory = Callable[[], nn.Module]
MemoryProbe = Callable[[], int]
EmptyCacheFn = Callable[[], None]


def materialize_reference_value(
    value: Any,
    *,
    max_tensor_numel: int = DEFAULT_MAX_CAPTURE_TENSOR_NUMEL,
    field_path: str = "reference",
) -> Any:
    """Materialize a small reference value as CPU-owned data."""

    return _materialize_observed_value(value, max_tensor_numel=max_tensor_numel, field_path=field_path)


@dataclass(frozen=True)
class HookTap:
    """A reference tap captured from a submodule ``forward`` output."""

    name: str
    module_path: str


def resolve_submodule(root: nn.Module, module_path: str) -> nn.Module:
    """Resolve a dotted submodule path from a reference model."""

    try:
        return root.get_submodule(module_path)
    except AttributeError:
        module: nn.Module = root
        for part in module_path.split("."):
            module = getattr(module, part)
        return module


@contextmanager
def capture_hook_taps(
    reference_model: nn.Module,
    taps: Iterable[HookTap],
    *,
    sink: MutableMapping[str, list[Any]],
    max_tensor_numel: int = DEFAULT_MAX_CAPTURE_TENSOR_NUMEL,
) -> Iterator[MutableMapping[str, list[Any]]]:
    """Capture submodule ``forward`` outputs into ``sink``."""

    handles: list[Any] = []

    def _make_hook(tap: HookTap):
        def _hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            sink.setdefault(tap.name, []).append(
                materialize_reference_value(
                    output,
                    max_tensor_numel=max_tensor_numel,
                    field_path=tap.name,
                )
            )

        return _hook

    try:
        for tap in taps:
            module = resolve_submodule(reference_model, tap.module_path)
            handles.append(module.register_forward_hook(_make_hook(tap)))
        yield sink
    finally:
        for handle in reversed(handles):
            handle.remove()


class ReferenceDriver(Protocol):
    """Protocol implemented by per-case reference drivers."""

    def run_reference_recipe(
        self,
        ref_model: nn.Module,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
    ) -> Any:
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


def build_reference_capture_plan(
    ref_taps: Iterable[tuple[str, RefTapSpec]],
) -> ReferenceCapturePlan:
    """Convert pure probe ref_tap declarations into a runtime capture plan."""

    hook_taps: list[HookTap] = []
    extractor_taps: list[ExtractorTap] = []
    for name, ref_tap in ref_taps:
        if ref_tap.kind == "hook":
            hook_taps.append(HookTap(name=name, module_path=ref_tap.target))
            continue
        if ref_tap.kind == "extractor":
            extractor_taps.append(ExtractorTap(name=name, extractor=_load_extractor(ref_tap.target)))
            continue
        if ref_tap.kind == "output":
            extractor_taps.append(ExtractorTap(name=name, extractor=_load_output_extractor(ref_tap.target)))
            continue
        raise ValueError(f"Unsupported ref_tap kind: {ref_tap.kind}")
    return ReferenceCapturePlan(hook_taps=tuple(hook_taps), extractor_taps=tuple(extractor_taps))


def _load_extractor(entrypoint: str) -> Callable[[ReferenceCaptureContext], Any]:
    module_name, symbol_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _load_output_extractor(path: str) -> Callable[[ReferenceCaptureContext], Any]:
    parts = tuple(part for part in path.split(".") if part)
    if not parts:
        raise ValueError("ref_tap output path must not be empty.")

    def _extract(context: ReferenceCaptureContext) -> Any:
        value: Any = context.output
        for part in parts:
            if isinstance(value, Mapping):
                value = value[part]
                continue
            if isinstance(value, (list, tuple)):
                value = value[int(part)]
                continue
            value = getattr(value, part)
        return value

    return _extract


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
            run_output = driver.run_reference_recipe(ref_model, inputs, context)
            context.output = run_output
        # Extractors run after hooks are removed so driver-owned canonical output
        # can be captured without extending hook lifetimes.
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

    # Large reference oracles and V2 models usually cannot coexist on one GPU.
    # Treat a failed release as a parity-suite failure before V2 loading starts.
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
    "DEFAULT_MAX_CAPTURE_TENSOR_NUMEL",
    "ExtractorTap",
    "HookTap",
    "ReferenceCaptureContext",
    "ReferenceCapturePlan",
    "ReferenceCaptureResult",
    "ReferenceDriver",
    "build_reference_capture_plan",
    "capture_hook_taps",
    "capture_reference_taps",
    "materialize_reference_value",
    "resolve_submodule",
]
