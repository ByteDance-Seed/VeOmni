"""Plan-driven runtime helpers for reference capture execution.

Unlike subject-owned observation adapters, these helpers are generic suite-side
taps: forward hooks are registered from ``probes.yaml`` and extractor callables
read from the finalized ``ReferenceCaptureContext``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any

from torch import nn

from veomni.models.seed_omni.observer import (
    DEFAULT_MAX_CAPTURE_TENSOR_NUMEL,
    _materialize_observed_value,
)

from .spec import HookTap, ReferenceCaptureContext, ReferenceCapturePlan


MemoryProbe = Callable[[], int]
EmptyCacheFn = Callable[[], None]


# Hook capture -----------------------------------------------------------------


@contextmanager
def capture_hook_taps(
    reference_model: nn.Module,
    taps: Iterable[HookTap],
    *,
    context: ReferenceCaptureContext,
) -> Iterator[MutableMapping[str, list[Any]]]:
    """Capture submodule ``forward`` outputs into ``context.hook_taps``."""

    handles: list[Any] = []

    def _make_hook(tap: HookTap):
        def _hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            context.hook_taps.setdefault(tap.name, []).append(
                materialize_reference_value(
                    output,
                    context=context,
                    field_path=tap.name,
                )
            )

        return _hook

    try:
        for tap in taps:
            module = resolve_submodule(reference_model, tap.module_path)
            handles.append(module.register_forward_hook(_make_hook(tap)))
        yield context.hook_taps
    finally:
        for handle in reversed(handles):
            handle.remove()


def resolve_submodule(root: nn.Module, module_path: str) -> nn.Module:
    """Resolve a dotted submodule path from a reference model."""

    try:
        return root.get_submodule(module_path)
    except AttributeError:
        module: nn.Module = root
        for part in module_path.split("."):
            module = getattr(module, part)
        return module


# Materialization --------------------------------------------------------------


def materialize_reference_value(
    value: Any,
    *,
    context: ReferenceCaptureContext,
    field_path: str = "reference",
) -> Any:
    """Materialize a small reference value as CPU-owned data."""

    return _materialize_observed_value(
        value,
        max_tensor_numel=int(context.capture_options.max_tensor_numel),
        field_path=field_path,
    )


def capture_extractor_taps(
    context: ReferenceCaptureContext,
    plan: ReferenceCapturePlan,
    *,
    observations: dict[str, list[Any]],
) -> None:
    for tap in plan.extractor_taps:
        value = materialize_reference_value(
            tap.extractor(context),
            context=context,
            field_path=tap.name,
        )
        values = value if isinstance(value, list) else [value]
        observations.setdefault(tap.name, []).extend(values)


def materialize_reference_observations(
    observations: Mapping[str, list[Any]],
    *,
    context: ReferenceCaptureContext,
) -> dict[str, list[Any]]:
    materialized: dict[str, list[Any]] = {}
    for name, values in observations.items():
        value_list = values if isinstance(values, list) else [values]
        materialized[str(name)] = [
            materialize_reference_value(
                value,
                context=context,
                field_path=str(name),
            )
            for value in value_list
        ]
    return materialized


def selected_reference_field_observations(
    observations: Mapping[str, list[Any]],
    plan: ReferenceCapturePlan,
) -> Mapping[str, list[Any]]:
    if not plan.field_taps:
        return observations
    selected = {tap.name for tap in plan.field_taps}
    return {name: values for name, values in observations.items() if name in selected}


# Cleanup helpers --------------------------------------------------------------


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
    "capture_extractor_taps",
    "capture_hook_taps",
    "materialize_reference_value",
    "materialize_reference_observations",
    "resolve_submodule",
    "selected_reference_field_observations",
    "_empty_cache",
    "_memory_allocated",
    "_release_module_storage",
]
