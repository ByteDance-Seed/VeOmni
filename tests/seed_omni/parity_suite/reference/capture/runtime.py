"""Runtime helpers for reference capture execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

from torch import nn

from veomni.models.seed_omni.observer import (
    DEFAULT_MAX_CAPTURE_TENSOR_NUMEL,
    _materialize_observed_value,
)

from .spec import HookTap


MemoryProbe = Callable[[], int]
EmptyCacheFn = Callable[[], None]


# Hook capture -----------------------------------------------------------------


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
    max_tensor_numel: int = DEFAULT_MAX_CAPTURE_TENSOR_NUMEL,
    field_path: str = "reference",
) -> Any:
    """Materialize a small reference value as CPU-owned data."""

    return _materialize_observed_value(value, max_tensor_numel=max_tensor_numel, field_path=field_path)


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
    "capture_hook_taps",
    "materialize_reference_value",
    "resolve_submodule",
    "_empty_cache",
    "_memory_allocated",
    "_release_module_storage",
]
