"""Forward-hook capture helpers for reference models."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from torch import nn

from .tensors import DEFAULT_MAX_CAPTURE_TENSOR_NUMEL, materialize_reference_value


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


__all__ = ["HookTap", "capture_hook_taps", "resolve_submodule"]
