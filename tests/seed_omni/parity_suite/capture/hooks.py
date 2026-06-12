"""Forward hook based capture helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

from tests.seed_omni.parity_suite.capture.common import tensor_to_cpu


@contextmanager
def capture_module_outputs(model: torch.nn.Module, module_names: list[str]) -> Iterator[dict[str, Any]]:
    outputs: dict[str, Any] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    modules = dict(model.named_modules())

    def _hook(name: str):
        def inner(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            outputs[name] = tensor_to_cpu(output)

        return inner

    try:
        for name in module_names:
            if name not in modules:
                raise KeyError(f"Cannot capture unknown module {name!r}.")
            handles.append(modules[name].register_forward_hook(_hook(name)))
        yield outputs
    finally:
        for handle in handles:
            handle.remove()
