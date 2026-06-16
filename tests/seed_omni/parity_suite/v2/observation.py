"""V2-side observation helpers for the parity suite."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

from tests.seed_omni.parity_suite.core.utilities import to_cpu
from veomni.models.seed_omni.module import ModuleMixin


ObserverSink = MutableMapping[tuple[str, str], list[dict[str, Any]]]
ModuleObservationSink = MutableMapping[tuple[str, str], list[dict[str, Any]]]


@contextmanager
def arm_generation_observer(
    whitelist: Mapping[tuple[str, str], Iterable[str]],
    *,
    sink: ObserverSink | None = None,
    max_tensor_numel: int = 1_000_000,
) -> Iterator[ObserverSink]:
    """Arm the durable V2 generation observer through ``ModuleMixin``."""

    with ModuleMixin.arm_observer(whitelist, sink=sink, max_tensor_numel=max_tensor_numel) as records:
        yield records


@contextmanager
def capture_forward_outputs(
    modules: Mapping[str, nn.Module],
    targets: Iterable[str],
    *,
    sink: MutableMapping[str, list[Any]] | None = None,
) -> Iterator[MutableMapping[str, list[Any]]]:
    """Capture raw module ``forward`` outputs with test-side hooks."""

    records: MutableMapping[str, list[Any]] = sink if sink is not None else {}
    handles: list[Any] = []

    def _make_hook(name: str):
        def _hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            records.setdefault(name, []).append(output)

        return _hook

    try:
        for name in targets:
            module = modules[name]
            handles.append(module.register_forward_hook(_make_hook(name)))
        yield records
    finally:
        for handle in reversed(handles):
            handle.remove()


def get_training_node_output(forward_result: Mapping[str, Any], node: str, field: str) -> Any:
    """Read a V2 training node field from ``OmniModel.forward`` output."""

    return forward_result["outputs"][node][field]


def get_training_loss(forward_result: Mapping[str, Any], name: str = "loss") -> Any:
    """Read total or named loss values from ``OmniModel.forward`` output."""

    if name == "loss":
        return forward_result["loss"]
    return forward_result["losses"][name]


def record_module_output(
    observations: ModuleObservationSink,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    state: str,
    node: str,
    out: Mapping[str, Any],
    max_tensor_numel: int = 1_000_000,
) -> None:
    """Record whitelisted small tensor fields from a directly invoked V2 node."""

    fields = whitelist.get((state, node))
    if not fields:
        return
    # Module-tier calls return raw node outputs. Keep the same small,
    # whitelisted tensor contract as the durable graph observer.
    record = {
        field: to_cpu(value)
        for field, value in out.items()
        if field in fields and torch.is_tensor(value) and value.numel() <= max_tensor_numel
    }
    if record:
        observations.setdefault((state, node), []).append(record)
