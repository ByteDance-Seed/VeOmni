"""V2-side observation helpers for the parity suite."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

from tests.seed_omni.parity_suite.core import PARITY_ENABLE_ENV, to_cpu
from veomni.models.seed_omni.graphs.generation_graph import FSM_SIGNAL_KEY
from veomni.models.seed_omni.mixins.modulemixin import ModuleMixin
from veomni.models.seed_omni.observer import _materialize_observed_value
from veomni.models.seed_omni.utils.conversation import ConversationItem, is_dummy


ObserverSink = MutableMapping[tuple[str, str], list[dict[str, Any]]]
ModuleObservationSink = MutableMapping[tuple[str, str], list[dict[str, Any]]]

LOSS_FIELD = "_loss"


@contextmanager
def arm_generation_observer(
    whitelist: Mapping[tuple[str, str], Iterable[str]],
    *,
    sink: ObserverSink | None = None,
    max_tensor_numel: int = 1_000_000,
) -> Iterator[ObserverSink]:
    """Arm V2 generation observation for parity-suite probes."""

    normalized = _normalize_observation_whitelist(whitelist)
    records: ObserverSink = sink if sink is not None else {}
    if os.environ.get(PARITY_ENABLE_ENV) != "1":
        yield records
        return

    original_observe = ModuleMixin.observe

    def observe_with_parity_capture(
        self: ModuleMixin,
        state: str,
        node: str,
        out: Mapping[str, Any],
    ) -> None:
        del self
        if not isinstance(out, Mapping):
            return
        record_node_output(
            records,
            normalized,
            state=state,
            node=node,
            out=out,
            max_tensor_numel=max_tensor_numel,
        )
        conversation_list = out.get("conversation_list")
        if conversation_list is not None:
            record_conversation_output(
                records,
                normalized,
                state=state,
                node=node,
                conversation_list=conversation_list,
                fsm_signal=out.get(FSM_SIGNAL_KEY),
                max_tensor_numel=max_tensor_numel,
            )

    ModuleMixin.observe = observe_with_parity_capture
    try:
        yield records
    finally:
        ModuleMixin.observe = original_observe


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


def record_node_output(
    observations: ModuleObservationSink,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    state: str,
    node: str,
    out: Mapping[str, Any],
    max_tensor_numel: int = 1_000_000,
) -> None:
    """Record whitelisted top-level node fields for parity observations."""

    fields = whitelist.get((state, node))
    if not fields:
        return
    record: dict[str, Any] = {}
    for field in fields:
        if field not in out or field == "conversation_list":
            continue
        record[field] = _materialize_observed_value(
            out[field],
            max_tensor_numel=max_tensor_numel,
            field_path=f"{state}:{node}.{field}",
        )
    if record:
        observations.setdefault((state, node), []).append(record)


def record_conversation_output(
    observations: ModuleObservationSink,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    state: str,
    node: str,
    conversation_list: Any,
    fsm_signal: str | None = None,
    max_tensor_numel: int = 1_000_000,
) -> None:
    """Record whitelisted tensor fields from a conversation-list carrier.

    This is the default observation path for conversation-contract nodes:
    ``field: value`` maps to ``item.value`` and other fields map to
    ``item.meta[field]``. Extend this helper before adding per-model driver
    hooks for ordinary carrier fields.
    """

    fields = whitelist.get((state, node))
    if not fields:
        return
    for item in _iter_conversation_items(conversation_list):
        if is_dummy(item):
            continue
        record: dict[str, Any] = {}
        if "value" in fields and torch.is_tensor(item.value) and item.value.numel() <= max_tensor_numel:
            record["value"] = to_cpu(item.value)
        for field in fields:
            if field == "value":
                continue
            value = item.meta.get(field)
            if torch.is_tensor(value) and value.numel() <= max_tensor_numel:
                record[field] = to_cpu(value)
        if record:
            record["_item_type"] = item.type
            record["_item_source"] = item.source
            if fsm_signal is not None:
                record["_fsm_signal"] = fsm_signal
            observations.setdefault((state, node), []).append(record)


def _normalize_observation_whitelist(
    whitelist: Mapping[tuple[str, str], Iterable[str]],
) -> dict[tuple[str, str], frozenset[str]]:
    return {
        (str(state), str(node)): frozenset(str(field) for field in fields)
        for (state, node), fields in whitelist.items()
    }


def _iter_conversation_items(conversation_list: Any) -> Iterator[ConversationItem]:
    if isinstance(conversation_list, ConversationItem):
        yield conversation_list
        return
    if isinstance(conversation_list, list):
        for entry in conversation_list:
            if isinstance(entry, ConversationItem):
                yield entry
            elif isinstance(entry, list):
                for item in entry:
                    if isinstance(item, ConversationItem):
                        yield item
