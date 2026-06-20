"""Tests for SeedOmni V2 parity observation plumbing."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.seed_omni.parity_suite.core import PARITY_ENABLE_ENV, ProbeMapping, RefTapSpec, v2_probe_values
from tests.seed_omni.parity_suite.v2 import (
    arm_generation_observer,
    capture_forward_outputs,
    record_conversation_output,
)
from veomni.models.seed_omni.conversation import ConversationItem
from veomni.models.seed_omni.generation_graph import GenerationGraph
from veomni.models.seed_omni.module import ModuleMixin


class _ToyMixinModule(ModuleMixin):
    def __init__(self, output: dict[str, object]) -> None:
        self.output = output
        super().__init__()

    def generate(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return dict(self.output)


class _PlainModule:
    def generate(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {"small": torch.tensor([7.0])}


def _single_node_graph() -> GenerationGraph:
    return GenerationGraph(
        {
            "initial": "step",
            "states": {
                "step": {
                    "body": [{"from": "toy", "to": "end"}],
                    "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                }
            },
        }
    )


def test_observer_is_noop_when_gate_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PARITY_ENABLE_ENV, raising=False)
    graph = _single_node_graph()
    module = _ToyMixinModule({"small": torch.tensor([1.0])})
    sink: dict[tuple[str, str], list[dict[str, object]]] = {}

    with arm_generation_observer({("step", "toy.generate"): ["small"]}, sink=sink):
        graph.step({"toy": module}, {})

    assert sink == {}


def test_observer_records_whitelisted_fields_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PARITY_ENABLE_ENV, "1")
    graph = _single_node_graph()
    module = _ToyMixinModule(
        {
            "small": torch.tensor([1.0]),
            "large_unmapped": torch.ones(4),
            "past_key_values": object(),
        }
    )
    sink: dict[tuple[str, str], list[dict[str, object]]] = {}

    with arm_generation_observer({("step", "toy.generate"): ["small"]}, sink=sink):
        graph.step({"toy": module}, {})
        graph.step({"toy": module}, {})

    records = sink[("step", "toy.generate")]
    assert len(records) == 2
    assert set(records[0]) == {"small"}
    assert torch.equal(records[0]["small"], torch.tensor([1.0]))
    assert records[0]["small"].device.type == "cpu"


def test_generation_graph_observe_guard_ignores_non_mixin_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PARITY_ENABLE_ENV, "1")
    graph = _single_node_graph()
    sink: dict[tuple[str, str], list[dict[str, object]]] = {}

    with arm_generation_observer({("step", "toy.generate"): ["small"]}, sink=sink):
        graph.step({"toy": _PlainModule()}, {})

    assert sink == {}


def test_generation_observer_records_conversation_fields_from_test_side_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PARITY_ENABLE_ENV, "1")
    graph = _single_node_graph()
    conversation = [
        [
            ConversationItem(
                type="text",
                role="assistant",
                value=torch.tensor([[1.0, 2.0]]),
                meta={"input_ids": torch.tensor([3, 4])},
            )
        ]
    ]
    module = _ToyMixinModule({"conversation_list": conversation})
    sink: dict[tuple[str, str], list[dict[str, object]]] = {}

    with arm_generation_observer({("step", "toy.generate"): ["value", "input_ids"]}, sink=sink):
        graph.step({"toy": module}, {})

    records = sink[("step", "toy.generate")]
    assert len(records) == 1
    assert torch.equal(records[0]["value"], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(records[0]["input_ids"], torch.tensor([3, 4]))
    assert records[0]["_item_type"] == "text"


def test_whitelisted_large_tensor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PARITY_ENABLE_ENV, "1")
    graph = _single_node_graph()
    module = _ToyMixinModule({"small": torch.ones(2)})

    with arm_generation_observer({("step", "toy.generate"): ["small"]}, max_tensor_numel=1):
        with pytest.raises(ValueError, match="exceeding the capture limit"):
            graph.step({"toy": module}, {})


def test_forward_hook_capture_is_test_side_only() -> None:
    module = nn.Linear(2, 1, bias=False)
    modules = {"linear": module}

    with capture_forward_outputs(modules, ["linear"]) as records:
        module(torch.ones(1, 2))

    assert len(records["linear"]) == 1
    assert records["linear"][0].shape == (1, 1)


def test_conversation_output_records_value_and_meta_fields() -> None:
    observations: dict[tuple[str, str], list[dict[str, object]]] = {}
    conversation = [
        [
            ConversationItem(
                type="text",
                role="assistant",
                value=torch.tensor([[1.0, 2.0]]),
                meta={
                    "input_ids": torch.tensor([3, 4]),
                    "labels": torch.tensor([5, 6]),
                    "ignored": torch.tensor([9]),
                },
            ),
            ConversationItem(
                type="text",
                role="dummy",
                value=torch.tensor([[0.0, 0.0]]),
                meta={"input_ids": torch.tensor([0])},
            ),
        ]
    ]

    record_conversation_output(
        observations,
        {("train", "toy.encode"): frozenset({"value", "input_ids", "labels"})},
        state="train",
        node="toy.encode",
        conversation_list=conversation,
    )

    records = observations[("train", "toy.encode")]
    assert len(records) == 1
    assert torch.equal(records[0]["value"], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(records[0]["input_ids"], torch.tensor([3, 4]))
    assert torch.equal(records[0]["labels"], torch.tensor([5, 6]))
    assert "ignored" not in records[0]


def test_probe_values_can_filter_conversation_item_type() -> None:
    mapping = ProbeMapping(
        node="vision",
        probe="vision.embeds",
        v2_field="value",
        ref_tap=RefTapSpec(kind="field", target="vision_embeds", field="vision_embeds"),
        tol="tensor",
        state="train",
        v2_item_type="image",
    )
    case = type(
        "Case",
        (),
        {"nodes": (type("Node", (), {"name": "vision", "state": "train"})(),)},
    )()
    observations = {
        ("train", "vision"): [
            {"value": torch.tensor([1]), "_item_type": "text"},
            {"value": torch.tensor([2]), "_item_type": "image"},
        ]
    }

    values = v2_probe_values(observations, mapping, case=case)

    assert len(values) == 1
    assert torch.equal(values[0], torch.tensor([2]))
