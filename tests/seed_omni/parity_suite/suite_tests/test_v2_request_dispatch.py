"""Tests for V2 request handler dispatch."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.seed_omni.parity_suite.core import ParityCase, RecipeSpec, RunSpec
from tests.seed_omni.parity_suite.driver import ParityDriver
from tests.seed_omni.parity_suite.reference.contract import make_reference_run_output
from tests.seed_omni.parity_suite.v2 import tier_runners
from tests.seed_omni.parity_suite.v2.request import ConversationRequestBuilder, V2RequestContext
from tests.seed_omni.parity_suite.v2.tier_runners import graph, module


def _recipe(*, reference: dict[str, Any] | None = None, missing_kind: bool = False) -> RecipeSpec:
    resolved_reference: dict[str, Any]
    if missing_kind:
        resolved_reference = {}
    elif reference is not None:
        resolved_reference = dict(reference)
    else:
        resolved_reference = {"kind": "text_und"}
    return RecipeSpec(
        id="toy",
        graph="infer_toy",
        stimulus={"prompt": "hello"},
        reference=resolved_reference,
        runs=(
            RunSpec(id="one_step", tier="graph", kind="graph", probes=()),
            RunSpec(id="one_step", tier="module", kind="module", probes=()),
        ),
    )


def _case(*, reference: dict[str, Any] | None = None, missing_kind: bool = False) -> ParityCase:
    return ParityCase(
        model=SimpleNamespace(name="toy"),
        recipe=_recipe(reference=reference, missing_kind=missing_kind),
        run=RunSpec(id="one_step", tier="graph", kind="graph", probes=()),
        graph=SimpleNamespace(name="infer_toy", domain="inference"),
        nodes=(),
    )


def _handler(ctx: V2RequestContext) -> dict[str, Any]:
    return {"conversation_list": [[{"kind": ctx.kind, "prompt": ctx.stimulus["prompt"]}]]}


class _ToyDriver(ParityDriver):
    def build_text_und_request(self, ctx: V2RequestContext) -> dict[str, Any]:
        return _handler(ctx)


def test_reference_kind_infers_default_handler_name() -> None:
    driver = _ToyDriver(_case())
    assert driver._v2_request_method_name("text_und") == "build_text_und_request"
    reference_output = make_reference_run_output({"prompt": "x"}, {})
    request = driver.v2_request_kwargs(reference_output, device=torch.device("cpu"))
    assert request["conversation_list"][0][0]["kind"] == "text_und"


def test_missing_reference_kind_raises_clear_error() -> None:
    driver = _ToyDriver(_case(missing_kind=True))
    with pytest.raises(ValueError, match="must declare reference.kind"):
        driver.v2_request_kwargs(make_reference_run_output({}, {}), device=torch.device("cpu"))


def test_missing_request_hook_raises_clear_error() -> None:
    driver = _ToyDriver(_case(reference={"kind": "missing_kind"}))
    with pytest.raises(NotImplementedError, match="no method 'build_missing_kind_request'"):
        driver.v2_request_kwargs(make_reference_run_output({}, {}), device=torch.device("cpu"))


def test_v2_request_kwargs_rejects_invalid_reference_output() -> None:
    driver = _ToyDriver(_case())
    with pytest.raises(TypeError, match='missing required key "reference"'):
        driver.v2_request_kwargs({"canonical": {"prompt": "x"}}, device=torch.device("cpu"))
    with pytest.raises(TypeError, match='missing required key "canonical"'):
        driver.v2_request_kwargs({"prompt": "x"}, device=torch.device("cpu"))
    with pytest.raises(TypeError, match='must be a mapping shaped as {"canonical": ..., "reference": ...}'):
        driver.v2_request_kwargs("invalid", device=torch.device("cpu"))


def test_v2_request_kwargs_uses_empty_canonical_without_reference_output() -> None:
    captured: dict[str, Any] = {}

    class _Driver(_ToyDriver):
        def build_text_und_request(self, ctx: V2RequestContext) -> dict[str, Any]:
            captured["canonical"] = ctx.canonical
            captured["reference_output"] = ctx.reference_output
            return {"conversation_list": [[]]}

    driver = _Driver(_case())
    driver.v2_request_kwargs(None, device=torch.device("cpu"))
    assert captured["canonical"] == {}
    assert captured["reference_output"] is None


def test_graph_tier_runners_call_v2_request_kwargs() -> None:
    assert "v2_request_kwargs" in inspect.getsource(graph.run_v2_infer_graph)
    assert "v2_request_kwargs" in inspect.getsource(graph.run_v2_train_graph)
    assert "v2_infer_request" not in inspect.getsource(graph.run_v2_infer_graph)
    assert "v2_train_batch_kwargs" not in inspect.getsource(graph.run_v2_train_graph)


def test_module_tier_runners_call_v2_request_kwargs() -> None:
    assert "v2_request_kwargs" in inspect.getsource(module.run_v2_infer_module)
    assert "v2_request_kwargs" in inspect.getsource(module.run_v2_train_module)
    assert "v2_infer_request" not in inspect.getsource(module.run_v2_infer_module)
    assert "v2_train_batch_kwargs" not in inspect.getsource(module.run_v2_train_module)


def test_framework_tier_uses_v2_request_kwargs() -> None:
    source = inspect.getsource(tier_runners.framework.run_v2_train_framework)
    assert "v2_request_kwargs" in source
    assert "v2_train_batch_kwargs" not in source


def test_conversation_request_builder_materializes_nested_paths() -> None:
    canonical = {
        "prompt_input": {
            "packed_text_ids": torch.tensor([1, 2, 3]),
            "packed_text_position_ids": torch.tensor([0, 1, 2]),
        }
    }
    builder = ConversationRequestBuilder(canonical, device=torch.device("cpu"))
    item = builder.text(
        builder.path("prompt_input.packed_text_ids", dtype=torch.long),
        meta={
            "position_ids": builder.path("prompt_input.packed_text_position_ids", dtype=torch.long),
            "literal_flag": builder.literal(True),
        },
        source="oracle",
    )
    assert item.type == "text"
    assert item.value.tolist() == [1, 2, 3]
    assert item.meta["position_ids"].tolist() == [0, 1, 2]
    assert item.meta["literal_flag"] is True

    request = builder.request(item)
    assert request == {"conversation_list": [item]}

    batched_request = builder.batched_request(item)
    assert batched_request == {"conversation_list": [[item]]}
