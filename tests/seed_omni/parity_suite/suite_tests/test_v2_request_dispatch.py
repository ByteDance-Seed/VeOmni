"""Tests for V2 request handler dispatch."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from tests.seed_omni.parity_suite.core import (
    ParityCase,
    RecipeSpec,
    RunCaptureOptions,
    RunSpec,
    conversation_stimulus_to_batched_specs,
)
from tests.seed_omni.parity_suite.driver import ParityDriver
from tests.seed_omni.parity_suite.driver.v2_run import V2RunContext, canonical_from_reference_output
from tests.seed_omni.parity_suite.reference.contract import ReferenceRunResult
from tests.seed_omni.parity_suite.v2.request import V2RequestContext
from tests.seed_omni.parity_suite.v2.tier_runners import framework, graph, module


def _recipe(
    *,
    reference: dict[str, Any] | None = None,
    missing_kind: bool = False,
    stimulus: dict[str, Any] | None = None,
) -> RecipeSpec:
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
        stimulus={"prompt": "hello"} if stimulus is None else stimulus,
        reference=resolved_reference,
        runs=(
            RunSpec(id="one_step", tier="graph", kind="graph", probes=()),
            RunSpec(id="one_step", tier="module", kind="module", probes=()),
        ),
    )


def _case(
    *,
    reference: dict[str, Any] | None = None,
    missing_kind: bool = False,
    stimulus: dict[str, Any] | None = None,
    graph_name: str = "infer_toy",
    graph_domain: str = "inference",
) -> ParityCase:
    return ParityCase(
        model=SimpleNamespace(name="toy"),
        recipe=_recipe(reference=reference, missing_kind=missing_kind, stimulus=stimulus),
        run=RunSpec(id="one_step", tier="graph", kind="graph", probes=()),
        graph=SimpleNamespace(name=graph_name, domain=graph_domain),
        nodes=(),
    )


def _handler(ctx: V2RequestContext) -> dict[str, Any]:
    return {"conversation_list": [[{"kind": ctx.kind, "prompt": ctx.stimulus["prompt"]}]]}


def _v2_ctx(
    driver: ParityDriver,
    reference_output: ReferenceRunResult | None = None,
    *,
    device: torch.device | None = None,
) -> V2RunContext:
    device = torch.device("cpu") if device is None else device
    return V2RunContext(
        case=driver.case,
        tier=driver.case.tier,
        domain=driver.case.graph.domain,
        reference_output=reference_output,
        canonical=canonical_from_reference_output(reference_output),
        whitelist={},
        device=device,
        dtype=torch.float32,
        capture_options=RunCaptureOptions(),
    )


def _build_request(driver: ParityDriver, reference_output: ReferenceRunResult | None = None) -> dict[str, Any]:
    return driver.build_v2_request(_v2_ctx(driver, reference_output))


class _ToyDriver(ParityDriver):
    def build_text_und_request(self, ctx: V2RequestContext) -> dict[str, Any]:
        return _handler(ctx)

    def build_infer_toy_request(self, ctx: V2RequestContext) -> dict[str, Any]:
        return _handler(ctx)

    def build_train_forward_backward_request(self, ctx: V2RequestContext) -> dict[str, Any]:
        return _handler(ctx)


def test_reference_kind_infers_default_handler_name() -> None:
    driver = _ToyDriver(_case())
    assert driver._v2_request_method_name("text_und") == "build_text_und_request"
    reference_output = ReferenceRunResult(canonical={"prompt": "x"}, observations={})
    request = _build_request(driver, reference_output)
    assert request["conversation_list"][0][0]["kind"] == "text_und"


def test_missing_reference_kind_defaults_to_graph_name() -> None:
    driver = _ToyDriver(_case(missing_kind=True))
    request = _build_request(driver)
    assert request["conversation_list"][0][0]["kind"] == "infer_toy"


def test_training_reference_kind_defaults_to_forward_backward() -> None:
    driver = _ToyDriver(_case(missing_kind=True, graph_name="train", graph_domain="training"))
    request = _build_request(driver)
    assert request["conversation_list"][0][0]["kind"] == "train_forward_backward"


def test_missing_request_hook_raises_clear_error_without_canonical_fallback() -> None:
    driver = _ToyDriver(_case(reference={"kind": "missing_kind"}))
    reference_output = ReferenceRunResult(
        canonical={
            "conversation_list": [
                [
                    {
                        "type": "text",
                        "role": "assistant",
                        "value": {"kind": "tensor", "tensor": [5, 6], "dtype": "long"},
                        "meta": {"input_ids": {"kind": "tensor", "tensor": [7, 8], "dtype": "long"}},
                    }
                ]
            ]
        },
        observations={},
    )

    with pytest.raises(NotImplementedError, match="no method 'build_missing_kind_request'"):
        driver.build_v2_request(_v2_ctx(driver, reference_output))


def test_inference_conversation_stimulus_materializes_flat_request() -> None:
    driver = _ToyDriver(
        _case(
            reference={"kind": "missing_kind"},
            stimulus={
                "conversation_list": [
                    {
                        "type": "output",
                        "role": "user",
                        "value": {"kind": "tensor", "tensor": [1, 2, 3], "dtype": "long"},
                    },
                    {
                        "type": "text",
                        "role": "user",
                        "source": "recipe",
                        "value": "hello",
                    },
                    {
                        "type": "image",
                        "role": "user",
                        "value": {"kind": "image", "width": 8, "height": 6},
                    },
                ]
            },
        )
    )

    request = _build_request(driver)

    assert len(request["conversation_list"]) == 3
    tensor_item, text_item, image_item = request["conversation_list"]
    assert tensor_item.type == "output"
    assert tensor_item.value.tolist() == [1, 2, 3]
    assert text_item.type == "text"
    assert text_item.role == "user"
    assert text_item.source == "recipe"
    assert text_item.value == "hello"
    assert isinstance(image_item.value, Image.Image)
    assert image_item.value.mode == "RGB"
    assert image_item.value.size == (8, 6)


def test_training_conversation_stimulus_materializes_batched_request() -> None:
    driver = _ToyDriver(
        _case(
            reference={"kind": "missing_kind"},
            graph_name="train",
            graph_domain="training",
            stimulus={
                "conversation_list": [
                    {
                        "type": "text",
                        "role": "assistant",
                        "value": "hello",
                    }
                ]
            },
        )
    )

    request = _build_request(driver)

    assert len(request["conversation_list"]) == 1
    assert request["conversation_list"][0][0].value == "hello"


def test_reference_inputs_keep_batched_oracle_shape() -> None:
    stimulus = {
        "conversation_list": [
            {
                "type": "text",
                "role": "user",
                "value": "hello",
            }
        ]
    }
    driver = _ToyDriver(_case(reference={"kind": "missing_kind"}, stimulus=stimulus))

    inputs = driver.reference_inputs()

    assert inputs["conversation_list"] == [stimulus["conversation_list"]]


def test_v2_request_explicit_batched_stimulus_keeps_batch_shape() -> None:
    driver = _ToyDriver(
        _case(
            reference={"kind": "missing_kind"},
            stimulus={
                "batched_conversation_list": [
                    [
                        {
                            "type": "output",
                            "value": {"kind": "tensor", "tensor": [1], "dtype": "long"},
                        }
                    ],
                    [
                        {
                            "type": "output",
                            "value": {"kind": "tensor", "tensor": [2], "dtype": "long"},
                        }
                    ],
                ]
            },
        )
    )

    request = _build_request(driver)

    assert [sample[0].value.item() for sample in request["conversation_list"]] == [1, 2]


def test_conversation_stimulus_rejects_ambiguous_single_and_batched_keys() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        conversation_stimulus_to_batched_specs({"conversation_list": [], "batched_conversation_list": []})


def test_conversation_value_requires_tagged_kind() -> None:
    driver = _ToyDriver(
        _case(
            reference={"kind": "missing_kind"},
            stimulus={
                "conversation_list": [
                    {
                        "type": "image",
                        "value": {"tensor": [1, 2], "dtype": "long"},
                    }
                ]
            },
        )
    )
    with pytest.raises(ValueError, match="must declare kind"):
        _build_request(driver)


def test_conversation_item_type_is_allowlisted() -> None:
    driver = _ToyDriver(
        _case(
            reference={"kind": "missing_kind"},
            stimulus={
                "conversation_list": [
                    {
                        "type": "latent",
                        "value": {"kind": "tensor", "tensor": [1.0], "dtype": "float"},
                    }
                ]
            },
        )
    )
    with pytest.raises(ValueError, match="Unsupported conversation item type"):
        _build_request(driver)


def test_random_conversation_values_are_deterministic() -> None:
    driver = _ToyDriver(
        _case(
            reference={"kind": "missing_kind"},
            stimulus={
                "conversation_list": [
                    {
                        "type": "image",
                        "value": {
                            "kind": "random",
                            "shape": [3, 4, 4],
                            "distribution": "uniform",
                            "seed": 123,
                            "dtype": "float",
                            "low": -0.5,
                            "high": 0.5,
                        },
                    }
                ]
            },
        )
    )

    first = _build_request(driver)
    second = _build_request(driver)

    first_value = first["conversation_list"][0].value
    second_value = second["conversation_list"][0].value
    assert first_value.shape == (3, 4, 4)
    assert first_value.dtype == torch.float32
    assert torch.equal(first_value, second_value)
    assert torch.all(first_value >= -0.5)
    assert torch.all(first_value <= 0.5)

    default_seed_driver = _ToyDriver(
        _case(
            reference={"kind": "missing_kind"},
            stimulus={
                "conversation_list": [
                    {
                        "type": "image",
                        "value": {"kind": "random", "shape": [1, 2], "distribution": "normal", "dtype": "float"},
                    }
                ]
            },
        )
    )

    first = _build_request(default_seed_driver)
    second = _build_request(default_seed_driver)

    assert torch.equal(first["conversation_list"][0].value, second["conversation_list"][0].value)


def test_random_conversation_value_rejects_invalid_distribution() -> None:
    driver = _ToyDriver(
        _case(
            reference={"kind": "missing_kind"},
            stimulus={
                "conversation_list": [
                    {
                        "type": "image",
                        "value": {"kind": "random", "shape": [1], "distribution": "triangular"},
                    }
                ]
            },
        )
    )
    with pytest.raises(ValueError, match="Unsupported random distribution"):
        _build_request(driver)


def test_linspace_conversation_meta_materializes_tensor_shape_and_transform() -> None:
    driver = _ToyDriver(
        _case(
            reference={"kind": "missing_kind"},
            stimulus={
                "conversation_list": [
                    {
                        "type": "image",
                        "value": {"kind": "image", "width": 2, "height": 2},
                        "meta": {
                            "noise": {
                                "kind": "linspace",
                                "start": -0.25,
                                "end": 0.25,
                                "steps": 6,
                                "shape": [2, 3],
                            },
                            "timestep": {
                                "kind": "linspace",
                                "start": -0.5,
                                "end": 0.5,
                                "steps": 2,
                                "transform": "sigmoid",
                            },
                        },
                    }
                ]
            },
        )
    )

    request = _build_request(driver)
    item = request["conversation_list"][0]

    assert item.meta["noise"].shape == (2, 3)
    assert torch.equal(item.meta["noise"], torch.linspace(-0.25, 0.25, steps=6).reshape(2, 3))
    assert torch.allclose(item.meta["timestep"], torch.sigmoid(torch.tensor([-0.5, 0.5])))


def test_canonical_from_reference_output_rejects_invalid_reference_output() -> None:
    with pytest.raises(TypeError, match="expects ReferenceRunResult"):
        canonical_from_reference_output({"canonical": {"prompt": "x"}})
    with pytest.raises(TypeError, match="expects ReferenceRunResult"):
        canonical_from_reference_output("invalid")


def test_build_v2_request_uses_empty_canonical_without_reference_output() -> None:
    captured: dict[str, Any] = {}

    class _Driver(_ToyDriver):
        def build_text_und_request(self, ctx: V2RequestContext) -> dict[str, Any]:
            captured["canonical"] = ctx.canonical
            captured["reference_output"] = ctx.reference_output
            return {"conversation_list": [[]]}

    driver = _Driver(_case())
    driver.build_v2_request(_v2_ctx(driver, None))
    assert captured["canonical"] == {}
    assert captured["reference_output"] is None


def test_tier_runners_call_build_v2_request() -> None:
    assert "build_v2_request" in inspect.getsource(graph.run_v2_infer_graph)
    assert "build_v2_request" in inspect.getsource(graph.run_v2_train_graph)
    assert "v2_infer_request" not in inspect.getsource(graph.run_v2_infer_graph)
    assert "v2_train_batch_kwargs" not in inspect.getsource(graph.run_v2_train_graph)
    assert "apply_training_cpu_preprocessors" in inspect.getsource(graph._run_v2_train_graph_batch)
    assert "build_v2_request" in inspect.getsource(module.run_v2_infer_module)
    assert "v2_infer_request" not in inspect.getsource(module.run_v2_infer_module)
    source = inspect.getsource(framework.run_v2_train_framework)
    assert "build_v2_request" in source
    assert "v2_train_batch_kwargs" not in source
    assert "apply_training_cpu_preprocessors" in inspect.getsource(framework._run_v2_train_framework_batch)


def test_conversation_request_materializes_nested_meta_tensors() -> None:
    driver = _ToyDriver(
        _case(
            reference={"kind": "missing_kind"},
            stimulus={
                "conversation_list": [
                    {
                        "type": "output",
                        "role": "user",
                        "source": "recipe",
                        "value": {"kind": "tensor", "tensor": [1, 2, 3], "dtype": "long"},
                        "meta": {
                            "position_ids": {"kind": "tensor", "tensor": [0, 1, 2], "dtype": "long"},
                            "literal_flag": True,
                        },
                    }
                ]
            },
        )
    )

    request = _build_request(driver)

    [item] = request["conversation_list"]
    assert item.type == "output"
    assert item.source == "recipe"
    assert item.value.tolist() == [1, 2, 3]
    assert item.meta["position_ids"].tolist() == [0, 1, 2]
    assert item.meta["literal_flag"] is True
