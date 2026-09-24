"""Unit tests for :mod:`veomni.trainer.omni.omni_inferencer` helpers (no model, no GPU).

The end-to-end launches through ``tasks/omni/*.py`` live in ``test_omni_launch.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from veomni.arguments.omni_arguments_types import OmniModuleRuntimeArguments
from veomni.arguments.parser import _instantiate_recursive
from veomni.trainer.omni.omni_inferencer import (
    InferenceRequest,
    OmniInferencer,
    _extract_generated_text,
    _module_needs_distributed,
)


def _inferencer_with(infer_type: str, generation_kwargs: dict) -> OmniInferencer:
    inferencer = OmniInferencer.__new__(OmniInferencer)
    inferencer.args = SimpleNamespace(
        model=SimpleNamespace(launcher_config=lambda key: infer_type if key == "infer_type" else None),
        infer=SimpleNamespace(generation_kwargs=generation_kwargs),
    )
    return inferencer


def test_inference_request_defaults_are_empty():
    req = InferenceRequest(prompt="hi")
    assert req.images == []
    assert req.generation_kwargs == {}


def test_inference_request_is_a_plain_dataclass():
    # ``OmniInferencer.run_request`` accepts an InferenceRequest; we want it
    # to be JSON-loggable for traceability so the dataclass MUST NOT carry
    # private state (only the documented fields).
    fields = InferenceRequest.__dataclass_fields__
    assert set(fields) == {
        "prompt",
        "images",
        "audios",
        "videos",
        "mm_configs",
        "generation_kwargs",
    }


def test_runtime_generation_kwargs_attach_the_resolved_infer_type_without_mutating_args():
    inferencer = _inferencer_with("infer_gen", {"temperature": 0.5})

    assert inferencer._runtime_generation_kwargs() == {"temperature": 0.5, "infer_type": "infer_gen"}
    assert inferencer.args.infer.generation_kwargs == {"temperature": 0.5}


def test_runtime_generation_kwargs_reject_a_conflicting_infer_type():
    inferencer = _inferencer_with("infer_gen", {"infer_type": "infer_und"})

    with pytest.raises(ValueError, match="conflicts"):
        inferencer._runtime_generation_kwargs()


@pytest.mark.parametrize(("fsdp_mode", "expected"), [("fsdp2", True), ("ddp", True), ("eager", False)])
def test_module_needs_distributed_only_when_not_eager(fsdp_mode, expected):
    module_args = _instantiate_recursive(
        OmniModuleRuntimeArguments,
        {"model_path": "fake_module_a", "accelerator": {"fsdp_config": {"fsdp_mode": fsdp_mode}}},
    )
    assert _module_needs_distributed(module_args) is expected


@pytest.mark.parametrize(("distributed", "expected_device"), [(True, "meta"), (False, "cpu")])
def test_run_places_request_tensors_like_training_when_distributed(distributed, expected_device):
    """The processor builds CPU tensors. Distributed modules all sit on the rank's
    device, so the request moves there as a training batch does; an eager
    ``device_map="auto"`` load leaves placement to each module."""
    seen = {}

    def generate(request, generation_kwargs=None):
        seen["input_ids"] = request["input_ids"]
        return []

    inferencer = _inferencer_with("infer_gen", {})
    inferencer._distributed = distributed
    inferencer.device = torch.device("meta")
    inferencer.processor = lambda **kwargs: {"input_ids": torch.ones(2, dtype=torch.long)}
    inferencer.model = SimpleNamespace(reset=lambda: None, generate=generate)

    inferencer._run(InferenceRequest(prompt="hi"))

    assert seen["input_ids"].device.type == expected_device


def test_extract_generated_text_keeps_only_filled_text_items():
    generated = [
        {"type": "text", "value": "a"},
        {"type": "image", "value": object()},
        {"type": "text", "value": None},
        {"type": "text", "value": "b"},
    ]
    assert _extract_generated_text(generated) == "a\nb"
