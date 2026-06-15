"""Tests for reference-side online capture."""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import nn

from tests.seed_omni.parity_suite.reference import (
    ExtractorTap,
    HookTap,
    ReferenceCaptureContext,
    ReferenceCapturePlan,
    capture_hook_taps,
    capture_reference_taps,
)


class _TinyReference(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(2))
            self.head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))


class _TwoStepDriver:
    def run_reference_recipe(
        self,
        ref_model: nn.Module,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
    ) -> dict[str, bool]:
        greedy: list[torch.Tensor] = []
        with torch.no_grad():
            for scale in (1.0, 2.0):
                hidden = ref_model.encoder(inputs["x"] * scale)
                logits = ref_model.head(hidden)
                greedy.append(logits.argmax(dim=-1))
        context.record_extra("greedy", greedy)
        return {"ran": True}


def test_reference_capture_collects_hook_and_extractor_taps_on_cpu() -> None:
    plan = ReferenceCapturePlan(
        hook_taps=(HookTap(name="text.hidden", module_path="encoder"),),
        extractor_taps=(ExtractorTap(name="text.greedy_token", extractor=lambda ctx: ctx.extras["greedy"]),),
    )

    result = capture_reference_taps(
        reference_factory=_TinyReference,
        driver=_TwoStepDriver(),
        inputs={"x": torch.tensor([[1.0, 3.0]])},
        plan=plan,
    )

    assert result.run_output == {"ran": True}
    assert len(result.taps["text.hidden"]) == 2
    assert torch.equal(result.taps["text.hidden"][0], torch.tensor([[1.0, 3.0]]))
    assert result.taps["text.hidden"][0].device.type == "cpu"
    assert len(result.taps["text.greedy_token"]) == 2
    assert torch.equal(result.taps["text.greedy_token"][0], torch.tensor([1]))


def test_reference_capture_releases_reference_and_checks_memory_drop() -> None:
    ref_weak: weakref.ReferenceType[_TinyReference] | None = None
    memory_values = iter([128, 64])
    empty_cache_called = False

    def reference_factory() -> _TinyReference:
        nonlocal ref_weak
        ref = _TinyReference()
        ref_weak = weakref.ref(ref)
        return ref

    def empty_cache() -> None:
        nonlocal empty_cache_called
        empty_cache_called = True

    result = capture_reference_taps(
        reference_factory=reference_factory,
        driver=_TwoStepDriver(),
        inputs={"x": torch.tensor([[1.0, 1.0]])},
        plan=ReferenceCapturePlan(),
        memory_probe=lambda: next(memory_values),
        empty_cache_fn=empty_cache,
        require_memory_drop=True,
    )

    assert result.memory_before_release == 128
    assert result.memory_after_release == 64
    assert empty_cache_called
    assert ref_weak is not None and ref_weak() is None


def test_reference_capture_rejects_large_hook_tap() -> None:
    class LargeDriver:
        def run_reference_recipe(
            self,
            ref_model: nn.Module,
            inputs: Mapping[str, Any],
            context: ReferenceCaptureContext,
        ) -> None:
            del inputs, context
            ref_model.encoder(torch.ones(1, 2))

    with pytest.raises(ValueError, match="exceeding the capture limit"):
        capture_reference_taps(
            reference_factory=_TinyReference,
            driver=LargeDriver(),
            inputs={},
            plan=ReferenceCapturePlan(hook_taps=(HookTap(name="large", module_path="encoder"),)),
            max_tensor_numel=1,
        )


def test_capture_hook_taps_removes_handles_after_context() -> None:
    ref = _TinyReference()
    records: dict[str, list[Any]] = {}

    with capture_hook_taps(ref, [HookTap(name="hidden", module_path="encoder")], sink=records):
        ref.encoder(torch.ones(1, 2))
    ref.encoder(torch.ones(1, 2))

    assert len(records["hidden"]) == 1
