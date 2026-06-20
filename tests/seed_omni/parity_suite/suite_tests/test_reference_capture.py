"""Tests for reference-side hook capture helpers."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from tests.seed_omni.parity_suite.core import RunCaptureOptions
from tests.seed_omni.parity_suite.reference.capture.runtime import capture_hook_taps
from tests.seed_omni.parity_suite.reference.capture.spec import HookTap, ReferenceCaptureContext


class _TinyReference(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(2))


def test_capture_hook_taps_materializes_outputs_on_cpu() -> None:
    ref = _TinyReference()
    records: dict[str, list[Any]] = {}
    context = ReferenceCaptureContext(ref_model=ref, inputs={}, hook_taps=records)

    with capture_hook_taps(ref, [HookTap(name="hidden", module_path="encoder")], context=context):
        ref.encoder(torch.tensor([[1.0, 2.0]]))

    assert len(records["hidden"]) == 1
    assert torch.equal(records["hidden"][0], torch.tensor([[1.0, 2.0]]))
    assert records["hidden"][0].device.type == "cpu"


def test_capture_hook_taps_removes_handles_after_context() -> None:
    ref = _TinyReference()
    records: dict[str, list[Any]] = {}
    context = ReferenceCaptureContext(ref_model=ref, inputs={}, hook_taps=records)

    with capture_hook_taps(ref, [HookTap(name="hidden", module_path="encoder")], context=context):
        ref.encoder(torch.ones(1, 2))
    ref.encoder(torch.ones(1, 2))

    assert len(records["hidden"]) == 1


def test_capture_hook_taps_rejects_large_outputs() -> None:
    ref = _TinyReference()
    records: dict[str, list[Any]] = {}
    context = ReferenceCaptureContext(
        ref_model=ref,
        inputs={},
        hook_taps=records,
        capture_options=RunCaptureOptions(max_tensor_numel=1),
    )

    with pytest.raises(ValueError, match="exceeding the capture limit"):
        with capture_hook_taps(ref, [HookTap(name="large", module_path="encoder")], context=context):
            ref.encoder(torch.ones(1, 2))
