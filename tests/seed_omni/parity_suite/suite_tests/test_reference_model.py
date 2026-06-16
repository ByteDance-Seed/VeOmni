"""Tests for the parity reference model contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.seed_omni.parity_suite.reference.model import ParityReferenceModel, reference_options


class _TinyReference(ParityReferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(flag=False, scale=1)
        self.model = SimpleNamespace(config=SimpleNamespace(flag=False, scale=1))
        self.calls: list[tuple[str, Any]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(("forward", self.config.flag))
        return x + self.config.scale

    def run_reference_custom(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        self.calls.append(("custom", context))
        return {"x": inputs["x"]}


def test_reference_kind_defaults_to_forward() -> None:
    model = _TinyReference()

    result = model.run_reference_kind(None, {"x": torch.tensor(1)}, context=None)

    assert result.item() == 2
    assert model.calls == [("forward", False)]


def test_reference_kind_dispatches_to_method() -> None:
    model = _TinyReference()
    context = object()

    assert model.run_reference_kind("custom", {"x": 3}, context) == {"x": 3}
    assert model.calls == [("custom", context)]


def test_reference_kind_rejects_missing_handler() -> None:
    with pytest.raises(NotImplementedError, match="missing"):
        _TinyReference().run_reference_kind("missing", {}, context=None)


def test_reference_options_restore_both_configs() -> None:
    model = _TinyReference()

    with model.reference_options({"flag": True, "scale": 4}):
        assert model.config.flag is True
        assert model.model.config.flag is True
        assert model(torch.tensor(2)).item() == 6

    assert model.config.flag is False
    assert model.model.config.flag is False
    assert model.config.scale == 1
    assert model.model.config.scale == 1


def test_reference_options_reject_unknown_attrs() -> None:
    with pytest.raises(AttributeError, match="unknown"):
        with reference_options(_TinyReference(), {"unknown": True}):
            pass
