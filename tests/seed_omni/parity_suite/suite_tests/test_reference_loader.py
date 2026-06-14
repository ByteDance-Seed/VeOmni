"""Tests for generic reference oracle loading contracts."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from tests.seed_omni.parity_suite.core import ReferenceSpec
from tests.seed_omni.parity_suite.reference.loader import load_reference_model


def test_vendored_loader_uses_functional_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[Path | None, dict[str, Any]]] = []

    def load_vendored_model(checkpoint: Path | None, **kwargs: Any) -> dict[str, Any]:
        calls.append((checkpoint, kwargs))
        return {"loaded": checkpoint}

    module = types.SimpleNamespace(load_vendored_model=load_vendored_model)
    monkeypatch.setitem(sys.modules, "tests.fake_vendored_reference", module)
    checkpoint = tmp_path / "checkpoint"
    spec = ReferenceSpec(loader="vendored", module="tests.fake_vendored_reference", checkpoint=checkpoint)

    result = load_reference_model(spec, device="cpu")

    assert result == {"loaded": checkpoint}
    assert calls == [(checkpoint, {"device": "cpu"})]


def test_vendored_loader_requires_functional_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tests.fake_invalid_reference", types.SimpleNamespace())
    spec = ReferenceSpec(loader="vendored", module="tests.fake_invalid_reference")

    with pytest.raises(ValueError, match="load_vendored_model"):
        load_reference_model(spec)


def test_transformers_loader_prefers_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append((model_id, kwargs))
            return {"model_id": model_id}

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModel=AutoModel))
    checkpoint = tmp_path / "checkpoint"
    spec = ReferenceSpec(loader="transformers", module="owner/model", checkpoint=checkpoint)

    result = load_reference_model(spec, local_files_only=True)

    assert result == {"model_id": checkpoint}
    assert calls == [(checkpoint, {"local_files_only": True})]


def test_transformers_loader_falls_back_to_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append((model_id, kwargs))
            return {"model_id": model_id}

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModel=AutoModel))
    spec = ReferenceSpec(loader="transformers", module="owner/model")

    result = load_reference_model(spec)

    assert result == {"model_id": "owner/model"}
    assert calls == [("owner/model", {})]
