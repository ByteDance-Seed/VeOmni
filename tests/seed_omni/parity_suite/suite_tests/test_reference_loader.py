"""Tests for generic reference oracle loading contracts."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

from tests.seed_omni.parity_suite.reference import model as reference_model
from tests.seed_omni.parity_suite.reference.model import ParityReferenceModel, load_transformers_reference_model


def test_reference_model_registers_auto_classes(monkeypatch: Any) -> None:
    calls: list[tuple[str, Any]] = []

    class Config:
        model_type = "fake"

    class Reference(ParityReferenceModel):
        config_class = Config

    class AutoConfig:
        @staticmethod
        def register(model_type: str, config_class: type[Any], *, exist_ok: bool) -> None:
            calls.append(("config", model_type, config_class, exist_ok))

    class AutoModel:
        @staticmethod
        def register(config_class: type[Any], model_class: type[Any], *, exist_ok: bool) -> None:
            calls.append(("model", config_class, model_class, exist_ok))

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig=AutoConfig, AutoModel=AutoModel))

    Reference.register_auto_model(exist_ok=False)

    assert calls == [
        ("config", "fake", Config, False),
        ("model", Config, Reference, False),
    ]


def test_transformers_loader_registers_reference_class_before_load(monkeypatch: Any) -> None:
    events: list[tuple[str, Any]] = []

    def import_module(name: str) -> types.ModuleType:
        events.append(("import", name))
        module = types.ModuleType(name)

        class Reference:
            @staticmethod
            def register_auto_model() -> None:
                events.append(("register", Reference))

        module.Reference = Reference
        return module

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id: Any, **kwargs: Any) -> dict[str, Any]:
            events.append(("from_pretrained", model_id))
            return {"model_id": model_id}

    monkeypatch.setattr(reference_model.importlib, "import_module", import_module)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModel=AutoModel))

    result = load_transformers_reference_model(module="tests.fake_reference_registration:Reference", checkpoint=None)

    assert result == {"model_id": "tests.fake_reference_registration:Reference"}
    assert events[0] == ("import", "tests.fake_reference_registration")
    assert events[1][0] == "register"
    assert events[2] == ("from_pretrained", "tests.fake_reference_registration:Reference")


def test_transformers_loader_prefers_checkpoint(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append((model_id, kwargs))
            return {"model_id": model_id}

    module = types.ModuleType("tests.fake_reference_registration")
    module.Reference = types.SimpleNamespace(register_auto_model=lambda: calls.append(("register", {})))
    monkeypatch.setitem(sys.modules, "tests.fake_reference_registration", module)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModel=AutoModel))
    checkpoint = tmp_path / "checkpoint"

    result = load_transformers_reference_model(
        module="tests.fake_reference_registration:Reference",
        checkpoint=checkpoint,
        local_files_only=True,
    )

    assert result == {"model_id": checkpoint}
    assert calls == [("register", {}), (checkpoint, {"local_files_only": True})]


def test_transformers_loader_falls_back_to_module(monkeypatch: Any) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append((model_id, kwargs))
            return {"model_id": model_id}

    module = types.ModuleType("tests.fake_reference_registration")
    module.Reference = types.SimpleNamespace(register_auto_model=lambda: None)
    monkeypatch.setitem(sys.modules, "tests.fake_reference_registration", module)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModel=AutoModel))

    result = load_transformers_reference_model(
        module="tests.fake_reference_registration:Reference",
        checkpoint=None,
    )

    assert result == {"model_id": "tests.fake_reference_registration:Reference"}
    assert calls == [("tests.fake_reference_registration:Reference", {})]


def test_transformers_loader_requires_explicit_reference_class() -> None:
    try:
        load_transformers_reference_model(module="tests.fake_reference_registration", checkpoint=None)
    except ValueError as exc:
        assert "module.path:ClassName" in str(exc)
    else:
        raise AssertionError("load_transformers_reference_model should require an explicit reference class.")
