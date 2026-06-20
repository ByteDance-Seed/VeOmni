"""Tests for generic hf_model subject loading contracts."""

from __future__ import annotations

import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from tests.seed_omni.parity_suite.reference.oracles import hf_model
from tests.seed_omni.parity_suite.reference.oracles.hf_model import HfModelSubject, load_hf_model_subject


class _ReferenceSubject(HfModelSubject):
    def __init__(self, case: Any) -> None:
        super().__init__(case)
        self.root = nn.Linear(1, 1)

    @property
    def hook_root(self) -> nn.Module:
        return self.root


def test_hf_model_oracle_loads_subject_class(monkeypatch: Any, tmp_path: Path) -> None:
    events: list[tuple[str, Any]] = []
    checkpoint = tmp_path / "checkpoint"

    class Reference(_ReferenceSubject):
        @classmethod
        def load_reference_subject(
            cls,
            *,
            case: Any,
            checkpoint: Path | None,
            device: torch.device,
            dtype: torch.dtype,
            **kwargs: Any,
        ) -> HfModelSubject:
            events.append(("load_subject", case.node_id, checkpoint, device, dtype, kwargs))
            return cls(case)

    def import_module(name: str) -> types.ModuleType:
        events.append(("import", name))
        module = types.ModuleType(name)
        module.Reference = Reference
        return module

    monkeypatch.setattr(hf_model.importlib, "import_module", import_module)

    case = _case(checkpoint=checkpoint, load_kwargs={"local_files_only": True})
    subject = load_hf_model_subject(case=case, device=torch.device("cpu"), dtype=torch.float32)

    assert isinstance(subject, Reference)
    assert events == [
        ("import", "tests.fake_reference_subject"),
        (
            "load_subject",
            "toy.full.graph.forward",
            checkpoint,
            torch.device("cpu"),
            torch.float32,
            {"local_files_only": True},
        ),
    ]


def test_hf_model_oracle_requires_subject_loader(monkeypatch: Any) -> None:
    module = types.ModuleType("tests.fake_reference_subject")
    module.Reference = object
    monkeypatch.setattr(hf_model.importlib, "import_module", lambda _: module)

    with pytest.raises(TypeError, match="load_reference_subject"):
        load_hf_model_subject(case=_case(), device=torch.device("cpu"), dtype=torch.float32)


def test_hf_model_oracle_requires_subject_return(monkeypatch: Any) -> None:
    class Reference:
        @classmethod
        def load_reference_subject(cls, **kwargs: Any) -> object:
            del kwargs
            return object()

    module = types.ModuleType("tests.fake_reference_subject")
    module.Reference = Reference
    monkeypatch.setattr(hf_model.importlib, "import_module", lambda _: module)

    with pytest.raises(TypeError, match="HfModelSubject"):
        load_hf_model_subject(case=_case(), device=torch.device("cpu"), dtype=torch.float32)


def test_hf_model_oracle_requires_explicit_reference_class() -> None:
    with pytest.raises(ValueError, match="module.path:ClassName"):
        load_hf_model_subject(
            case=_case(module="tests.fake_reference_subject"),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def _case(
    *,
    module: str = "tests.fake_reference_subject:Reference",
    checkpoint: Path | None = None,
    load_kwargs: dict[str, Any] | None = None,
) -> SimpleNamespace:
    hf_model_spec = SimpleNamespace(
        module=module,
        checkpoint=checkpoint,
        load_kwargs=load_kwargs or {},
        options={},
    )
    return SimpleNamespace(
        node_id="toy.full.graph.forward",
        model=SimpleNamespace(reference=SimpleNamespace(hf_model=hf_model_spec)),
        recipe=SimpleNamespace(reference={"oracle": "hf_model", "kind": "train_forward"}),
    )
