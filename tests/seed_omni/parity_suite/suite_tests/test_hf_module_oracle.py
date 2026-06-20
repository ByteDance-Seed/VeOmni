"""Tests for the module-scoped HF reference oracle backend."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from tests.seed_omni.parity_suite.reference.capture import (
    ExtractorTap,
    HookTap,
    ReferenceCaptureContext,
    ReferenceCapturePlan,
)
from tests.seed_omni.parity_suite.reference.contract import ReferenceRunResult
from tests.seed_omni.parity_suite.reference.oracles import (
    HfModuleReferenceOracle,
)
from tests.seed_omni.parity_suite.reference.oracles.hf_model import HfModelSubject
from tests.seed_omni.parity_suite.reference.oracles.hf_module import HfModuleLoadRequest, HfModuleSubject


EVENTS: list[tuple[str, Any]] = []


class _TinyRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(7.0)


class _TinyModelSubject(HfModelSubject):
    def __init__(self, case: Any) -> None:
        super().__init__(case)
        self.root = _TinyRoot()
        self.config = SimpleNamespace(scale=1)

    @property
    def hook_root(self) -> nn.Module:
        return self.root

    @classmethod
    def create_hf_module_subject(cls, request: HfModuleLoadRequest) -> HfModuleSubject:
        EVENTS.append(("create_module_subject", request.module_name))
        return _TinyModuleSubject.load_reference_subject(model_subject_cls=cls, request=request)


class _TinyModuleSubject(HfModuleSubject):
    @classmethod
    def load_reference_subject(
        cls,
        *,
        model_subject_cls: type[HfModelSubject],
        request: HfModuleLoadRequest,
    ) -> HfModuleSubject:
        EVENTS.append(
            (
                "module_loader",
                request.module_name,
                request.kind,
                sorted(request.requested_fields),
                dict(request.load_kwargs),
            )
        )
        subject = super().load_reference_subject(model_subject_cls=model_subject_cls, request=request)
        return cls(model_subject=subject.model_subject, module_name=request.module_name, load_request=request)

    def run_reference_tiny_encode(
        self,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        options: Mapping[str, Any],
    ) -> ReferenceRunResult:
        EVENTS.append(("run", self.module_name, dict(options), context.ref_model is self.hook_root))
        output = self.model_subject.root.linear(inputs["x"])
        context.record_extra("extra", output + 1)
        return ReferenceRunResult(
            canonical={"x": inputs["x"]},
            observations={"hidden": [output], "scale": [options["scale"]]},
            raw_output={"kind": "encode"},
        )


def test_hf_module_oracle_reuses_hf_model_subject(monkeypatch: Any) -> None:
    _install_reference_subject(monkeypatch)
    EVENTS.clear()
    oracle = HfModuleReferenceOracle(case=_case(), name="tiny")

    result = oracle.capture(
        inputs={"x": torch.tensor([[2.0]])},
        plan=ReferenceCapturePlan(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert result.run_output.canonical["x"].shape == (1, 1)
    assert torch.equal(result.observations["hidden"][0], torch.tensor([[14.0]]))
    assert result.observations["scale"] == [3]
    assert EVENTS == [
        ("create_module_subject", "tiny"),
        ("module_loader", "tiny", "encode", [], {"local_files_only": True}),
        ("loader", None, "cpu", torch.float32, {"local_files_only": True}),
        ("run", "tiny", {"scale": 3}, True),
    ]


def test_hf_module_oracle_captures_hook_and_extractor_taps(monkeypatch: Any) -> None:
    _install_reference_subject(monkeypatch)
    oracle = HfModuleReferenceOracle(case=_case(), name="tiny")

    result = oracle.capture(
        inputs={"x": torch.tensor([[2.0]])},
        plan=ReferenceCapturePlan(
            hook_taps=(HookTap(name="hook_hidden", module_path="linear"),),
            extractor_taps=(ExtractorTap(name="extra_hidden", extractor=lambda ctx: ctx.extras["extra"]),),
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(result.observations["hook_hidden"][0], torch.tensor([[14.0]]))
    assert torch.equal(result.observations["extra_hidden"][0], torch.tensor([[15.0]]))


def _case() -> SimpleNamespace:
    hf_model_spec = SimpleNamespace(
        module="tests.fake:Reference",
        checkpoint=None,
        load_kwargs={"local_files_only": True},
        options={"scale": 3},
    )
    return SimpleNamespace(
        node_id="toy.tiny.module.encode",
        model=SimpleNamespace(reference=SimpleNamespace(hf_model=hf_model_spec, hf_module=("tiny",))),
        recipe=SimpleNamespace(reference={"oracle": "hf_module.tiny", "kind": "encode"}),
    )


def _install_reference_subject(monkeypatch: Any) -> None:
    import types

    from tests.seed_omni.parity_suite.reference.oracles import hf_model

    class Reference(_TinyModelSubject):
        @classmethod
        def load_reference_subject(
            cls,
            *,
            case: Any,
            checkpoint: Any,
            device: torch.device,
            dtype: torch.dtype,
            **kwargs: Any,
        ) -> HfModelSubject:
            EVENTS.append(("loader", checkpoint, device.type, dtype, dict(kwargs)))
            return cls(case)

    module = types.ModuleType("tests.fake")
    module.Reference = Reference
    monkeypatch.setattr(hf_model.importlib, "import_module", lambda _: module)
