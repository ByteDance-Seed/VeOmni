"""Tests for the full-model HF reference oracle backend."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from tests.seed_omni.parity_suite.core import RunCaptureOptions
from tests.seed_omni.parity_suite.reference.capture.observation_adapter import (
    MethodPatchObservationAdapter,
    ReferenceObservationAdapter,
)
from tests.seed_omni.parity_suite.reference.capture.spec import (
    ExtractorTap,
    HookTap,
    ReferenceCaptureContext,
    ReferenceCapturePlan,
)
from tests.seed_omni.parity_suite.reference.contract import ReferenceRunResult
from tests.seed_omni.parity_suite.reference.oracles.hf_model import (
    HfModelReferenceOracle,
    HfModelSubject,
    reference_options,
)


class _TinyReferenceRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.encoder.weight.fill_(2.0)


class _TinyFullSubject(HfModelSubject):
    def __init__(self, case: Any) -> None:
        super().__init__(case)
        self.root = _TinyReferenceRoot()

    @property
    def hook_root(self) -> nn.Module:
        return self.root

    def run_reference_train_forward(
        self,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        *,
        observation_adapter: ReferenceObservationAdapter,
    ) -> ReferenceRunResult:
        del observation_adapter
        hidden = self.root.encoder(inputs["x"])
        context.record_extra("extra", hidden + 1)
        return ReferenceRunResult(
            canonical={"x": inputs["x"]},
            observations={"field_hidden": [hidden + 2]},
            raw_output={"hidden": hidden},
        )


class _ComposedReferenceSubject(HfModelSubject):
    def __init__(self, case: Any) -> None:
        super().__init__(case)
        self.vendor = _TinyReferenceRoot()

    @property
    def hook_root(self) -> nn.Module:
        return self.vendor

    @property
    def owned_modules(self) -> tuple[nn.Module, ...]:
        return (self.vendor,)

    def run_reference(
        self,
        kind: str | None,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        options: Mapping[str, Any],
    ) -> ReferenceRunResult:
        assert kind == "train_forward"
        assert options == {}
        assert context.ref_model is self.vendor
        hidden = self.vendor.encoder(inputs["x"])
        return ReferenceRunResult(
            canonical={"x": inputs["x"]},
            observations={"field_hidden": [hidden + 2]},
            raw_output={"hidden": hidden},
        )


class _MethodPatchAdapter(MethodPatchObservationAdapter):
    def configure(self, subject: Any) -> None:
        def scale_adapter(original: Any):
            def wrapper(value: torch.Tensor) -> torch.Tensor:
                self.record("captured_input", value)
                return original(value) + 1

            return wrapper

        self.patch_method(subject, "scale", scale_adapter)


class _AdapterSubject(_TinyFullSubject):
    def scale(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2

    def reference_observation_adapter(
        self,
        kind: str | None,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        options: Mapping[str, Any],
    ) -> _MethodPatchAdapter:
        del kind, inputs, context, options
        return _MethodPatchAdapter()

    def run_reference_train_forward(
        self,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        *,
        observation_adapter: ReferenceObservationAdapter,
    ) -> ReferenceRunResult:
        del observation_adapter
        hidden = self.scale(inputs["x"])
        return ReferenceRunResult(
            canonical={"x": inputs["x"]},
            observations={"field_hidden": [hidden]},
            raw_output={"hidden": hidden},
        )


def test_hf_model_oracle_combines_fields_hooks_and_extractors(monkeypatch: Any) -> None:
    from tests.seed_omni.parity_suite.reference.oracles import hf_model

    _install_reference_subject(monkeypatch, hf_model, _TinyFullSubject)
    oracle = HfModelReferenceOracle(case=_case())

    result = oracle.capture(
        inputs={"x": torch.tensor([[3.0]])},
        plan=ReferenceCapturePlan(
            hook_taps=(HookTap(name="hook_hidden", module_path="encoder"),),
            extractor_taps=(ExtractorTap(name="extra_hidden", extractor=lambda ctx: ctx.extras["extra"]),),
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
        capture_options=RunCaptureOptions(),
    )

    assert torch.equal(result.observations["field_hidden"][0], torch.tensor([[8.0]]))
    assert torch.equal(result.observations["hook_hidden"][0], torch.tensor([[6.0]]))
    assert torch.equal(result.observations["extra_hidden"][0], torch.tensor([[7.0]]))
    assert result.run_output.canonical["x"].shape == (1, 1)


def test_hf_model_oracle_runs_composed_subject_on_hook_root(monkeypatch: Any) -> None:
    from tests.seed_omni.parity_suite.reference.oracles import hf_model

    _install_reference_subject(monkeypatch, hf_model, _ComposedReferenceSubject)
    oracle = HfModelReferenceOracle(case=_case())

    result = oracle.capture(
        inputs={"x": torch.tensor([[3.0]])},
        plan=ReferenceCapturePlan(hook_taps=(HookTap(name="hook_hidden", module_path="encoder"),)),
        device=torch.device("cpu"),
        dtype=torch.float32,
        capture_options=RunCaptureOptions(),
    )

    assert torch.equal(result.observations["field_hidden"][0], torch.tensor([[8.0]]))
    assert torch.equal(result.observations["hook_hidden"][0], torch.tensor([[6.0]]))


def test_hf_model_subject_merges_capture_observations_and_restores_patch() -> None:
    subject = _AdapterSubject(_case())
    context = ReferenceCaptureContext(ref_model=subject.hook_root, inputs={}, hook_taps={})

    result = subject.run_reference("train_forward", {"x": torch.tensor([[3.0]])}, context, {})

    assert torch.equal(result.observations["captured_input"][0], torch.tensor([[3.0]]))
    assert torch.equal(result.observations["field_hidden"][0], torch.tensor([[7.0]]))
    assert torch.equal(subject.scale(torch.tensor([[3.0]])), torch.tensor([[6.0]]))


def test_reference_options_restore_both_configs() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(flag=False, scale=1),
        model=SimpleNamespace(config=SimpleNamespace(flag=False, scale=1)),
    )

    with reference_options(model, {"flag": True, "scale": 4}):
        assert model.config.flag is True
        assert model.model.config.flag is True
        assert model.config.scale == 4

    assert model.config.flag is False
    assert model.model.config.flag is False
    assert model.config.scale == 1
    assert model.model.config.scale == 1


def test_reference_options_reject_unknown_attrs() -> None:
    model = SimpleNamespace(config=SimpleNamespace(flag=False))

    with pytest.raises(AttributeError, match="unknown"):
        with reference_options(model, {"unknown": True}):
            pass


def _case() -> SimpleNamespace:
    hf_model_spec = SimpleNamespace(
        module="tests.fake:Reference",
        checkpoint=None,
        load_kwargs={"local_files_only": True},
        options={},
    )
    return SimpleNamespace(
        node_id="toy.full.graph.forward",
        model=SimpleNamespace(reference=SimpleNamespace(hf_model=hf_model_spec)),
        recipe=SimpleNamespace(reference={"oracle": "hf_model", "kind": "train_forward"}),
    )


def _install_reference_subject(monkeypatch: Any, hf_model: Any, subject_cls: type[HfModelSubject]) -> None:
    import types

    class Reference(subject_cls):
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
            del checkpoint, device, dtype, kwargs
            return cls(case)

    module = types.ModuleType("tests.fake")
    module.Reference = Reference
    monkeypatch.setattr(hf_model.importlib, "import_module", lambda _: module)
