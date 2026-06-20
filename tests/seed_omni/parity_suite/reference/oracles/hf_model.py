"""HF/full-model reference oracle backend."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open
from torch import nn
from transformers.initialization import no_init_weights

from tests.seed_omni.parity_suite.core import (
    ParityCase,
    RunCaptureOptions,
    configure_torch_determinism,
    effective_reference_kind,
)
from tests.seed_omni.parity_suite.reference.capture.observation_adapter import (
    NullReferenceObservationAdapter,
    ReferenceObservationAdapter,
)
from tests.seed_omni.parity_suite.reference.capture.runtime import (
    _empty_cache,
    _memory_allocated,
    _release_module_storage,
    capture_extractor_taps,
    capture_hook_taps,
    materialize_reference_observations,
    selected_reference_field_observations,
)
from tests.seed_omni.parity_suite.reference.capture.spec import (
    ReferenceCaptureContext,
    ReferenceCapturePlan,
)
from tests.seed_omni.parity_suite.reference.contract import (
    ReferenceCaptureResult,
    ReferenceOracle,
    ReferenceRunResult,
    ReferenceSubject,
    merge_reference_observations,
    normalize_reference_run_result,
)
from veomni.models.module_utils import init_empty_weights


# Full-model reference subject -------------------------------------------------


class HfModelSubject(ReferenceSubject):
    """Composition subject for full-model reference execution."""

    def __init__(self, case: ParityCase) -> None:
        self.case = case

    @classmethod
    def load_reference_subject(
        cls,
        *,
        case: ParityCase,
        checkpoint: str | Path | None,
        device: torch.device,
        dtype: torch.dtype,
        **kwargs: Any,
    ) -> HfModelSubject:
        del checkpoint, device, dtype, kwargs
        return cls(case=case)

    @property
    def hook_root(self) -> nn.Module:
        raise NotImplementedError(f"{type(self).__name__} must expose a torch.nn.Module hook_root.")

    @property
    def owned_modules(self) -> tuple[nn.Module, ...]:
        return (self.hook_root,)

    def run_reference(
        self,
        kind: str | None,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        options: Mapping[str, Any],
    ) -> ReferenceRunResult:
        if kind is None:
            raise NotImplementedError(f"{type(self).__name__} requires reference.kind.")

        observation_adapter = self.reference_observation_adapter(kind, inputs, context, options)
        with observation_adapter.install(self):
            method = getattr(self, f"run_reference_{kind}", None)
            if method is None:
                raise NotImplementedError(f"{type(self).__name__} does not implement reference kind {kind!r}.")
            result = normalize_reference_run_result(
                _call_reference_method(method, inputs, context, observation_adapter)
            )

        return merge_reference_observations(result, observation_adapter.observations())

    def reference_observation_adapter(
        self,
        kind: str | None,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        options: Mapping[str, Any],
    ) -> ReferenceObservationAdapter:
        del kind, inputs, context, options
        return NullReferenceObservationAdapter()

    @classmethod
    def create_hf_module_subject(cls, request: Any):
        """Create the module-tier facade for ``request``."""

        from tests.seed_omni.parity_suite.reference.oracles.hf_module import HfModuleSubject

        return HfModuleSubject.load_reference_subject(model_subject_cls=cls, request=request)

    @contextmanager
    def reference_options(self, options: Mapping[str, Any] | None) -> Iterator[None]:
        with reference_options(self, options):
            yield


# Full-model oracle ------------------------------------------------------------


@dataclass
class HfModelReferenceOracle(ReferenceOracle):
    """Reference oracle that loads and runs one full-model subject."""

    case: ParityCase
    load_kwargs: Mapping[str, Any] | None = None

    def capture(
        self,
        *,
        inputs: Mapping[str, Any],
        plan: ReferenceCapturePlan,
        device: torch.device,
        dtype: torch.dtype,
        capture_options: RunCaptureOptions,
    ) -> ReferenceCaptureResult:
        memory_before_load = _memory_allocated(None)

        subject = load_hf_model_subject(
            case=self.case,
            device=device,
            dtype=dtype,
            load_kwargs=self.load_kwargs,
        )

        memory_after_release = 0
        try:
            # Execute the official reference path while collecting field, hook,
            # extractor, and subject-owned observation-adapter values.
            payload = self._capture_subject(
                subject,
                inputs=inputs,
                plan=plan,
                capture_options=capture_options,
            )
        finally:
            # Only CPU-materialized observations should survive this point.
            _release_reference_subject(subject)
            del subject
            _empty_cache(None)
            memory_after_release = _memory_allocated(None)

        # Large HF subjects must be released before the V2 side is loaded.
        self._assert_reference_released(
            memory_before_load=memory_before_load,
            memory_before_release=payload.memory_before_release,
            memory_after_release=memory_after_release,
        )

        return ReferenceCaptureResult(
            observations=dict(payload.run_output.observations),
            run_output=payload.run_output,
            memory_before_release=payload.memory_before_release,
            memory_after_release=memory_after_release,
        )

    def _capture_subject(
        self,
        subject: HfModelSubject,
        *,
        inputs: Mapping[str, Any],
        plan: ReferenceCapturePlan,
        capture_options: RunCaptureOptions,
    ) -> _HfModelCapturePayload:
        hook_root = _reference_hook_root(subject)
        hook_observations: dict[str, list[Any]] = {}
        context = ReferenceCaptureContext(
            ref_model=hook_root,
            inputs=inputs,
            hook_taps=hook_observations,
            capture_options=capture_options,
        )
        try:
            run_output = self._run_with_hook_capture(subject, inputs, context, plan=plan)
            run_output = self._with_materialized_observations(
                run_output,
                context=context,
                plan=plan,
                hook_observations=hook_observations,
            )
            return _HfModelCapturePayload(
                run_output=run_output,
                memory_before_release=_memory_allocated(None),
            )
        finally:
            context.ref_model = None
            context.output = None

    def _run_with_hook_capture(
        self,
        subject: HfModelSubject,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        *,
        plan: ReferenceCapturePlan,
    ) -> ReferenceRunResult:
        hook_root = _reference_hook_root(subject)
        with capture_hook_taps(hook_root, plan.hook_taps, context=context):
            configure_torch_determinism(int(getattr(self.case.model, "seed", 1234)))
            raw_output = self._run(subject, inputs, context)
            run_output = normalize_reference_run_result(raw_output)
            context.output = run_output
            return run_output

    def _with_materialized_observations(
        self,
        run_output: ReferenceRunResult,
        *,
        context: ReferenceCaptureContext,
        plan: ReferenceCapturePlan,
        hook_observations: Mapping[str, list[Any]],
    ) -> ReferenceRunResult:
        observations = {
            **materialize_reference_observations(
                selected_reference_field_observations(run_output.observations, plan),
                context=context,
            ),
            **hook_observations,
        }
        capture_extractor_taps(context, plan, observations=observations)
        return ReferenceRunResult(
            canonical=run_output.canonical,
            observations=observations,
            raw_output=run_output.raw_output,
        )

    def _run(
        self,
        subject: HfModelSubject,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
    ) -> ReferenceRunResult:
        reference = self.case.recipe.reference
        kind = effective_reference_kind(self.case)
        options = _merged_options(self.case.model.reference.hf_model.options, reference.get("options", {}))
        with subject.reference_options(options):
            return subject.run_reference(kind, inputs, context, options)

    def _assert_reference_released(
        self,
        *,
        memory_before_load: int,
        memory_before_release: int,
        memory_after_release: int,
    ) -> None:
        if memory_before_release <= memory_before_load or memory_after_release < memory_before_release:
            return
        raise AssertionError(
            "Reference model memory was not released before V2 load: "
            f"before_load={memory_before_load}, before_release={memory_before_release}, "
            f"after_release={memory_after_release}."
        )


@dataclass(frozen=True)
class _HfModelCapturePayload:
    run_output: ReferenceRunResult
    memory_before_release: int


def _call_reference_method(
    method: Any,
    inputs: Mapping[str, Any],
    context: ReferenceCaptureContext,
    observation_adapter: ReferenceObservationAdapter,
) -> Any:
    return method(inputs, context, observation_adapter=observation_adapter)


# Reference option patching ----------------------------------------------------


@contextmanager
def reference_options(model: Any, options: Mapping[str, Any] | None) -> Iterator[None]:
    """Temporarily set config attributes on ``model`` and its wrapped model."""

    if not options:
        with nullcontext():
            yield
        return
    if not isinstance(options, Mapping):
        raise TypeError("reference.options must be a mapping.")

    targets = _config_targets(model)
    if not targets:
        raise AttributeError(f"{type(model).__name__} has no config object for reference options.")

    originals: list[tuple[Any, str, Any]] = []
    try:
        for key, value in options.items():
            if not all(hasattr(target, key) for target in targets):
                raise AttributeError(f"Unknown reference option {key!r} for {type(model).__name__} config.")
            for target in targets:
                originals.append((target, key, getattr(target, key)))
                setattr(target, key, value)
        yield
    finally:
        for target, key, value in reversed(originals):
            setattr(target, key, value)


def _config_targets(model: Any) -> tuple[Any, ...]:
    targets: list[Any] = []
    for candidate in (
        getattr(model, "config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(_reference_hook_root_or_none(model), "config", None),
    ):
        if candidate is None or any(candidate is existing for existing in targets):
            continue
        targets.append(candidate)
    return tuple(targets)


# Reference loading helpers ----------------------------------------------------


@contextmanager
def empty_init_context() -> Iterator[None]:
    with no_init_weights(), init_empty_weights():
        yield


def load_safetensors_weights(
    model: nn.Module,
    weights_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    include_prefixes: tuple[str, ...] | None = None,
) -> None:
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference weights not found: {path}")
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if include_prefixes is None or key.startswith(include_prefixes):
                state_dict[key] = handle.get_tensor(key).to(device=device, dtype=dtype)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if include_prefixes is not None:
        relevant_missing = [key for key in missing if key.startswith(include_prefixes)]
        relevant_unexpected = [key for key in unexpected if key.startswith(include_prefixes)]
    else:
        relevant_missing = list(missing)
        relevant_unexpected = list(unexpected)
    if relevant_missing:
        raise RuntimeError(f"Missing reference weight keys from {path}: {relevant_missing[:20]}")
    if relevant_unexpected:
        raise RuntimeError(f"Unexpected reference weight keys from {path}: {relevant_unexpected[:20]}")


def _merged_options(*options: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in options:
        if item:
            merged.update(dict(item))
    return merged


def load_hf_model_subject(
    *,
    case: ParityCase,
    device: torch.device,
    dtype: torch.dtype,
    load_kwargs: Mapping[str, Any] | None = None,
) -> HfModelSubject:
    """Load the configured full-model reference subject for ``case``."""

    spec = case.model.reference.hf_model
    if spec is None:
        raise ValueError(f"{case.node_id} requires reference.hf_model, but it is not configured.")
    subject_cls = load_hf_model_subject_class(spec.module)
    load_subject = getattr(subject_cls, "load_reference_subject", None)
    if load_subject is None:
        raise TypeError(f"{subject_cls.__name__} must implement load_reference_subject().")
    subject = load_subject(
        case=case,
        checkpoint=spec.checkpoint,
        device=device,
        dtype=dtype,
        **dict(spec.load_kwargs),
        **dict(load_kwargs or {}),
    )
    if not isinstance(subject, HfModelSubject):
        raise TypeError(
            f"reference.hf_model subject loaders must return HfModelSubject; got {type(subject).__name__}."
        )
    return subject


def load_hf_model_subject_class(reference_module: str | None) -> type[Any]:
    """Load the configured full-model reference subject class."""

    if reference_module is None:
        raise ValueError("reference.hf_model.module must declare 'module.path:ClassName'.")
    if ":" not in reference_module:
        raise ValueError("reference.hf_model.module must use 'module.path:ClassName'.")
    module_name, class_name = reference_module.rsplit(":", 1)
    if not module_name or not class_name:
        raise ValueError("reference.hf_model.module must use 'module.path:ClassName'.")

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# Reference cleanup helpers ----------------------------------------------------


def _reference_hook_root(subject: Any) -> nn.Module:
    hook_root = getattr(subject, "hook_root", None)
    if not isinstance(hook_root, nn.Module):
        raise TypeError(f"{type(subject).__name__}.hook_root must be a torch.nn.Module.")
    return hook_root


def _reference_hook_root_or_none(subject: Any) -> nn.Module | None:
    try:
        return _reference_hook_root(subject)
    except TypeError:
        return None


def _release_reference_subject(subject: Any) -> None:
    modules = getattr(subject, "owned_modules", None) or ()
    for module in modules:
        if isinstance(module, nn.Module):
            _release_module_storage(module)


__all__ = [
    "HfModelReferenceOracle",
    "HfModelSubject",
    "empty_init_context",
    "load_hf_model_subject_class",
    "load_hf_model_subject",
    "load_safetensors_weights",
    "reference_options",
]
