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

from tests.seed_omni.parity_suite.core import ParityCase, configure_torch_determinism, effective_reference_kind
from tests.seed_omni.parity_suite.reference.capture import (
    NullReferenceObservationCapture,
    ReferenceCaptureContext,
    ReferenceCapturePlan,
    ReferenceObservationCapture,
    capture_hook_taps,
    materialize_reference_value,
)
from tests.seed_omni.parity_suite.reference.capture.runtime import (
    _empty_cache,
    _memory_allocated,
    _release_module_storage,
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
        method = getattr(self, f"run_reference_{kind}", None)
        if method is None:
            raise NotImplementedError(f"{type(self).__name__} does not implement reference kind {kind!r}.")
        capture = self.reference_observation_capture(kind, inputs, context, options)
        with capture.install(self):
            result = normalize_reference_run_result(_call_reference_method(method, inputs, context, capture))
        return merge_reference_observations(result, capture.observations())

    def reference_observation_capture(
        self,
        kind: str | None,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        options: Mapping[str, Any],
    ) -> ReferenceObservationCapture:
        del kind, inputs, context, options
        return NullReferenceObservationCapture()

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
    ) -> ReferenceCaptureResult:
        memory_before_load = _memory_allocated(None)
        subject = self._load(device=device, dtype=dtype)
        hook_root = _reference_hook_root(subject)
        hook_observations: dict[str, list[Any]] = {}
        run_output: ReferenceRunResult | None = None
        memory_before_release = 0
        memory_after_release = 0
        max_tensor_numel = _max_tensor_numel(_run_options(self.case))
        try:
            context = ReferenceCaptureContext(ref_model=hook_root, inputs=inputs, hook_taps=hook_observations)
            with capture_hook_taps(
                hook_root,
                plan.hook_taps,
                sink=hook_observations,
                max_tensor_numel=max_tensor_numel,
            ):
                configure_torch_determinism(int(getattr(self.case.model, "seed", 1234)))
                raw_output = self._run(subject, inputs, context)
                run_output = normalize_reference_run_result(raw_output)
                context.output = run_output
            observations = {
                **_materialize_observations(
                    _selected_field_observations(run_output.observations, plan),
                    max_tensor_numel=max_tensor_numel,
                ),
                **hook_observations,
            }
            _capture_extractors(context, plan, observations=observations, max_tensor_numel=max_tensor_numel)
            run_output = ReferenceRunResult(
                canonical=run_output.canonical,
                observations=observations,
                raw_output=run_output.raw_output,
            )
            memory_before_release = _memory_allocated(None)
        finally:
            if "context" in locals():
                context.ref_model = None
                context.output = None
            _release_reference_subject(subject)
            del subject
            _empty_cache(None)
            memory_after_release = _memory_allocated(None)

        if run_output is None:
            raise RuntimeError("HF model reference oracle did not produce a run output.")
        if memory_before_release > memory_before_load and memory_after_release >= memory_before_release:
            raise AssertionError(
                "Reference model memory was not released before V2 load: "
                f"before_load={memory_before_load}, before_release={memory_before_release}, "
                f"after_release={memory_after_release}."
            )
        return ReferenceCaptureResult(
            observations=dict(run_output.observations),
            run_output=run_output,
            memory_before_release=memory_before_release,
            memory_after_release=memory_after_release,
        )

    def _load(self, *, device: torch.device, dtype: torch.dtype) -> HfModelSubject:
        return load_hf_model_subject(
            case=self.case,
            device=device,
            dtype=dtype,
            load_kwargs=self.load_kwargs,
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


# Capture materialization helpers ---------------------------------------------


def _capture_extractors(
    context: ReferenceCaptureContext,
    plan: ReferenceCapturePlan,
    *,
    observations: dict[str, list[Any]],
    max_tensor_numel: int,
) -> None:
    for tap in plan.extractor_taps:
        value = materialize_reference_value(
            tap.extractor(context),
            max_tensor_numel=max_tensor_numel,
            field_path=tap.name,
        )
        values = value if isinstance(value, list) else [value]
        observations.setdefault(tap.name, []).extend(values)


def _call_reference_method(
    method: Any,
    inputs: Mapping[str, Any],
    context: ReferenceCaptureContext,
    capture: ReferenceObservationCapture,
) -> Any:
    return method(inputs, context, capture=capture)


def _materialize_observations(
    observations: Mapping[str, list[Any]],
    *,
    max_tensor_numel: int,
) -> dict[str, list[Any]]:
    materialized: dict[str, list[Any]] = {}
    for name, values in observations.items():
        value_list = values if isinstance(values, list) else [values]
        materialized[str(name)] = [
            materialize_reference_value(
                value,
                max_tensor_numel=max_tensor_numel,
                field_path=str(name),
            )
            for value in value_list
        ]
    return materialized


def _selected_field_observations(
    observations: Mapping[str, list[Any]],
    plan: ReferenceCapturePlan,
) -> Mapping[str, list[Any]]:
    if not plan.field_taps:
        return observations
    selected = {tap.name for tap in plan.field_taps}
    return {name: values for name, values in observations.items() if name in selected}


def _max_tensor_numel(options: Mapping[str, Any] | None) -> int:
    if not options:
        return 1_000_000
    return int(options.get("max_tensor_numel", 1_000_000))


def _run_options(case: Any) -> Mapping[str, Any] | None:
    return getattr(getattr(case, "run", None), "options", None)


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
    subject_cls = _load_reference_class(spec.module)
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

    return _load_reference_class(reference_module)


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


def _load_reference_class(reference_module: str | None) -> type[Any]:
    if reference_module is None:
        raise ValueError("reference.hf_model.module must declare 'module.path:ClassName'.")
    if ":" not in reference_module:
        raise ValueError("reference.hf_model.module must use 'module.path:ClassName'.")
    module_name, class_name = reference_module.rsplit(":", 1)
    if not module_name or not class_name:
        raise ValueError("reference.hf_model.module must use 'module.path:ClassName'.")

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


__all__ = [
    "HfModelReferenceOracle",
    "HfModelSubject",
    "empty_init_context",
    "load_hf_model_subject_class",
    "load_hf_model_subject",
    "load_safetensors_weights",
    "reference_options",
]
