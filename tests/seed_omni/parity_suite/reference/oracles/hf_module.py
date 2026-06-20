"""HF/module reference oracle backend.

Module-tier references reuse the configured full-model HF subject and expose a
module-boundary observation facade. This keeps the oracle on the model's
official inference path instead of configuring per-module standalone clones.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tests.seed_omni.parity_suite.core import ParityCase, effective_reference_kind
from tests.seed_omni.parity_suite.reference.capture import (
    ReferenceCaptureContext,
    ReferenceCapturePlan,
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
    normalize_reference_run_result,
)
from tests.seed_omni.parity_suite.reference.oracles.hf_model import HfModelSubject, load_hf_model_subject_class


# Module reference load request ------------------------------------------------


@dataclass(frozen=True)
class HfModuleLoadRequest:
    """Load request passed to module-tier reference subjects."""

    case: ParityCase
    module_name: str
    kind: str | None
    requested_fields: frozenset[str]
    plan: ReferenceCapturePlan
    device: torch.device
    dtype: torch.dtype
    options: Mapping[str, Any]
    load_kwargs: Mapping[str, Any]


# Module reference subject -----------------------------------------------------


class HfModuleSubject(ReferenceSubject):
    """Module-tier facade backed by a full-model ``HfModelSubject``."""

    def __init__(
        self,
        *,
        model_subject: HfModelSubject,
        module_name: str,
        load_request: HfModuleLoadRequest | None = None,
    ) -> None:
        self.model_subject = model_subject
        self.module_name = module_name
        self.load_request = load_request

    @classmethod
    def load_reference_subject(
        cls,
        *,
        model_subject_cls: type[HfModelSubject],
        request: HfModuleLoadRequest,
    ) -> HfModuleSubject:
        """Load the backing reference subject for a module-tier run.

        The default path intentionally keeps compatibility by loading the full
        model subject. Model-specific module subjects may override this method
        to implement lazy backing-subject assembly.
        """

        spec = request.case.model.reference.hf_model
        model_subject = model_subject_cls.load_reference_subject(
            case=request.case,
            checkpoint=spec.checkpoint,
            device=request.device,
            dtype=request.dtype,
            **dict(request.load_kwargs),
        )
        if not isinstance(model_subject, HfModelSubject):
            raise TypeError(
                "HfModuleSubject.load_reference_subject() backing loader must return HfModelSubject; "
                f"got {type(model_subject).__name__}."
            )
        return cls(model_subject=model_subject, module_name=request.module_name, load_request=request)

    @property
    def hook_root(self) -> nn.Module:
        return self.model_subject.hook_root

    @property
    def owned_modules(self) -> tuple[nn.Module, ...]:
        return self.model_subject.owned_modules

    def run_reference(
        self,
        kind: str | None,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        options: Mapping[str, Any],
    ) -> ReferenceRunResult:
        if kind is None:
            raise NotImplementedError(f"{type(self).__name__} requires reference.kind.")
        method = getattr(self, f"run_reference_{self.module_name}_{kind}", None)
        if method is not None:
            result = method(inputs, context, options)
        else:
            result = self.model_subject.run_reference(kind, inputs, context, options)
        return self._filter_reference_observations(result, kind)

    def reference_observation_fields(self, kind: str | None) -> tuple[str, ...] | None:
        """Return the subject observations exposed for this module run.

        ``None`` preserves all returned subject observations. Model-specific
        facades can override this to keep module-tier references scoped without
        duplicating the execution method for each module.
        """

        return None

    def canonical_observation_fields(self, kind: str | None) -> tuple[str, ...]:
        """Return observation fields that should also be exposed as canonical inputs.

        This is opt-in because most compared observations are outputs only, not
        reusable V2 module-boundary inputs.
        """

        del kind
        return ()

    def _filter_reference_observations(self, result: ReferenceRunResult, kind: str | None) -> ReferenceRunResult:
        fields = self.reference_observation_fields(kind)
        if fields is None:
            return result
        return ReferenceRunResult(
            canonical=result.canonical,
            observations={field: result.observations.get(field, []) for field in fields},
            raw_output=result.raw_output,
        )


# Module oracle ----------------------------------------------------------------


@dataclass
class HfModuleReferenceOracle(ReferenceOracle):
    """Reference oracle that runs a module facade on the full HF subject."""

    case: ParityCase
    name: str

    def capture(
        self,
        *,
        inputs: Mapping[str, Any],
        plan: ReferenceCapturePlan,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ReferenceCaptureResult:
        self._validate_module_name()
        reference = self.case.recipe.reference
        kind = effective_reference_kind(self.case)
        options = _merged_options(self.case.model.reference.hf_model.options, reference.get("options", {}))
        memory_before_load = _memory_allocated(None)
        subject = self._load(device=device, dtype=dtype, plan=plan, kind=kind, options=options)
        hook_root = subject.hook_root
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
                raw_output = self._run(subject, inputs, context, kind=kind, options=options)
            normalized = normalize_reference_run_result(raw_output)
            context.output = normalized
            observations = {
                **_materialize_observations(
                    _selected_field_observations(normalized.observations, plan),
                    max_tensor_numel=max_tensor_numel,
                ),
                **hook_observations,
            }
            _capture_extractors(context, plan, observations=observations, max_tensor_numel=max_tensor_numel)
            canonical_fields = subject.canonical_observation_fields(kind)
            canonical_observations = _materialize_observations(
                {field: normalized.observations.get(field, []) for field in canonical_fields},
                max_tensor_numel=max_tensor_numel,
            )
            run_output = ReferenceRunResult(
                canonical=_canonical_with_observations(
                    normalized.canonical,
                    {**canonical_observations, **observations},
                    fields=canonical_fields,
                ),
                observations=observations,
                raw_output=normalized.raw_output,
            )
            memory_before_release = _memory_allocated(None)
        finally:
            if "context" in locals():
                context.ref_model = None
                context.output = None
            _release_subject_storage(subject)
            del subject
            gc.collect()
            _empty_cache(None)
            memory_after_release = _memory_allocated(None)
        if run_output is None:
            raise RuntimeError(f"HF module reference oracle {self.name!r} did not produce a run output.")
        if memory_before_release > memory_before_load and memory_after_release >= memory_before_release:
            raise AssertionError(
                "Reference module memory was not released before V2 load: "
                f"before_load={memory_before_load}, before_release={memory_before_release}, "
                f"after_release={memory_after_release}."
            )
        return ReferenceCaptureResult(
            observations=dict(run_output.observations),
            run_output=run_output,
            memory_before_release=memory_before_release,
            memory_after_release=memory_after_release,
        )

    def _validate_module_name(self) -> None:
        available = tuple(self.case.model.reference.hf_module)
        if self.name not in available:
            raise KeyError(f"Unknown hf_module reference {self.name!r}; available modules: {sorted(available)}")

    def _load(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        plan: ReferenceCapturePlan,
        kind: str | None,
        options: Mapping[str, Any],
    ) -> HfModuleSubject:
        spec = self.case.model.reference.hf_model
        model_subject_cls = load_hf_model_subject_class(spec.module)
        request = HfModuleLoadRequest(
            case=self.case,
            module_name=self.name,
            kind=kind,
            requested_fields=_requested_fields(plan),
            plan=plan,
            device=device,
            dtype=dtype,
            options=options,
            load_kwargs=dict(spec.load_kwargs),
        )
        subject = model_subject_cls.create_hf_module_subject(request)
        if not isinstance(subject, HfModuleSubject):
            raise TypeError(
                f"HfModelSubject.create_hf_module_subject() must return HfModuleSubject; got {type(subject).__name__}."
            )
        return subject

    def _run(
        self,
        subject: HfModuleSubject,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        *,
        kind: str | None,
        options: Mapping[str, Any],
    ) -> ReferenceRunResult:
        with subject.model_subject.reference_options(options):
            return subject.run_reference(kind, inputs, context, options)


# Reference cleanup helpers ----------------------------------------------------


def _release_subject_storage(subject: HfModuleSubject) -> None:
    release = getattr(subject, "release_reference", None) or getattr(subject.model_subject, "release_reference", None)
    if release is not None:
        release()
        return
    for module in subject.owned_modules:
        if isinstance(module, nn.Module):
            _release_module_storage(module)


# Canonical payload helpers ----------------------------------------------------


def _canonical_with_observations(
    canonical: Mapping[str, Any],
    observations: Mapping[str, list[Any]],
    *,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    values = dict(canonical)
    for field in fields:
        observed = observations.get(field)
        if observed:
            values[field] = observed
    return values


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


# Runtime option helpers -------------------------------------------------------


def _max_tensor_numel(options: Mapping[str, Any] | None) -> int:
    if not options:
        return 1_000_000
    return int(options.get("max_tensor_numel", 1_000_000))


def _run_options(case: Any) -> Mapping[str, Any] | None:
    return getattr(getattr(case, "run", None), "options", None)


def _requested_fields(plan: ReferenceCapturePlan) -> frozenset[str]:
    fields: set[str] = set()
    fields.update(tap.name for tap in plan.field_taps)
    fields.update(tap.name for tap in plan.hook_taps)
    fields.update(tap.name for tap in plan.extractor_taps)
    return frozenset(fields)


def _merged_options(*options: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in options:
        if item:
            merged.update(dict(item))
    return merged


__all__ = ["HfModuleLoadRequest", "HfModuleReferenceOracle", "HfModuleSubject"]
