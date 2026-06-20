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

from tests.seed_omni.parity_suite.core import ParityCase, RunCaptureOptions, effective_reference_kind
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
        capture_options: RunCaptureOptions,
    ) -> ReferenceCaptureResult:
        self._validate_module_name()

        reference = self.case.recipe.reference
        kind = effective_reference_kind(self.case)
        options = _merged_options(self.case.model.reference.hf_model.options, reference.get("options", {}))

        memory_before_load = _memory_allocated(None)
        subject = self._load(device=device, dtype=dtype, plan=plan, kind=kind, options=options)

        memory_after_release = 0
        try:
            # Run the full-model reference facade for this module and collect
            # both generic taps and module-owned observation-adapter values.
            payload = self._capture_subject(
                subject,
                inputs=inputs,
                plan=plan,
                kind=kind,
                options=options,
                capture_options=capture_options,
            )
        finally:
            # The facade may own a full HF model, so release it before V2-side
            # execution starts.
            _release_subject_storage(subject)
            del subject
            gc.collect()
            _empty_cache(None)
            memory_after_release = _memory_allocated(None)

        # Module tier exists partly to avoid co-resident reference/V2 graphs;
        # make that memory boundary explicit.
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

    def _capture_subject(
        self,
        subject: HfModuleSubject,
        *,
        inputs: Mapping[str, Any],
        plan: ReferenceCapturePlan,
        kind: str | None,
        options: Mapping[str, Any],
        capture_options: RunCaptureOptions,
    ) -> _HfModuleCapturePayload:
        hook_observations: dict[str, list[Any]] = {}
        context = ReferenceCaptureContext(
            ref_model=subject.hook_root,
            inputs=inputs,
            hook_taps=hook_observations,
            capture_options=capture_options,
        )
        try:
            run_output = self._run_with_hook_capture(
                subject,
                inputs,
                context,
                plan=plan,
                kind=kind,
                options=options,
            )
            run_output = self._with_materialized_observations(
                subject,
                run_output,
                context=context,
                plan=plan,
                kind=kind,
                hook_observations=hook_observations,
            )
            return _HfModuleCapturePayload(
                run_output=run_output,
                memory_before_release=_memory_allocated(None),
            )
        finally:
            context.ref_model = None
            context.output = None

    def _run_with_hook_capture(
        self,
        subject: HfModuleSubject,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        *,
        plan: ReferenceCapturePlan,
        kind: str | None,
        options: Mapping[str, Any],
    ) -> ReferenceRunResult:
        with capture_hook_taps(subject.hook_root, plan.hook_taps, context=context):
            raw_output = self._run(subject, inputs, context, kind=kind, options=options)
            run_output = normalize_reference_run_result(raw_output)
            context.output = run_output
            return run_output

    def _with_materialized_observations(
        self,
        subject: HfModuleSubject,
        run_output: ReferenceRunResult,
        *,
        context: ReferenceCaptureContext,
        plan: ReferenceCapturePlan,
        kind: str | None,
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
        canonical_fields = subject.canonical_observation_fields(kind)
        canonical_observations = materialize_reference_observations(
            {field: run_output.observations.get(field, []) for field in canonical_fields},
            context=context,
        )
        return ReferenceRunResult(
            canonical=_canonical_with_observations(
                run_output.canonical,
                {**canonical_observations, **observations},
                fields=canonical_fields,
            ),
            observations=observations,
            raw_output=run_output.raw_output,
        )

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
            "Reference module memory was not released before V2 load: "
            f"before_load={memory_before_load}, before_release={memory_before_release}, "
            f"after_release={memory_after_release}."
        )


@dataclass(frozen=True)
class _HfModuleCapturePayload:
    run_output: ReferenceRunResult
    memory_before_release: int


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
