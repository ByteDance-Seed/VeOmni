"""BAGEL module-tier reference facade and lazy loading profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tests.seed_omni.parity_suite.reference.capture.observation_adapter import (
    ReferenceObservationAdapterSatisfied,
    ReferenceObservationAdapterStopPolicy,
    reference_observation_adapter_stop_policy,
)
from tests.seed_omni.parity_suite.reference.capture.spec import ReferenceCaptureContext
from tests.seed_omni.parity_suite.reference.contract import (
    ReferenceRunResult,
    merge_reference_observations,
    normalize_reference_run_result,
)
from tests.seed_omni.parity_suite.reference.oracles.hf_model import _call_reference_method
from tests.seed_omni.parity_suite.reference.oracles.hf_module import HfModuleLoadRequest, HfModuleSubject

from .assembly import BagelAssemblyPlan, load_bagel_reference_bundle


@dataclass(frozen=True)
class BagelModuleProfile:
    """BAGEL-owned lazy module reference profile."""

    assembly_plan: BagelAssemblyPlan
    observation_fields: tuple[str, ...]
    stop_policy: ReferenceObservationAdapterStopPolicy


class BagelModuleSubject(HfModuleSubject):
    """Module-tier facade that delegates execution to official BAGEL inference."""

    def __init__(
        self,
        *,
        model_subject: Any,
        module_name: str,
        load_request: HfModuleLoadRequest | None = None,
        profile: BagelModuleProfile | None = None,
    ) -> None:
        super().__init__(model_subject=model_subject, module_name=module_name, load_request=load_request)
        self.profile = profile

    @classmethod
    def load_reference_subject(
        cls,
        *,
        model_subject_cls: type[Any],
        request: HfModuleLoadRequest,
    ) -> BagelModuleSubject:
        profile = cls._profile_for(request)
        if profile is None:
            return super().load_reference_subject(model_subject_cls=model_subject_cls, request=request)

        checkpoint = request.case.model.reference.hf_model.checkpoint
        if checkpoint is None:
            raise ValueError("BAGEL hf_module reference requires reference.hf_model.checkpoint.")
        bundle = load_bagel_reference_bundle(
            checkpoint=checkpoint,
            plan=profile.assembly_plan,
            device=request.device,
            dtype=request.dtype,
            init_on_meta=True,
        )
        model_subject = model_subject_cls(
            case=request.case,
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            new_token_ids=bundle.new_token_ids,
            vae_model=bundle.vae_model,
            reference_device=bundle.device,
        )
        return cls(
            model_subject=model_subject,
            module_name=request.module_name,
            load_request=request,
            profile=profile,
        )

    @classmethod
    def _profile_for(cls, request: HfModuleLoadRequest) -> BagelModuleProfile | None:
        observation_fields = _module_observation_fields(request)
        if not observation_fields:
            return None
        return BagelModuleProfile(
            assembly_plan=_module_assembly_plan(request.module_name, observation_fields),
            observation_fields=observation_fields,
            stop_policy=ReferenceObservationAdapterStopPolicy(fields=frozenset(observation_fields)),
        )

    def run_reference(self, kind, inputs, context, options):
        if self.profile is None:
            return super().run_reference(kind, inputs, context, options)
        try:
            result = self._run_model_subject_with_stop_policy(kind, inputs, context, options)
        except ReferenceObservationAdapterSatisfied as exc:
            result = ReferenceRunResult(
                canonical={"conversation_list": inputs["conversation_list"]},
                observations=exc.observations,
                raw_output=None,
            )
        return self._filter_reference_observations(result, kind)

    def _run_model_subject_with_stop_policy(
        self,
        kind: str | None,
        inputs,
        context: ReferenceCaptureContext,
        options,
    ) -> ReferenceRunResult:
        if kind is None:
            raise NotImplementedError(f"{type(self).__name__} requires reference.kind.")
        method = getattr(self.model_subject, f"run_reference_{kind}", None)
        if method is None:
            raise NotImplementedError(
                f"{type(self.model_subject).__name__} does not implement reference kind {kind!r}."
            )

        observation_adapter = self.model_subject.reference_observation_adapter(kind, inputs, context, options)
        with reference_observation_adapter_stop_policy(observation_adapter, self.profile.stop_policy):
            with observation_adapter.install(self.model_subject):
                result = normalize_reference_run_result(
                    _call_reference_method(method, inputs, context, observation_adapter)
                )

        return merge_reference_observations(result, observation_adapter.observations())

    def reference_observation_fields(self, kind: str | None) -> tuple[str, ...] | None:
        if self.profile is not None:
            return self.profile.observation_fields
        del kind
        return None

    def canonical_observation_fields(self, kind: str | None) -> tuple[str, ...]:
        del kind
        return ()


_BAGEL_OBSERVATION_ORDER = (
    "prompt_input_ids",
    "vision_embeds",
    "mot_prefill_hidden",
    "timestep",
    "x_t",
    "latent_query",
    "denoise_hidden",
    "velocity",
)
_TEXT_ONLY_FIELDS = frozenset({"prompt_input_ids"})
_VISUAL_UND_FIELDS = frozenset({"vision_embeds"})
_LANGUAGE_MODEL_FIELDS = frozenset({"mot_prefill_hidden", "denoise_hidden"})
_FLOW_FIELDS = frozenset({"latent_query", "velocity"})
_VAE_FIELDS = frozenset({"x_t"})
_GEN_FIELDS = frozenset({"timestep", "x_t", "latent_query", "denoise_hidden", "velocity"})
_FLOW_PATH_FIELDS = frozenset({"latent_query", "denoise_hidden", "velocity"})


def _module_observation_fields(request: HfModuleLoadRequest) -> tuple[str, ...]:
    supported = _supported_fields_for_module(request.module_name)
    if not supported:
        return ()
    fields = request.requested_fields & supported
    return tuple(field for field in _BAGEL_OBSERVATION_ORDER if field in fields)


def _supported_fields_for_module(module_name: str) -> frozenset[str]:
    if module_name == "text_encoder":
        return _TEXT_ONLY_FIELDS
    if module_name == "siglip_navit":
        return _VISUAL_UND_FIELDS
    if module_name == "qwen2_mot":
        return _LANGUAGE_MODEL_FIELDS
    if module_name == "flow_connector":
        return _FLOW_FIELDS | frozenset({"latent_query", "denoise_hidden", "timestep"})
    if module_name == "vae":
        return _VAE_FIELDS
    return frozenset()


def _module_assembly_plan(module_name: str, observation_fields: tuple[str, ...]) -> BagelAssemblyPlan:
    fields = frozenset(observation_fields)
    return BagelAssemblyPlan.lazy(
        visual_und=bool(fields & _VISUAL_UND_FIELDS),
        language_model=bool(fields & _LANGUAGE_MODEL_FIELDS),
        flow=module_name == "flow_connector" or bool(fields & _FLOW_PATH_FIELDS),
        ae=module_name == "vae" or bool(fields & _VAE_FIELDS),
        visual_gen=bool(fields & _GEN_FIELDS),
    )


__all__ = ["BagelModuleProfile", "BagelModuleSubject"]
