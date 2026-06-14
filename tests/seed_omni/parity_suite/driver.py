"""Base driver contract for model-specific parity execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from tests.seed_omni.parity_suite.core import (
    ParityCase,
    ParityReport,
    ProbeMapping,
    configure_torch_determinism,
    to_device,
)
from tests.seed_omni.parity_suite.core.utilities import sample_named_grad
from tests.seed_omni.parity_suite.reference.capture import (
    ReferenceCaptureContext,
)
from tests.seed_omni.parity_suite.v2.module import InferModulePolicy
from tests.seed_omni.parity_suite.v2.observation import record_module_output
from veomni.models.seed_omni.modeling_omni import OmniModel


class ParityDriver(ABC):
    """Model-specific execution contract used by the shared parity runner."""

    generation_defaults: Mapping[str, Any] = {
        "max_new_tokens": 1,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    }

    def __init__(self, case: ParityCase) -> None:
        self.case = case

    def dtype(self) -> torch.dtype:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def configure_determinism(self, seed: int) -> None:
        configure_torch_determinism(seed)

    def reference_inputs(self) -> Mapping[str, Any]:
        return self.case.scenario.stimulus

    def generation_kwargs(self, model_or_config: Any) -> dict[str, Any]:
        config = getattr(model_or_config, "config", model_or_config)
        kwargs = dict(getattr(config, "generation_kwargs", None) or {})
        for key, default in self.generation_defaults.items():
            kwargs[key] = self.case.scenario.stimulus.get(key, default)
        return kwargs

    @abstractmethod
    def load_reference(self, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        """Load the independent reference oracle."""

    @abstractmethod
    def load_v2_model(self, *, device: torch.device, dtype: torch.dtype) -> OmniModel:
        """Load the V2 model under test."""

    @abstractmethod
    def run_reference(
        self,
        ref_model: nn.Module,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
    ) -> Any:
        """Run the reference recipe and return driver-owned canonical output."""

    def run_v2_infer_graph(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 inference graph through the shared graph dispatcher."""

        from tests.seed_omni.parity_suite.v2.graph import run_v2_infer_graph

        return run_v2_infer_graph(self, reference_output, whitelist, device=device, dtype=dtype)

    def run_v2_train_graph(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 training graph through the shared graph dispatcher."""

        from tests.seed_omni.parity_suite.v2.graph import run_v2_train_graph

        return run_v2_train_graph(self, reference_output, whitelist, device=device, dtype=dtype)

    def run_v2_infer_module(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 inference graph through the shared module dispatcher."""

        from tests.seed_omni.parity_suite.v2.module import run_v2_infer_module

        return run_v2_infer_module(self, reference_output, whitelist, device=device, dtype=dtype)

    def run_v2_train_module(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 training graph through the shared module dispatcher."""

        from tests.seed_omni.parity_suite.v2.module import run_v2_train_module

        return run_v2_train_module(self, reference_output, whitelist, device=device, dtype=dtype)

    def run_v2_infer_framework(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any] | ParityReport:
        """Run an inference framework-tier case through the shared framework dispatcher."""

        from tests.seed_omni.parity_suite.v2.framework import run_v2_infer_framework

        return run_v2_infer_framework(self, reference_output, whitelist, device=device, dtype=dtype)

    def run_v2_train_framework(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any] | ParityReport:
        """Run a training framework-tier case through the shared framework dispatcher."""

        from tests.seed_omni.parity_suite.v2.framework import run_v2_train_framework

        return run_v2_train_framework(self, reference_output, whitelist, device=device, dtype=dtype)

    def sample_v2_framework_parameters(
        self,
        model: OmniModel,
        batch: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        del model, batch
        return {}

    def v2_infer_request(self, reference_output: Any, *, device: torch.device) -> dict[str, Any]:
        """Adapt driver-owned canonical output into an ``OmniModel.generate`` request."""

        del reference_output, device
        raise NotImplementedError(f"{type(self).__name__} does not implement inference request adaptation.")

    def v2_train_batch_kwargs(self, reference_output: Any, *, device: torch.device) -> dict[str, Any]:
        """Adapt driver-owned canonical output into kwargs for ``OmniModel.forward``."""

        if not isinstance(reference_output, Mapping):
            raise TypeError(f"{type(self).__name__} expected a mapping reference output.")
        canonical = reference_output.get("canonical")
        if isinstance(canonical, Mapping) and isinstance(canonical.get("train_batch"), Mapping):
            return dict(to_device(canonical["train_batch"], device))
        raise NotImplementedError(f"{type(self).__name__} does not implement training batch adaptation.")

    def v2_infer_module_policy(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
    ) -> InferModulePolicy:
        """Return the module-tier FSM policy for inference parity."""

        del reference_output, whitelist
        policy = self.case.run.policy
        return InferModulePolicy(
            max_steps=policy.max_steps,
            required_nodes=frozenset(policy.required_nodes),
            allow_finalize=policy.allow_finalize,
        )

    def collect_v2_train_observations(
        self,
        model: OmniModel,
        forward_result: Mapping[str, Any],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        observations: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for node, out in forward_result["outputs"].items():
            record_module_output(observations, whitelist, state="train", node=node, out=out)
        self.record_v2_train_gradient_observations(model, observations, whitelist, batch=batch)
        self.record_v2_train_extra_observations(model, observations, whitelist, batch=batch)
        return observations

    def collect_v2_train_framework_observations(
        self,
        model: OmniModel,
        loss_dict: Mapping[str, torch.Tensor],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        observations: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for node, loss in loss_dict.items():
            record_module_output(observations, whitelist, state="train", node=node, out={"_loss": loss})
        self.record_v2_train_gradient_observations(model, observations, whitelist, batch=batch)
        self.record_v2_train_extra_observations(model, observations, whitelist, batch=batch)
        return observations

    def record_v2_train_gradient_observations(
        self,
        model: OmniModel,
        observations: dict[tuple[str, str], list[dict[str, Any]]],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> None:
        for mapping in self.v2_train_gradient_mappings():
            fields = whitelist.get(("train", mapping.node), frozenset())
            if mapping.v2_field not in fields or mapping.v2_grad is None:
                continue
            rows = _rows_from_batch(batch, mapping.v2_grad.rows_from)
            out = {
                mapping.v2_field: sample_named_grad(
                    model.get_module(mapping.v2_grad.module),
                    mapping.v2_grad.parameter,
                    rows=rows,
                )
            }
            record_module_output(observations, whitelist, state="train", node=mapping.node, out=out)

    def v2_train_gradient_mappings(self) -> tuple[ProbeMapping, ...]:
        selected = self.case.model.mapping.for_probe_names(self.case.run.probes)
        return tuple(mapping for mapping in selected if mapping.v2_grad is not None)

    def record_v2_train_extra_observations(
        self,
        model: OmniModel,
        observations: dict[tuple[str, str], list[dict[str, Any]]],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> None:
        del model, observations, whitelist, batch


def _rows_from_batch(batch: Mapping[str, Any], path: str | None) -> torch.Tensor | None:
    if path is None:
        return None
    value: Any = batch
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Unable to resolve v2_grad rows_from path {path!r}.")
        value = value[part]
    if not torch.is_tensor(value):
        raise TypeError(f"v2_grad rows_from path {path!r} must resolve to a tensor.")
    return torch.unique(value.detach().cpu()).to(dtype=torch.long)


__all__ = ["ParityDriver"]
