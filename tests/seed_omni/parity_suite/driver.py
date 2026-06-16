"""Base driver contract for model-specific parity execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from tests.seed_omni.parity_suite.core import (
    ParityCase,
    ParityReport,
    ProbeMapping,
    configure_torch_determinism,
)
from tests.seed_omni.parity_suite.core.utilities import sample_named_grad
from tests.seed_omni.parity_suite.reference.capture import (
    ReferenceCaptureContext,
)
from tests.seed_omni.parity_suite.reference.loader import load_transformers_reference_model
from tests.seed_omni.parity_suite.reference.model import reference_options
from tests.seed_omni.parity_suite.v2.model import (
    graph_active_module_names,
    load_graph_active_omni_config,
    load_graph_active_omni_modules,
)
from tests.seed_omni.parity_suite.v2.observation import record_module_output
from tests.seed_omni.parity_suite.v2.tier_runners.module import InferModulePolicy
from veomni.models.seed_omni.modeling_omni import OmniModel


class ParityDriver:
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
        return self.case.recipe.stimulus

    def generation_kwargs(self, model_or_config: Any) -> dict[str, Any]:
        config = getattr(model_or_config, "config", model_or_config)
        kwargs = dict(getattr(config, "generation_kwargs", None) or {})
        for key, default in self.generation_defaults.items():
            kwargs[key] = self.case.recipe.stimulus.get(key, default)
        return kwargs

    def reference_model_load_kwargs(self, *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        """Return model-specific kwargs for ``AutoModel.from_pretrained``."""

        del device, dtype
        return {}

    def load_reference_model(self, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        """Load the independent reference oracle."""

        return load_transformers_reference_model(
            self.case.model.reference,
            **self.reference_model_load_kwargs(device=device, dtype=dtype),
        )

    def load_v2_model(self, *, device: torch.device, dtype: torch.dtype) -> OmniModel:
        """Load the V2 model under test."""

        module_names = self.v2_module_names()
        config = load_graph_active_omni_config(self.case, module_names)
        modules = self.load_v2_modules(config.module_names, device=device, dtype=dtype)
        return OmniModel(config, modules).eval()

    def v2_module_names(self) -> frozenset[str]:
        """Return the complete module set referenced by the selected V2 graph."""

        return graph_active_module_names(self.case)

    def load_v2_modules(
        self,
        module_names: tuple[str, ...] | list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mapping[str, nn.Module]:
        """Load graph-active V2 modules.

        Drivers with non-standard checkpoint layouts can override this hook
        while keeping the shared graph-driven config behavior.
        """

        return load_graph_active_omni_modules(self.case, module_names, device=device, dtype=dtype)

    def run_reference_recipe(
        self,
        ref_model: nn.Module,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
    ) -> Any:
        """Run the reference recipe and return driver-owned canonical output."""

        reference = self.case.recipe.reference
        kind = reference.get("kind")
        if kind is not None:
            kind = str(kind)
        options = reference.get("options", {}) or {}
        with reference_options(ref_model, options):
            run_reference_kind = getattr(ref_model, "run_reference_kind", None)
            if run_reference_kind is not None:
                return run_reference_kind(kind, inputs, context)
            if kind is None:
                return ref_model(**inputs)
        raise NotImplementedError(f"{type(ref_model).__name__} does not implement reference kind {kind!r}.")

    def run_reference_only_recipe(self) -> ParityReport:
        """Run a reference-only recipe."""

        raise NotImplementedError(f"{type(self).__name__} does not implement reference-only execution.")

    def run_v2_infer_graph_recipe(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 inference graph through the shared graph dispatcher."""

        from tests.seed_omni.parity_suite.v2.tier_runners.graph import run_v2_infer_graph

        return run_v2_infer_graph(self, reference_output, whitelist, device=device, dtype=dtype)

    def run_v2_train_graph_recipe(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 training graph through the shared graph dispatcher."""

        from tests.seed_omni.parity_suite.v2.tier_runners.graph import run_v2_train_graph

        return run_v2_train_graph(self, reference_output, whitelist, device=device, dtype=dtype)

    def run_v2_infer_module_recipe(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 inference graph through the shared module dispatcher."""

        from tests.seed_omni.parity_suite.v2.tier_runners.module import run_v2_infer_module

        return run_v2_infer_module(self, reference_output, whitelist, device=device, dtype=dtype)

    def run_v2_train_module_recipe(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 training graph through the shared module dispatcher."""

        from tests.seed_omni.parity_suite.v2.tier_runners.module import run_v2_train_module

        return run_v2_train_module(self, reference_output, whitelist, device=device, dtype=dtype)

    def run_v2_infer_framework_recipe(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any] | ParityReport:
        """Run an inference framework-tier case through the shared framework dispatcher."""

        from tests.seed_omni.parity_suite.v2.tier_runners.framework import run_v2_infer_framework

        return run_v2_infer_framework(self, reference_output, whitelist, device=device, dtype=dtype)

    def run_v2_train_framework_recipe(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any] | ParityReport:
        """Run a training framework-tier case through the shared framework dispatcher."""

        from tests.seed_omni.parity_suite.v2.tier_runners.framework import run_v2_train_framework

        return run_v2_train_framework(self, reference_output, whitelist, device=device, dtype=dtype)

    def sample_v2_framework_parameters(
        self,
        model: OmniModel,
        batch: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        del model, batch
        return {}

    def v2_infer_request(self, reference_output: Any, *, device: torch.device) -> dict[str, Any]:
        """Adapt driver-owned canonical output into an ``OmniModel.generate`` request.

        Shared suite code treats the returned mapping as opaque apart from common
        V2 request keys such as ``conversation_list``. Concrete conversion from
        reference canonical data or model-owned fixtures belongs to model drivers.
        """

        del reference_output, device
        raise NotImplementedError(f"{type(self).__name__} does not implement inference request adaptation.")

    def v2_train_batch_kwargs(self, reference_output: Any, *, device: torch.device) -> dict[str, Any]:
        """Adapt driver-owned canonical output into kwargs for ``OmniModel.forward``.

        The base suite defines only the adapter shape. It must not know a
        model's internal tensor layout. Model-specific drivers should return a
        V2 request, typically ``{"conversation_list": ...}``, and let runtime
        hooks convert that request into model internals.
        """

        del reference_output, device
        raise NotImplementedError(f"{type(self).__name__} does not implement training batch adaptation.")

    def v2_infer_module_policy(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
    ) -> InferModulePolicy:
        """Return the module-tier FSM policy for inference parity."""

        del reference_output, whitelist
        options = self.case.run.options
        required_nodes = tuple(tuple(item) for item in options.get("required_nodes", ()) or ())
        max_steps = options.get("max_steps")
        return InferModulePolicy(
            max_steps=None if max_steps is None else int(max_steps),
            required_nodes=frozenset((str(state), str(node)) for state, node in required_nodes),
            allow_finalize=bool(options.get("allow_finalize", False)),
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
