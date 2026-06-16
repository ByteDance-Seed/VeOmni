"""Train observation and gradient capture helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import ParityCase, ProbeMapping, sample_named_grad
from tests.seed_omni.parity_suite.v2.observation import record_module_output
from tests.seed_omni.parity_suite.v2.tier_runners.module import InferModulePolicy
from veomni.models.seed_omni.modeling_omni import OmniModel


class TrainObservationMixin:
    """Collect training observations and optional gradient probes."""

    case: ParityCase

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
        selected = self.case.model.probes.for_probe_names(self.case.run.probes)
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
