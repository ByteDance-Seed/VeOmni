"""Train observation and gradient capture helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import ParityCase, ProbeMapping, sample_named_grad
from tests.seed_omni.parity_suite.v2.observation import record_conversation_output, record_module_output
from tests.seed_omni.parity_suite.v2.tier_runners.module import InferModulePolicy
from veomni.models.seed_omni.modeling_omni import OmniModel


class TrainObservationMixin:
    """Collect training observations and optional gradient probes."""

    case: ParityCase

    # Inference module-tier policy -------------------------------------------------

    def v2_infer_module_policy(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
    ) -> InferModulePolicy:
        """Return the module-tier FSM policy for inference parity."""

        del reference_output
        options = self.case.run.options
        max_steps = options.get("max_steps")
        max_tensor_numel = options.get("max_tensor_numel", 1_000_000)
        selected = self.case.model.probes.for_probe_names(self.case.run.probes)
        needs_all_steps = any(mapping.step == "all" for mapping in selected)
        required_observations = frozenset(
            (mapping.state, mapping.node, mapping.v2_field, mapping.v2_item_type)
            for mapping in selected
            if mapping.state is not None
        )
        return InferModulePolicy(
            max_steps=None if max_steps is None else int(max_steps),
            required_nodes=frozenset() if needs_all_steps else frozenset(whitelist.keys()),
            required_observations=frozenset() if needs_all_steps else required_observations,
            allow_finalize=bool(options.get("allow_finalize", False)),
            max_tensor_numel=int(max_tensor_numel),
        )

    # Public train-tier collectors -------------------------------------------------

    def collect_v2_train_graph_observations(
        self,
        model: OmniModel,
        forward_result: Mapping[str, Any],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        observations = self._collect_v2_train_forward_observations(forward_result, whitelist)
        self._record_v2_train_gradient_observations(model, observations, whitelist, batch=batch)
        self._record_v2_train_graph_extra_observations(model, observations, whitelist, batch=batch)
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
        self._record_v2_train_loss_observations(loss_dict, observations, whitelist)
        self._record_v2_train_gradient_observations(model, observations, whitelist, batch=batch)
        self._record_v2_train_framework_extra_observations(model, observations, whitelist, batch=batch)
        return observations

    # Driver extension hooks -------------------------------------------------------

    def gradient_rows(
        self,
        batch: Mapping[str, Any],
        mapping: ProbeMapping,
    ) -> torch.Tensor | None:
        """Return optional row indices for gradient sampling."""

        del batch, mapping
        return None

    def _record_v2_train_graph_extra_observations(
        self,
        model: OmniModel,
        observations: dict[tuple[str, str], list[dict[str, Any]]],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> None:
        del model, observations, whitelist, batch

    def _record_v2_train_framework_extra_observations(
        self,
        model: OmniModel,
        observations: dict[tuple[str, str], list[dict[str, Any]]],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> None:
        del model, observations, whitelist, batch

    # Internal recorders -----------------------------------------------------------

    def _collect_v2_train_forward_observations(
        self,
        forward_result: Mapping[str, Any],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        observations: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for node, out in forward_result["outputs"].items():
            record_module_output(observations, whitelist, state="train", node=node, out=out)
            record_conversation_output(
                observations,
                whitelist,
                state="train",
                node=node,
                conversation_list=out.get("conversation_list"),
            )
        return observations

    @staticmethod
    def _record_v2_train_loss_observations(
        loss_dict: Mapping[str, torch.Tensor],
        observations: dict[tuple[str, str], list[dict[str, Any]]],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
    ) -> None:
        # Framework forward_backward currently exposes loss_dict, not per-node
        # carrier outputs. If trainer paths start returning node outputs or the
        # final conversation carrier, route them through
        # _collect_v2_train_forward_observations / record_conversation_output
        # instead of adding model-specific carrier readers.
        for node, loss in loss_dict.items():
            record_module_output(observations, whitelist, state="train", node=node, out={"_loss": loss})

    def _record_v2_train_gradient_observations(
        self,
        model: OmniModel,
        observations: dict[tuple[str, str], list[dict[str, Any]]],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> None:
        for mapping in self._v2_train_gradient_mappings():
            fields = whitelist.get(("train", mapping.node), frozenset())
            if mapping.v2_field not in fields or mapping.v2_grad is None:
                continue
            rows = self.gradient_rows(batch, mapping)
            out = {
                mapping.v2_field: sample_named_grad(
                    model.get_module(mapping.v2_grad.module),
                    mapping.v2_grad.parameter,
                    rows=rows,
                )
            }
            record_module_output(observations, whitelist, state="train", node=mapping.node, out=out)

    def _v2_train_gradient_mappings(self) -> tuple[ProbeMapping, ...]:
        selected = self.case.model.probes.for_probe_names(self.case.run.probes)
        return tuple(mapping for mapping in selected if mapping.v2_grad is not None)
