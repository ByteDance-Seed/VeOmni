"""Train observation and gradient capture helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import ParityCase, ProbeMapping, sample_named_grad
from tests.seed_omni.parity_suite.driver.v2_run import V2RunContext
from tests.seed_omni.parity_suite.v2.generation_runtime import InferModulePolicy
from tests.seed_omni.parity_suite.v2.observation import record_conversation_output, record_module_output
from veomni.models.seed_omni.modeling_omni import OmniModel


def shifted_label_rows_from_conversation(
    conversation_list: Any,
    *,
    label_key: str = "labels",
    ignore_index: int = -100,
) -> torch.Tensor | None:
    """Return unique next-token label rows from a conversation carrier."""

    labels: list[torch.Tensor] = []
    for sample in conversation_list or []:
        for item in sample:
            meta = getattr(item, "meta", {})
            item_labels = meta.get(label_key) if isinstance(meta, dict) else None
            if not torch.is_tensor(item_labels):
                continue
            shifted = item_labels.reshape(-1)[1:]
            shifted = shifted[shifted != ignore_index]
            if int(shifted.numel()) > 0:
                labels.append(shifted.detach().cpu())
    if not labels:
        return None
    return torch.unique(torch.cat(labels)).to(dtype=torch.long)


class TrainObservationMixin:
    """Collect training observations and optional gradient probes."""

    case: ParityCase

    def v2_module_fsm_policy(self, ctx: V2RunContext) -> InferModulePolicy:
        """Return the module-tier FSM policy for inference parity."""

        options = self.case.run.options
        max_steps = options.get("max_steps")
        selected = self.case.model.probes.for_probe_names(self.case.run.probes)
        needs_all_steps = any(mapping.step == "all" for mapping in selected)
        required_observations = frozenset(
            (mapping.state, mapping.node, mapping.v2_field, mapping.v2_item_type)
            for mapping in selected
            if mapping.state is not None
        )
        return InferModulePolicy(
            max_steps=None if max_steps is None else int(max_steps),
            required_nodes=frozenset() if needs_all_steps else frozenset(ctx.whitelist.keys()),
            required_observations=frozenset() if needs_all_steps else required_observations,
            allow_finalize=bool(options.get("allow_finalize", False)),
            max_tensor_numel=ctx.capture_options.max_tensor_numel,
        )

    def collect_v2_observations(
        self,
        ctx: V2RunContext,
        *,
        model: OmniModel,
        result: Mapping[str, Any],
        batch: Mapping[str, Any] | None = None,
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        batch = {} if batch is None else batch
        if ctx.domain == "training" and ctx.tier == "graph":
            observations = self._collect_v2_train_forward_observations(result, ctx.whitelist)
            self._record_v2_train_gradient_observations(ctx, model, observations, batch=batch)
            self._record_v2_train_graph_extra_observations(model, observations, ctx.whitelist, batch=batch)
            return observations
        if ctx.domain == "training" and ctx.tier == "framework":
            losses = result.get("losses", result)
            if not isinstance(losses, Mapping):
                raise TypeError("V2 training framework observation result must contain a mapping of losses.")
            observations: dict[tuple[str, str], list[dict[str, Any]]] = {}
            self._record_v2_train_loss_observations(losses, observations, ctx.whitelist)
            self._record_v2_train_gradient_observations(ctx, model, observations, batch=batch)
            self._record_v2_train_framework_extra_observations(model, observations, ctx.whitelist, batch=batch)
            return observations
        raise NotImplementedError(f"Unsupported V2 observation context: domain={ctx.domain!r}, tier={ctx.tier!r}.")

    # Driver extension hooks -------------------------------------------------------

    def v2_gradient_rows(
        self,
        ctx: V2RunContext,
        batch: Mapping[str, Any],
        mapping: ProbeMapping,
    ) -> torch.Tensor | None:
        """Return optional row indices for gradient sampling."""

        del ctx, batch, mapping
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
        outputs = forward_result.get("outputs")
        if isinstance(outputs, Mapping):
            for node, out in outputs.items():
                record_module_output(observations, whitelist, state="train", node=node, out=out)
                record_conversation_output(
                    observations,
                    whitelist,
                    state="train",
                    node=node,
                    conversation_list=out.get("conversation_list"),
                )
            return observations

        losses = forward_result.get("losses")
        if not isinstance(losses, Mapping):
            raise TypeError("V2 training graph observation result must contain either outputs or losses.")
        self._record_v2_train_loss_observations(losses, observations, whitelist)
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
        ctx: V2RunContext,
        model: OmniModel,
        observations: dict[tuple[str, str], list[dict[str, Any]]],
        *,
        batch: Mapping[str, Any],
    ) -> None:
        for mapping in self._v2_train_gradient_mappings():
            fields = ctx.whitelist.get(("train", mapping.node), frozenset())
            if mapping.v2_field not in fields or mapping.v2_grad is None:
                continue
            rows = self.v2_gradient_rows(ctx, batch, mapping)
            out = {
                mapping.v2_field: sample_named_grad(
                    model.get_module(mapping.v2_grad.module),
                    mapping.v2_grad.parameter,
                    rows=rows,
                )
            }
            record_module_output(observations, ctx.whitelist, state="train", node=mapping.node, out=out)

    def _v2_train_gradient_mappings(self) -> tuple[ProbeMapping, ...]:
        selected = self.case.model.probes.for_probe_names(self.case.run.probes)
        return tuple(mapping for mapping in selected if mapping.v2_grad is not None)
