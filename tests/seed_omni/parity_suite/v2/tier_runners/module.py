"""Default module-tier execution for SeedOmni V2 parity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tests.seed_omni.parity_suite.core import autocast_for_dtype
from tests.seed_omni.parity_suite.v2.model import load_graph_active_omni_config
from tests.seed_omni.parity_suite.v2.observation import arm_generation_observer
from tests.seed_omni.parity_suite.v2.tier_runners.graph import _cpu_origin_randn_for_reference_parity
from veomni.models.seed_omni.conversation import ConversationItem
from veomni.models.seed_omni.generation_graph import GenerationGraph


@dataclass
class _ModuleRuntime:
    config: Any
    modules: Mapping[str, nn.Module]
    device: torch.device
    generation_graph: GenerationGraph | None = None
    _generated: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self._generated = []

    @property
    def modules_dict(self) -> Mapping[str, nn.Module]:
        return self.modules

    @property
    def generated(self) -> list[dict[str, Any]]:
        return list(self._generated or [])

    def reset(self) -> None:
        if self.generation_graph is not None:
            self.generation_graph.reset()
        if self._generated is not None:
            self._generated.clear()
        for module in self.modules.values():
            reset = getattr(module, "reset_global_inference_state", None)
            if reset is not None:
                reset()

    def _append_generated(self, item: Any) -> None:
        if item is None:
            return
        if isinstance(item, dict) and "type" in item and "value" in item:
            normalized = {"type": item["type"], "value": item["value"]}
            if item.get("meta") is not None:
                normalized["meta"] = item["meta"]
            if self._generated is not None:
                self._generated.append(normalized)

    def _collect_generated(self, ctx: dict[str, Any], trace: list[str] | None = None) -> None:
        generated = ctx.pop("generated", None)
        self._append_generated(generated)
        if trace is not None and generated is not None:
            trace.append(f"generated:{generated['type']}")

    def _invoke_module_finalize(self, ctx: dict[str, Any], trace: list[str] | None = None) -> None:
        for name in self.config.module_names:
            module = self.modules[name]
            out = module.finalize(ctx=ctx)
            if not isinstance(out, dict):
                raise TypeError(f"{type(module).__name__}.finalize must return a dict, got {type(out).__name__}.")
            generated = out.pop("generated", None)
            self._append_generated(generated)
            if trace is not None and generated is not None:
                trace.append(f"finalize:{name} | generated:{generated['type']}")
            ctx.update(_materialize_for_device(out, self.device))


@dataclass(frozen=True)
class InferModulePolicy:
    """Controls the shared inference module-tier FSM loop.

    ``required_nodes`` lets short recipes stop after the evidence they compare.
    ``required_observations`` is stricter: each tuple is
    ``(state, node, field, item_type)`` and is satisfied only when the
    requested field is present on the requested carrier item type.  This keeps
    CFG branch collection from stopping on an intermediate node execution whose
    output item has not yet produced the compared value.
    ``allow_finalize`` permits policies that need final generated artifacts.
    """

    max_steps: int | None = None
    required_nodes: frozenset[tuple[str, str]] = frozenset()
    required_observations: frozenset[tuple[str, str, str, str | None]] = frozenset()
    allow_finalize: bool = False
    max_tensor_numel: int = 1_000_000


def run_v2_infer_module(
    driver: Any,
    reference_output: Any,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Run module-tier inference through eager modules and generation-graph automation."""

    config = load_graph_active_omni_config(driver.case, driver.v2_module_names())
    modules = driver.load_v2_modules(config.module_names, device=device, dtype=dtype)
    for module in modules.values():
        module.eval()
    model = _ModuleRuntime(
        config=config,
        modules=modules,
        device=device,
        generation_graph=GenerationGraph(config.generation_graph) if config.has_generation_graph() else None,
    )
    request = driver.v2_request_kwargs(reference_output, device=device)
    generation_kwargs = driver.generation_kwargs(model, reference_output)
    with torch.no_grad(), autocast_for_dtype(device, dtype), _cpu_origin_randn_for_reference_parity():
        return _run_infer_module_fsm(
            model,
            _materialize_for_device(dict(request), device),
            whitelist,
            generation_kwargs=_materialize_for_device(generation_kwargs, device),
            policy=driver.v2_infer_module_policy(reference_output, whitelist),
        )


def _run_infer_module_fsm(
    model: Any,
    request: Mapping[str, Any],
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    generation_kwargs: Mapping[str, Any],
    policy: InferModulePolicy | None = None,
) -> dict[str, Any]:
    if model.generation_graph is None:
        raise RuntimeError("Inference module-tier execution requires a generation graph.")

    active_policy = policy or InferModulePolicy()
    ctx = dict(request)
    trace: list[str] = []
    model.reset()
    modules = model.modules_dict
    max_steps = active_policy.max_steps
    if max_steps is None:
        max_steps = int(generation_kwargs.get("max_new_tokens", 2048))
    required = active_policy.required_nodes
    required_observations = active_policy.required_observations

    with arm_generation_observer(whitelist, max_tensor_numel=active_policy.max_tensor_numel) as observations:
        for _ in range(max_steps):
            # Recipes usually need only a few observed nodes, not a complete
            # generation. Stop as soon as the requested evidence is present.
            if _observed_all_required(observations, required, required_observations):
                break
            if model.generation_graph.is_done():
                break
            step_device = getattr(model, "device", torch.device("cpu"))
            ctx = _materialize_for_device(ctx, step_device)
            step_kwargs = _materialize_for_device(dict(generation_kwargs), step_device)
            ctx = model.generation_graph.step(modules, ctx, trace=trace, generation_kwargs=step_kwargs)
            ctx = _materialize_for_device(ctx, step_device)
            _collect_generated_if_supported(model, ctx, trace)
            model.generation_graph.maybe_transition(ctx, trace=trace)
            if _observed_all_required(observations, required, required_observations):
                break
        else:
            if active_policy.allow_finalize and not model.generation_graph.is_done():
                _invoke_finalize_if_supported(model, ctx, trace)

    if (required or required_observations) and not _observed_all_required(
        observations,
        required,
        required_observations,
    ):
        missing = _missing_required(observations, required, required_observations)
        raise RuntimeError(
            "Inference module-tier FSM stopped before observing all required nodes. "
            f"Missing: {missing}. Trace: {trace}"
        )
    return {"observations": dict(observations), "ctx": ctx, "trace": trace}


def _observed_all_required(
    observations: Mapping[tuple[str, str], list[dict[str, Any]]],
    required: frozenset[tuple[str, str]],
    required_observations: frozenset[tuple[str, str, str, str | None]] = frozenset(),
) -> bool:
    if required_observations:
        return all(
            _has_required_observation(observations, state, node, field, item_type)
            for state, node, field, item_type in required_observations
        )
    if not required:
        return False
    return required.issubset({key for key, records in observations.items() if records})


def _has_required_observation(
    observations: Mapping[tuple[str, str], list[dict[str, Any]]],
    state: str,
    node: str,
    field: str,
    item_type: str | None,
) -> bool:
    for record in observations.get((state, node), []):
        if item_type is not None and record.get("_item_type") != item_type:
            continue
        if field in record:
            return True
    return False


def _missing_required(
    observations: Mapping[tuple[str, str], list[dict[str, Any]]],
    required: frozenset[tuple[str, str]],
    required_observations: frozenset[tuple[str, str, str, str | None]],
) -> list[Any]:
    if required_observations:
        return sorted(
            requirement
            for requirement in required_observations
            if not _has_required_observation(observations, *requirement)
        )
    return sorted(required.difference(observations.keys()))


def _collect_generated_if_supported(model: Any, ctx: dict[str, Any], trace: list[str]) -> None:
    collect_generated = getattr(model, "_collect_generated", None)
    if collect_generated is not None:
        collect_generated(ctx, trace)


def _invoke_finalize_if_supported(model: Any, ctx: dict[str, Any], trace: list[str]) -> None:
    invoke_finalize = getattr(model, "_invoke_module_finalize", None)
    if invoke_finalize is not None:
        invoke_finalize(ctx, trace=trace)


def _materialize_for_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device=device)
    if isinstance(value, ConversationItem):
        value.value = _materialize_for_device(value.value, device)
        value.meta = _materialize_for_device(value.meta, device)
        return value
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _materialize_for_device(item, device)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _materialize_for_device(item, device)
        return value
    if isinstance(value, tuple):
        return tuple(_materialize_for_device(item, device) for item in value)
    return value


__all__ = [
    "InferModulePolicy",
    "run_v2_infer_module",
]
