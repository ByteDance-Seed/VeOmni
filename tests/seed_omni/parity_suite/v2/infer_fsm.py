"""Shared V2 inference FSM helpers for parity drivers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

from tests.seed_omni.parity_suite.v2.observation import arm_generation_observer, record_module_output


ModuleNode = tuple[str, str]


@dataclass(frozen=True)
class InferModulePolicy:
    """Controls the shared inference module-tier FSM loop."""

    max_steps: int | None = None
    required_nodes: frozenset[tuple[str, str]] = frozenset()
    allow_finalize: bool = False


def run_module_nodes(
    nodes: Sequence[ModuleNode],
    *,
    modules: Mapping[str, nn.Module],
    ctx: dict[str, Any],
    observations: dict[tuple[str, str], list[dict[str, Any]]],
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    state: str,
    generation_kwargs: Mapping[str, Any],
) -> None:
    for module_name, method in nodes:
        out = getattr(modules[module_name], method)(**ctx, generation_kwargs=generation_kwargs)
        record_module_output(
            observations,
            whitelist,
            state=state,
            node=f"{module_name}.{method}",
            out=out,
        )
        ctx.update(out)


def run_infer_module_fsm(
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

    with arm_generation_observer(whitelist) as observations:
        for _ in range(max_steps):
            if _observed_all_required(observations, required):
                break
            if model.generation_graph.is_done():
                break
            ctx = model.generation_graph.step(modules, ctx, trace=trace, generation_kwargs=dict(generation_kwargs))
            _collect_generated_if_supported(model, ctx, trace)
            model.generation_graph.maybe_transition(ctx, trace=trace)
            if _observed_all_required(observations, required):
                break
        else:
            if active_policy.allow_finalize and not model.generation_graph.is_done():
                _invoke_finalize_if_supported(model, ctx, trace)

    if required and not _observed_all_required(observations, required):
        missing = sorted(required.difference(observations.keys()))
        raise RuntimeError(
            "Inference module-tier FSM stopped before observing all required nodes. "
            f"Missing: {missing}. Trace: {trace}"
        )
    return {"observations": dict(observations), "ctx": ctx, "trace": trace}


def _observed_all_required(
    observations: Mapping[tuple[str, str], list[dict[str, Any]]],
    required: frozenset[tuple[str, str]],
) -> bool:
    if not required:
        return False
    return required.issubset({key for key, records in observations.items() if records})


def _collect_generated_if_supported(model: Any, ctx: dict[str, Any], trace: list[str]) -> None:
    collect_generated = getattr(model, "_collect_generated", None)
    if collect_generated is not None:
        collect_generated(ctx, trace)


def _invoke_finalize_if_supported(model: Any, ctx: dict[str, Any], trace: list[str]) -> None:
    invoke_finalize = getattr(model, "_invoke_module_finalize", None)
    if invoke_finalize is not None:
        invoke_finalize(ctx, trace=trace)


__all__ = [
    "InferModulePolicy",
    "ModuleNode",
    "run_infer_module_fsm",
    "run_module_nodes",
]
