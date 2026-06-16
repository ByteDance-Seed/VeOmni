"""Default module-tier execution for SeedOmni V2 parity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tests.seed_omni.parity_suite.core.utilities import autocast_for_dtype, sum_losses, zero_module_grads
from tests.seed_omni.parity_suite.v2.observation import arm_generation_observer, record_module_output
from veomni.models.seed_omni.modeling_omni import OmniModel


ModuleNode = tuple[str, str]


@dataclass(frozen=True)
class InferModulePolicy:
    """Controls the shared inference module-tier FSM loop.

    ``required_nodes`` lets short recipes stop after the evidence they compare.
    ``allow_finalize`` permits policies that need final generated artifacts.
    """

    max_steps: int | None = None
    required_nodes: frozenset[tuple[str, str]] = frozenset()
    allow_finalize: bool = False


def run_v2_infer_module(
    driver: Any,
    reference_output: Any,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Run a V2 inference graph through direct FSM module steps."""

    model = driver.load_v2_model(device=device, dtype=dtype)
    request = driver.v2_infer_request(reference_output, device=device)
    generation_kwargs = driver.generation_kwargs(model)
    return run_infer_module_fsm(
        model,
        request,
        whitelist,
        generation_kwargs=generation_kwargs,
        policy=driver.v2_infer_module_policy(reference_output, whitelist),
    )


def run_v2_train_module(
    driver: Any,
    reference_output: Any,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Run a V2 training graph one node at a time."""

    return run_v2_train_module_batch(
        driver,
        driver.v2_train_batch_kwargs(reference_output, device=device),
        whitelist,
        device=device,
        dtype=dtype,
    )


def run_v2_train_module_batch(
    driver: Any,
    batch_kwargs: Mapping[str, Any],
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    model = driver.load_v2_model(device=device, dtype=dtype).train()
    zero_module_grads(model.modules_dict.values())
    batch = dict(batch_kwargs)
    with torch.enable_grad(), autocast_for_dtype(device, dtype):
        forward_result = run_v2_train_nodes(driver, model, batch)
        loss = forward_result["loss"]
        if loss is None:
            raise RuntimeError(f"{type(driver).__name__} V2 train module tier produced no loss.")
    loss.backward()
    observations = driver.collect_v2_train_observations(model, forward_result, whitelist, batch=batch)
    return {"observations": observations, "ctx": forward_result, "trace": ["train:module"]}


def run_v2_train_nodes(driver: Any, model: OmniModel, batch: dict[str, Any]) -> dict[str, Any]:
    del driver
    if model.training_graph is None:
        raise RuntimeError("Train module tier requires a training graph.")
    node_outputs: dict[str, dict[str, Any]] = {}
    losses: dict[str, torch.Tensor] = {}
    for node_name in model.training_graph.execution_order:
        module_name = model.training_graph.module_of(node_name)
        method = model.training_graph.method_of(node_name)
        module = model.get_module(module_name)
        # Mirror OmniModel's graph wiring one node at a time so module-tier
        # tests can localize parity failures below the full graph boundary.
        kwargs = model.training_graph.collect_inputs(node_name, node_outputs, batch)
        call_kwargs = module.pre_forward(method, **kwargs)
        fn = module if method == "forward" else getattr(module, method)
        outputs = fn(**call_kwargs)
        out = module.post_forward(method, **outputs)
        node_outputs[node_name] = out
        for key, value in out.items():
            if key in batch:
                batch[key] = value
        if "_loss" in out:
            losses[node_name] = out["_loss"]
    return {"loss": sum_losses(losses), "losses": losses, "outputs": node_outputs}


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
            # Recipes usually need only a few observed nodes, not a complete
            # generation. Stop as soon as the requested evidence is present.
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
    "run_v2_infer_module",
    "run_v2_train_module",
    "run_v2_train_module_batch",
    "run_v2_train_nodes",
]
