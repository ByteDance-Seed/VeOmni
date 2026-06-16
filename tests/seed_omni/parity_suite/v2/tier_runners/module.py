"""Default module-tier execution for SeedOmni V2 parity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tests.seed_omni.parity_suite.core import autocast_for_dtype, sum_losses
from tests.seed_omni.parity_suite.v2.model import load_graph_active_omni_config
from tests.seed_omni.parity_suite.v2.observation import arm_generation_observer, record_module_output
from veomni.models.seed_omni.conversation import ConversationItem
from veomni.models.seed_omni.generation_graph import GenerationGraph
from veomni.models.seed_omni.training_graph import TrainingGraph


ModuleNode = tuple[str, str]


@dataclass
class _LazyGraphRuntime:
    config: Any
    module_runner: LazyModuleRunner
    training_graph: TrainingGraph | None = None
    generation_graph: GenerationGraph | None = None
    _generated: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self._generated = []

    @property
    def modules_dict(self) -> _LazyModuleMapping:
        return _LazyModuleMapping(self.module_runner)

    @property
    def generated(self) -> list[dict[str, Any]]:
        return list(self._generated or [])

    def reset(self) -> None:
        if self.generation_graph is not None:
            self.generation_graph.reset()
        if self._generated is not None:
            self._generated.clear()
        for module in self.module_runner.cached_modules():
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
            module = self.module_runner.ensure(name, training=False)
            out = module.finalize(ctx=ctx)
            if not isinstance(out, dict):
                raise TypeError(f"{type(module).__name__}.finalize must return a dict, got {type(out).__name__}.")
            generated = out.pop("generated", None)
            self._append_generated(generated)
            if trace is not None and generated is not None:
                trace.append(f"finalize:{name} | generated:{generated['type']}")
            ctx.update(_materialize_for_device(out, self.module_runner.offload_device))


class _LazyModuleMapping(Mapping[str, nn.Module]):
    def __init__(self, runner: LazyModuleRunner) -> None:
        self._runner = runner

    def __getitem__(self, key: str) -> nn.Module:
        return self._runner.ensure(key, training=False)

    def __iter__(self):
        return iter(self._runner.module_names)

    def __len__(self) -> int:
        return len(self._runner.module_names)

    def get(self, key: str, default: Any = None) -> nn.Module | Any:
        if key not in self._runner.module_names:
            return default
        return self._runner.ensure(key, training=False)


class LazyModuleRunner:
    """Load one V2 module onto the target device at a time for module-tier parity."""

    def __init__(
        self,
        driver: Any,
        module_names: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        offload_device: torch.device | None = None,
    ) -> None:
        self.driver = driver
        self.module_names = tuple(module_names)
        self.device = device
        self.dtype = dtype
        self.offload_device = offload_device or torch.device("cpu")
        self._modules: dict[str, nn.Module] = {}
        self._resident: str | None = None

    def ensure(self, module_name: str, *, training: bool) -> nn.Module:
        if module_name not in self.module_names:
            raise KeyError(f"Unknown module {module_name!r}; active modules: {sorted(self.module_names)}")
        if self._resident != module_name:
            self.release_resident()
            module = self._modules.get(module_name)
            if module is None:
                loaded = self.driver.load_v2_modules([module_name], device=self.device, dtype=self.dtype)
                module = loaded[module_name]
                self._modules[module_name] = module
            else:
                module.to(device=self.device)
            self._resident = module_name
        module = self._modules[module_name]
        module.train(training)
        return module

    def release_resident(self) -> None:
        if self._resident is None:
            return
        module = self._modules[self._resident]
        module.to(device=self.offload_device)
        self._resident = None
        _empty_cuda_cache()

    def cached_modules(self) -> tuple[nn.Module, ...]:
        return tuple(self._modules.values())

    def close(self) -> None:
        self.release_resident()
        self._modules.clear()
        _empty_cuda_cache()


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

    config = load_graph_active_omni_config(driver.case, driver.v2_module_names())
    module_runner = LazyModuleRunner(driver, config.module_names, device=device, dtype=dtype)
    model = _LazyGraphRuntime(
        config=config,
        module_runner=module_runner,
        generation_graph=GenerationGraph(config.generation_graph) if config.has_generation_graph() else None,
    )
    request = driver.v2_request_kwargs(reference_output, device=device)
    generation_kwargs = driver.generation_kwargs(model, reference_output)
    try:
        return run_infer_module_fsm(
            model,
            _materialize_for_device(dict(request), module_runner.offload_device),
            whitelist,
            generation_kwargs=_materialize_for_device(generation_kwargs, module_runner.offload_device),
            policy=driver.v2_infer_module_policy(reference_output, whitelist),
        )
    finally:
        module_runner.close()


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
        driver.v2_request_kwargs(reference_output, device=device),
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
    config = load_graph_active_omni_config(driver.case, driver.v2_module_names())
    training_graph = TrainingGraph(config.training_graph) if config.has_training_graph() else None
    if training_graph is None:
        raise RuntimeError("Train module tier requires a training graph.")
    module_runner = LazyModuleRunner(driver, config.module_names, device=device, dtype=dtype)
    batch = _materialize_for_device(dict(batch_kwargs), module_runner.offload_device)
    try:
        with torch.no_grad(), autocast_for_dtype(device, dtype):
            forward_result = run_v2_train_nodes(driver, training_graph, module_runner, batch)
        loss = forward_result["loss"]
        if loss is None:
            raise RuntimeError(f"{type(driver).__name__} V2 train module tier produced no loss.")
        observations = _collect_train_forward_observations(forward_result, whitelist)
        return {"observations": observations, "ctx": forward_result, "trace": ["train:module"]}
    finally:
        module_runner.close()


def run_v2_train_nodes(
    driver: Any,
    training_graph: TrainingGraph,
    module_runner: LazyModuleRunner,
    batch: dict[str, Any],
) -> dict[str, Any]:
    del driver
    node_outputs: dict[str, dict[str, Any]] = {}
    losses: dict[str, torch.Tensor] = {}
    for node_name in training_graph.execution_order:
        module_name = training_graph.module_of(node_name)
        method = training_graph.method_of(node_name)
        module = module_runner.ensure(module_name, training=True)
        # Mirror OmniModel's graph wiring one node at a time so module-tier
        # tests can localize parity failures below the full graph boundary.
        kwargs = training_graph.collect_inputs(node_name, node_outputs, batch)
        kwargs = _materialize_for_device(kwargs, module_runner.device)
        call_kwargs = module.pre_forward(method, **kwargs)
        fn = module if method == "forward" else getattr(module, method)
        outputs = fn(**call_kwargs)
        out = module.post_forward(method, **outputs)
        out = _materialize_for_device(out, module_runner.offload_device)
        node_outputs[node_name] = out
        convo = out.get("conversation_list")
        if convo is not None:
            batch["conversation_list"] = convo
        if "bagel_packed_batch" in out:
            batch["bagel_packed_batch"] = out["bagel_packed_batch"]
        if "_loss" in out:
            losses[node_name] = out["_loss"]
        module_runner.release_resident()
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
        module = modules[module_name]
        ctx.update(_materialize_for_device(ctx, _module_device(module)))
        out = getattr(module, method)(**ctx, generation_kwargs=generation_kwargs)
        out = _materialize_for_device(out, torch.device("cpu"))
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
            step_device = model.module_runner.device if hasattr(model, "module_runner") else torch.device("cpu")
            ctx = _materialize_for_device(ctx, step_device)
            step_kwargs = _materialize_for_device(dict(generation_kwargs), step_device)
            ctx = model.generation_graph.step(modules, ctx, trace=trace, generation_kwargs=step_kwargs)
            ctx = _materialize_for_device(ctx, torch.device("cpu"))
            if hasattr(model, "module_runner"):
                model.module_runner.release_resident()
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


def _collect_train_forward_observations(
    forward_result: Mapping[str, Any],
    whitelist: Mapping[tuple[str, str], frozenset[str]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    observations: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for node, out in forward_result["outputs"].items():
        record_module_output(observations, whitelist, state="train", node=node, out=out)
    return observations


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


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = [
    "InferModulePolicy",
    "LazyModuleRunner",
    "ModuleNode",
    "run_infer_module_fsm",
    "run_module_nodes",
    "run_v2_infer_module",
    "run_v2_train_module",
    "run_v2_train_module_batch",
    "run_v2_train_nodes",
]
