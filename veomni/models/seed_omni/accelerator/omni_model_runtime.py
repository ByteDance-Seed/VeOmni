# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Composite runtime over a clean :class:`~veomni.models.seed_omni.modeling_omni.OmniModel`."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import torch.nn as nn

from ....distributed.parallel_state import is_parallel_state_registered, use_parallel_state
from ....models.seed_omni.accelerator.executor import execute_generation_node, execute_train_node
from ....models.seed_omni.graphs.graph import NodeDef
from ....models.seed_omni.graphs.profiling import GraphProfiler
from ....models.seed_omni.mixins.metric_meter_mixin import MetricMeterResult
from ....models.seed_omni.modeling_omni import _LOSS_KEY, OmniModel, _sum_losses
from ....utils.logging import get_logger


if TYPE_CHECKING:
    from ....arguments import OmniGraphProfileArguments
    from ....omni_arguments.arguments_types import OmniModelRuntimeArguments, OmniTrainingArguments
    from ....trainer.callbacks import TrainerState
    from .module_runtime import ModuleRuntime


logger = get_logger(__name__)


class OmniModelRuntime:
    """VeOmni model handle for one composed :class:`OmniModel`.

    There are exactly two ways to build a SeedOmni model:

    * **Bare HF** — :meth:`OmniModel.from_config` / :meth:`OmniModel.from_pretrained`
      over a checkpoint root holding ``config.json``. Every sub-module is a plain
      ``PreTrainedModel`` and the composed model is a plain ``PreTrainedModel``;
      no VeOmni infrastructure is involved (eager single-process inference).
    * **VeOmni** — :meth:`from_model_runtime`. Every sub-module is owned by a
      :class:`~veomni.models.seed_omni.accelerator.module_runtime.ModuleRuntime`
      (FSDP2/DDP wrap, weight load, optimizer, checkpoint manager) and the
      composed model is *this* class. ``OmniTrainer.model`` /
      ``OmniInferencer.model`` hold it; training requires it because only this
      class runs the graph with ParallelState scoping, graph tracing and metric
      metering.

    Both are used through the single ``self.model`` handle on the trainer /
    inferencer. Everything not defined here is forwarded to the composed
    :class:`OmniModel` (see :meth:`__getattr__`), so ``config``,
    ``modules_dict``, ``save_pretrained`` and the ``nn.Module`` reads shared
    callbacks rely on work the same either way; :meth:`forward` and
    :meth:`generate` shadow the eager, profiler-free versions.
    """

    def __init__(
        self,
        model: OmniModel,
        *,
        module_runtimes: Mapping[str, ModuleRuntime] | None = None,
        module_parallel_state_names: Iterable[str] | None = None,
    ) -> None:
        self.model = model
        self.module_runtimes = dict(module_runtimes or {})
        self._module_parallel_state_names = set(module_parallel_state_names or ())
        self._step_profiler: GraphProfiler | None = None
        self._losses: dict[str, Any] = {}

    @classmethod
    def from_model_runtime(
        cls,
        model_runtime: OmniModelRuntimeArguments,
        *,
        train: OmniTrainingArguments,
        train_steps: int = -1,
        for_inference: bool = False,
    ) -> OmniModelRuntime:
        """Compose a VeOmni-managed model from a resolved :class:`OmniModelRuntimeArguments`."""
        from .module_runtime import ModuleRuntime

        omni_config = model_runtime.to_hf_config()
        module_runtime_args = model_runtime.modules
        module_runtimes: dict[str, ModuleRuntime] = {}
        modules: dict[str, nn.Module] = {}
        for name in omni_config.module_names:
            module_args = module_runtime_args[name]
            module_runtime = ModuleRuntime(
                module_args,
                module_name=name,
                train=train,
                train_steps=train_steps,
                for_inference=for_inference,
            )
            module_runtime.checkpoint_subfolder = omni_config.module_checkpoint_subfolder(name)
            module_runtimes[name] = module_runtime
            modules[name] = module_runtime.model
            logger.info_rank0(f"OmniModelRuntime: built ModuleRuntime '{name}' from {module_args.model_path}")

        logger.info_rank0(f"OmniModelRuntime: composed OmniModel with {len(modules)} module(s) ({list(modules)}).")
        return cls(
            OmniModel(omni_config, modules),
            module_runtimes=module_runtimes,
            module_parallel_state_names=[name for name in module_runtimes if is_parallel_state_registered(name)],
        )

    def __getattr__(self, name: str) -> Any:
        """Forward the composed :class:`OmniModel` surface for anything not defined here.

        Keeps ``trainer.model`` / ``inferencer.model`` a single handle whether it
        is the bare :class:`OmniModel` or this runtime: ``config``,
        ``modules_dict``, ``reset``, ``save_pretrained`` and the
        ``nn.Module`` reads shared callbacks use (e.g. ``modules()`` for the MoE
        router monitor) resolve identically.
        """
        try:
            model = object.__getattribute__(self, "model")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(model, name)

    @property
    def step_profiler(self) -> GraphProfiler | None:
        """Active graph profiler, or ``None`` outside a profile window.

        Consumed implicitly by :meth:`forward` / :meth:`generate` when the caller
        passes no explicit ``profiler``.
        """
        return self._step_profiler

    def begin_request_trace(self, profile: OmniGraphProfileArguments) -> GraphProfiler | None:
        """Start a graph profiler for one inference request.

        Unlike training there is no step window to gate on: inference always
        emits a per-request trace.
        """
        self._step_profiler = GraphProfiler.from_config(profile)
        return self._step_profiler

    def begin_step_trace(self, profile: OmniGraphProfileArguments) -> None:
        """Open this model's graph profiler for one training step.

        Scheduling (enabled flags / rank / step window) is owned by
        :class:`~veomni.trainer.callbacks.omni_callbacks.GraphProfileCallback`;
        this method always starts a profiler when called.
        """
        self._step_profiler = GraphProfiler.from_config(profile)

    def flush_step_trace(
        self,
        global_step: int,
        *,
        output_dir: str,
        rank: int,
        tag: str = "model",
    ) -> None:
        """Write this model's step graph profiler to disk and clear it.

        ``tag`` distinguishes composed handles when a trainer holds more than one
        (``model`` / ``student`` / ``teacher``) so they do not overwrite each other.
        """
        profiler = self._step_profiler
        if profiler is None:
            return
        trace_dir = os.path.join(output_dir, "graph_trace")
        os.makedirs(trace_dir, exist_ok=True)
        suffix = "" if tag == "model" else f"_{tag}"
        trace_path = os.path.join(trace_dir, f"step_{global_step:06d}_rank_{rank}{suffix}.txt")
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write("\n".join(profiler.save_records()) + "\n")
        logger.info_rank0(f"OmniModelRuntime[{tag}]: graph profile trace → {trace_path}")
        self._step_profiler = None

    def module_context(self, module_name: str):
        """Scope ``module_name``'s :class:`ParallelState` as current when registered."""
        if module_name in self._module_parallel_state_names:
            return use_parallel_state(module_name)
        return nullcontext()

    def forward(
        self,
        batch: dict[str, Any],
        *,
        profiler: GraphProfiler | None = None,
    ) -> dict[str, Any]:
        """Run the training DAG with VeOmni per-node execution.

        ``profiler`` defaults to :attr:`step_profiler` — the trainer only has to
        open the trace window (:meth:`OmniTrainer.init_graph_profile`) once per step.
        """
        profiler = profiler if profiler is not None else self._step_profiler
        model = self.model
        model.training_graph.reset()
        self._losses.clear()
        modules = model.modules_dict

        prev_node: NodeDef | None = None
        for node in model.training_graph.iter_nodes():
            if prev_node is not None and profiler is not None:
                profiler.record(f"transition: -> {node.name}")
            execute_train_node(
                modules,
                node,
                batch,
                profiler=profiler,
                scope_fn=self.module_context,
            )
            loss = batch.pop(_LOSS_KEY, None)
            if loss is not None:
                self._losses[node.name] = loss
                if profiler is not None:
                    profiler.record(f"loss:{node.name}")
            prev_node = node

        return {"loss": _sum_losses(self._losses), "losses": dict(self._losses)}

    def generate(
        self,
        request: dict[str, Any],
        generation_kwargs: dict[str, Any] | None = None,
        *,
        profiler: GraphProfiler | None = None,
    ) -> list[dict[str, Any]]:
        """Run the generation FSM with VeOmni per-node execution.

        Signature-compatible with :meth:`OmniModel.generate` so a caller holding
        either handle drives it the same way; ``profiler`` defaults to
        :attr:`step_profiler` (see :meth:`begin_request_trace`).

        ``request`` is mutated in place. Returns the collected generated artefacts.
        """
        profiler = profiler if profiler is not None else self._step_profiler
        model = self.model
        ctx: dict[str, Any] = request
        modules = model.modules_dict
        generation_kwargs = model.resolve_generation_kwargs(generation_kwargs)
        max_new_tokens = generation_kwargs.get("max_new_tokens", 2048)
        total_steps = 0

        while not model.generation_graph.is_done() and total_steps < max_new_tokens:
            model._emit_progress(total_steps)
            state_name = model.generation_graph.current_state_name
            for node in model.generation_graph.iter_nodes(ctx):
                execute_generation_node(
                    modules,
                    node,
                    ctx,
                    state_name=state_name,
                    generation_kwargs=generation_kwargs,
                    profiler=profiler,
                    scope_fn=self.module_context,
                )
            total_steps += 1
            generated = ctx.pop("generated", None)
            model._append_generated(generated)
            if profiler is not None and generated is not None:
                profiler.record(f"generated:{generated['type']}")
            fired = model.generation_graph.maybe_transition(ctx)
            if fired is not None and profiler is not None:
                profiler.record(f"transition: {fired.from_state} -> {fired.to_state} [{fired.condition}]")

        model._emit_progress(total_steps)

        if not model.generation_graph.is_done():
            for name, raw in model.named_omni_modules():
                out = raw.finalize(ctx=ctx)
                if not isinstance(out, dict):
                    raise TypeError(f"{type(raw).__name__}.finalize must return a dict, got {type(out).__name__}.")
                generated = out.pop("generated", None)
                model._append_generated(generated)
                if profiler is not None and generated is not None:
                    profiler.record(f"finalize:{name} | generated:{generated['type']}")

        return list(model._generated)

    def collect_step_metrics(self) -> dict[str, MetricMeterResult]:
        """Gather every metered module's ``(theoretical_flops, seqlens)`` on this model."""
        module_metrics: dict[str, MetricMeterResult] = {}
        for name, module_runtime in self.module_runtimes.items():
            result = module_runtime.collect_step_metrics()
            if result is not None:
                module_metrics[name] = result
        return module_metrics

    def load(self) -> None:
        """Resume every module's DCP checkpoint (no-op for frozen / unconfigured modules)."""
        for module_runtime in self.module_runtimes.values():
            module_runtime.load()

    def save_dcp(self, state: TrainerState) -> None:
        """Write every module's distributed checkpoint (train resume)."""
        for module_runtime in self.module_runtimes.values():
            module_runtime.save_dcp(state)

    def save_hf_or_lora(self, state: TrainerState) -> None:
        """Export every module's HF weights / LoRA adapter."""
        for module_runtime in self.module_runtimes.values():
            module_runtime.save_hf_or_lora(state)

    def save_pretrained(self, save_directory: str | os.PathLike, **kwargs: Any) -> None:
        """Save the omni-root HF layout (config + graphs + module sidecars)."""
        self.model.save_pretrained(save_directory, **kwargs)


__all__ = ["OmniModelRuntime"]
