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
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping

import torch.nn as nn

from ....distributed.parallel_state import is_parallel_state_registered, use_parallel_state
from ....utils.logging import get_logger
from ..graphs.base import NodeDef
from ..mixins.metric_meter_mixin import MetricMeterResult
from ..modeling_omni import OmniModel
from ..utils.graph_profiler import GraphProfiler
from .executor import execute_generation_node, execute_train_node
from .utils import iter_named_omni_modules, save_module_subdirectory


if TYPE_CHECKING:
    from ....arguments import OmniGraphProfileArguments
    from ....omni_arguments.arguments_types import OmniModelRuntimeArguments
    from ....trainer.callbacks import TrainerState
    from .module_runtime import ModuleRuntime


logger = get_logger(__name__)

# Must match the ``_loss`` key every OmniModule's ``post_forward`` emits.
_LOSS_KEY = "_loss"


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
    inferencer. APIs that need no wrapper handling are forwarded via
    :meth:`__getattr__` (``config``, ``modules_dict``, …).
    :meth:`forward`, :meth:`generate`, :meth:`save_pretrained`, :meth:`reset`,
    and :meth:`named_omni_modules` are implemented here instead.
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
        train: Any = None,
        for_inference: bool = False,
    ) -> OmniModelRuntime:
        """Compose a VeOmni-managed model from a resolved :class:`OmniModelRuntimeArguments`.

        ``train`` is the global :class:`~....omni_arguments.arguments_types.OmniTrainingArguments`
        (unset for inference) — forwarded to every :class:`ModuleRuntime` so its
        checkpoint manager can resolve the shared ``save_path``/``output_dir``/``load_path``.
        """
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
        """Forward undshadowed :class:`OmniModel` APIs."""
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
            for name, raw in self.named_omni_modules():
                out = raw.finalize(ctx=ctx)
                if not isinstance(out, dict):
                    raise TypeError(f"{type(raw).__name__}.finalize must return a dict, got {type(out).__name__}.")
                generated = out.pop("generated", None)
                model._append_generated(generated)
                if profiler is not None and generated is not None:
                    profiler.record(f"finalize:{name} | generated:{generated['type']}")

        return list(model._generated)

    def named_omni_modules(self) -> Iterator[tuple[str, Any]]:
        """Yield ``(name, BaseMixin)`` for every graph participant (unwraps wrappers)."""
        yield from iter_named_omni_modules(self.model._module_names, self.model.modules_dict)

    def reset(self) -> None:
        """Clear per-conversation inference runtime state (unwraps wrapped modules)."""
        model = self.model
        model.generation_graph.reset()
        model._generated.clear()
        for _, module in self.named_omni_modules():
            module.reset_global_inference_state()

    def save_pretrained(self, save_directory: str | os.PathLike, **kwargs: Any) -> None:
        """Save the omni-root HF layout (config + graphs + module sidecars).

        Unwraps DDP / LoRA wrappers when writing per-module assets; weight export
        still calls each wrapped module's ``save_pretrained`` so FSDP/DDP hooks run.
        """
        import torch.distributed as dist

        is_main_process = kwargs.pop("is_main_process", None)
        if is_main_process is None:
            is_main_process = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        if not is_main_process:
            return

        save_module_weights = kwargs.pop("save_module_weights", True)
        safe_serialization = kwargs.pop("safe_serialization", True)
        max_shard_size = kwargs.pop("max_shard_size", "5GB")

        save_directory = str(save_directory)
        os.makedirs(save_directory, exist_ok=True)

        model = self.model
        module_save_kwargs = {
            **kwargs,
            "safe_serialization": safe_serialization,
            "max_shard_size": max_shard_size,
        }
        for name in model._module_names:
            module = model.modules_dict[name]
            save_module_subdirectory(
                model.config,
                name,
                module,
                save_directory,
                save_module_weights=save_module_weights,
                **module_save_kwargs,
            )

        model.config.save_pretrained(save_directory)

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


def _sum_losses(losses: dict[str, Any]) -> Any | None:
    if not losses:
        return None
    it = iter(losses.values())
    total = next(it)
    for v in it:
        total = total + v
    return total


__all__ = ["OmniModelRuntime"]
