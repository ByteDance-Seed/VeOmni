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
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Mapping

import torch

from .....checkpoint import layout
from .....distributed.clip_grad_norm import veomni_omni_model_clip_grad_norm
from .....distributed.parallel_state import use_parallel_state
from .....utils.logging import get_logger
from ...graphs.training_graph import TrainingGraph
from ...mixins import MetricMeterMixin, MetricMeterResult
from ...modeling_omni import OmniModel
from ...utils.graph_profiler import GraphProfiler
from ...utils.hf_layout import HFSource, copy_source_non_weight_files, save_hf_source_checkpoint
from ..utils.executor import TrainNodeRunner, execute_generation_node
from ..utils.modules import save_module_subdirectory


if TYPE_CHECKING:
    from .....arguments.omni_arguments_types import OmniGraphProfileArguments, OmniTrainingArguments
    from .....trainer.callbacks import TrainerState
    from ..omni_module.omni_module_runtime import ModuleRuntime
    from .omni_model_config import OmniModelRuntimeArguments


logger = get_logger(__name__)


class MultiOptimizer:
    """Thin proxy over ``{module_name: torch.optim.Optimizer}``.

    Exposes the minimal :class:`torch.optim.Optimizer` surface the logging
    callbacks read (``param_groups``) and the train loop drives
    (``step`` / ``zero_grad``).  Optimizer state is checkpointed per module by
    each :class:`ModuleRuntime`'s own DCP manager, so no ``state_dict`` is needed
    here.
    """

    def __init__(self, optimizers: dict[str, torch.optim.Optimizer]):
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for opt in self.optimizers.values():
            groups.extend(opt.param_groups)
        return groups

    def step(self) -> None:
        for opt in self.optimizers.values():
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        # veomni.optim.MultiOptimizer (FSDP2 +ExtraParallel) has ``zero_grad()`` with no args
        # plain torch optimizers default to ``set_to_none=True``.
        for opt in self.optimizers.values():
            opt.zero_grad()


class MultiLRScheduler:
    """Thin proxy over ``{module_name: LRScheduler}`` (step-all / lr-read)."""

    def __init__(self, schedulers: dict[str, Any]):
        self.schedulers = schedulers

    def step(self) -> None:
        for sched in self.schedulers.values():
            sched.step()

    def get_last_lr(self) -> list[float]:
        lrs: list[float] = []
        for sched in self.schedulers.values():
            lrs.extend(sched.get_last_lr())
        return lrs or [0.0]

    def state_dict(self) -> dict[str, Any]:
        return {name: sched.state_dict() for name, sched in self.schedulers.items()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name, sched in self.schedulers.items():
            if name in state:
                sched.load_state_dict(state[name])


def _scoped_no_split_modules(module_runtimes: Mapping[str, ModuleRuntime]) -> list[str]:
    """Prefix each child's ``_no_split_modules`` with that child's name.

    A bare class name is ambiguous once the children share one FSDP tree:
    ``Embedding`` is a valid unit under a child that reads the weight only
    through the embedding's own forward, but not under
    the VQVAE, whose ``JanusVQVAEVectorQuantizer.forward`` reads
    ``self.embedding.weight`` directly and so never fires the embedding's own
    unshard hook. ``{child}.{ClassName}`` keeps each child's list applying to
    that child only (see ``_is_fsdp_wrap_target``). Child ``basic_modules``
    overlays are scoped the same way.

    OmniModule class names are deliberately not added: leftover params unshard
    on :meth:`OmniModel.forward`, the FSDP root entry.
    """
    scoped: list[str] = []
    for name, runtime in module_runtimes.items():
        for cls_name in getattr(runtime.model, "_no_split_modules", None) or []:
            scoped.append(f"{name}.{cls_name}")
        for cls_name in runtime.args.basic_modules or []:
            scoped.append(f"{name}.{cls_name}")
    return list(dict.fromkeys(scoped))


def _training_graph_methods(training_graph: list[dict]) -> dict[str, frozenset[str]]:
    """The methods the training graph calls on each module, e.g. ``{"bagel_vae": {"online_process"}}``."""
    if not training_graph:
        return {}
    methods: dict[str, set[str]] = {}
    for node in TrainingGraph(training_graph).active_nodes():
        methods.setdefault(node.module, set()).add(node.method)
    return {module: frozenset(called) for module, called in methods.items()}


_OFFLINE_ENDPOINT_OF_TASK = {"offline_embedding": "offline_encode", "offline_training": "online_process"}


def _reject_graph_that_mismatches_training_task(
    graph_methods: Mapping[str, frozenset[str]], train_args: OmniTrainingArguments
) -> None:
    """A run's training graph must call the offline endpoint of its ``training_task`` and no other."""
    task = train_args.training_task
    expected = _OFFLINE_ENDPOINT_OF_TASK.get(task)
    nodes_by_endpoint = {
        endpoint: sorted(f"{module}.{endpoint}" for module, called in graph_methods.items() if endpoint in called)
        for endpoint in _OFFLINE_ENDPOINT_OF_TASK.values()
    }
    if expected is not None and not nodes_by_endpoint[expected]:
        raise ValueError(
            f"train.training_task={task!r} needs a training graph that calls `<module>.{expected}`, "
            "but this graph calls it on no module. Use the graph YAML written for this task."
        )
    for endpoint, nodes in nodes_by_endpoint.items():
        if endpoint != expected and nodes:
            owner = next(t for t, e in _OFFLINE_ENDPOINT_OF_TASK.items() if e == endpoint)
            raise ValueError(
                f"The training graph calls {nodes}, which only train.training_task={owner!r} runs; "
                f"this run has train.training_task={task!r}."
            )


def _reject_lora_that_matched_nothing(
    module_runtimes: Mapping[str, ModuleRuntime], train_args: OmniTrainingArguments | None = None
) -> None:
    """Fail a LoRA run that left the composed model with nothing to train.

    A single module whose targets missed is normal — ``ModuleRuntime`` already
    logs and stays frozen. A sibling doing full-parameter SFT still trains.
    Raise only when LoRA was requested and **every** module is frozen, which
    would look like a healthy run whose loss never moves.

    A ``train.training_task='offline_embedding'`` run is exempt: it trains
    nothing by design.
    """
    if train_args is not None and train_args.training_task == "offline_embedding":
        return

    requested = [name for name, runtime in module_runtimes.items() if bool(runtime.args.lora_config)]
    if not requested:
        return
    if any(runtime.has_trainable_parameters for runtime in module_runtimes.values()):
        return
    raise ValueError(
        f"LoRA was configured for module(s) {requested} but produced no trainable "
        "adapters, and no other module has trainable parameters. "
        "Select at least one Linear or MoE target that a module actually declares."
    )


class OmniModelRuntime:
    """VeOmni model handle for one composed :class:`OmniModel`.

    There are exactly two ways to build a SeedOmni model:

    * **Bare HF** — :meth:`OmniModel.from_config` / :meth:`OmniModel.from_pretrained`
      over a checkpoint root holding ``config.json``. Every sub-module is a plain
      ``PreTrainedModel`` and the composed model is a plain ``PreTrainedModel``;
      no VeOmni infrastructure is involved (eager single-process inference).
    * **VeOmni** — :func:`build_omni_model_runtime`. Every sub-module is owned by a
      :class:`~veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime.ModuleRuntime`
      (FSDP2/DDP wrap, weight load, optimizer, checkpoint manager) and the
      composed model is *this* class. ``OmniTrainer.model`` /
      ``OmniInferencer.model`` hold it. Training enters the wrapped
      :class:`OmniModel` so FSDP root hooks fire; this class supplies
      ParallelState scoping, graph tracing and metric metering.

    The two lines stay apart: :class:`OmniModel` holds each runtime's bare
    module (:attr:`ModuleRuntime.omni_module`), and whatever a runtime wrapped
    it in (DDP, LoRA) stays on the runtime. Nodes and generation call the
    module through its runtime (:meth:`get_module`), so a DDP module still
    syncs its gradients.

    Both are used through the single ``self.model`` handle on the trainer /
    inferencer. APIs that need no wrapper handling are forwarded via
    :meth:`__getattr__` (``config``, ``modules_dict``, :meth:`OmniModel.reset`,
    :meth:`OmniModel.named_omni_modules`, …).
    :meth:`forward` enters the (possibly FSDP-wrapped) :class:`OmniModel` so
    root leftover params unshard, then :meth:`OmniModel.forward` runs the
    training graph. :meth:`get_module`, :meth:`clip_grad_norm`, :meth:`generate` and
    :meth:`save_pretrained` stay on this wrapper, and so do
    :attr:`optimizer` / :attr:`lr_scheduler`, which step every module's own.

    :attr:`hf_source` is set when the modules were read from an upstream HF
    checkpoint; exports then write that checkpoint's layout.
    """

    hf_source: HFSource | None = None

    def __init__(
        self,
        model: OmniModel,
        *,
        module_runtimes: Mapping[str, ModuleRuntime] | None = None,
        omni_model_runtime_args: OmniModelRuntimeArguments | None = None,
        train_args: OmniTrainingArguments | None = None,
        hf_source: HFSource | None = None,
    ) -> None:
        self.model = model
        self.module_runtimes = dict(module_runtimes or {})
        self.omni_model_runtime_args = omni_model_runtime_args
        self.train_args = train_args
        self.hf_source = hf_source
        self.optimizer: MultiOptimizer | None = None
        self.lr_scheduler: MultiLRScheduler | None = None
        self._step_profiler: GraphProfiler | None = None

    def _build_optimizer(self) -> None:
        """Wrap the trainable modules' optimizers in one :class:`MultiOptimizer`.

        Stays ``None`` when every module is frozen, like a frozen module's own.
        """
        optimizers = {
            name: module_runtime.optimizer
            for name, module_runtime in self.module_runtimes.items()
            if module_runtime.optimizer is not None
        }
        self.optimizer = MultiOptimizer(optimizers) if optimizers else None
        logger.info_rank0(f"OmniModelRuntime: wired {len(optimizers)} optimizer(s): {list(optimizers)}.")

    def _build_lr_scheduler(self, total_steps: int) -> None:
        """Build every module's lr-scheduler over ``total_steps`` and wrap them.

        A fully-frozen module builds none, so it contributes no entry; with no
        entry at all the wrapper stays ``None``.
        """
        for module_runtime in self.module_runtimes.values():
            module_runtime._build_lr_scheduler(total_steps)
        lr_schedulers = {
            name: module_runtime.lr_scheduler
            for name, module_runtime in self.module_runtimes.items()
            if module_runtime.lr_scheduler is not None
        }
        self.lr_scheduler = MultiLRScheduler(lr_schedulers) if lr_schedulers else None

    def _parallelize_composed_model(self, *, for_inference: bool = False) -> None:
        """``fully_shard`` the composed :class:`OmniModel` when ``fsdp_scope='model'``.

        Each :class:`ModuleRuntime` has already meta-initialized, frozen, and
        bound assets, but skipped its own wrap. One FSDP2 tree over the parent:
        wrap each child's ``_no_split_modules`` **scoped to that child**
        (``janus_llama.LlamaDecoderLayer``, ``janus_text_encoder.Embedding``, …),
        then ``fully_shard`` the OmniModel root.

        HF ``post_init``'s union is replaced rather than reused: it is a flat
        class-name set, so the text encoder's ``Embedding`` would also match the
        VQ codebook (see :func:`_scoped_no_split_modules`). Leftover params
        (aligner, final norm, anything not in a nested unit) unshard on
        :meth:`OmniModel.forward`.
        """
        from ..omni_module.omni_module_runtime import composed_model_owns_wrap

        args = self.omni_model_runtime_args
        acc = args.accelerator
        if not composed_model_owns_wrap(acc):
            return

        modules = self.module_runtimes
        wrap_units = _scoped_no_split_modules(modules)
        self.model._no_split_modules = wrap_units

        kwargs: dict[str, Any] = {
            "cpu_load_param_name": None,
            "module_skip_hf_weight_load": {name: runtime.skip_hf_weight_load for name, runtime in modules.items()},
            "module_is_peft_model": {name: bool(runtime.args.lora_config) for name, runtime in modules.items()},
            "module_adapter_path": {
                name: (runtime.args.lora_config or {}).get("lora_adapter") for name, runtime in modules.items()
            },
        }
        if any(kwargs["module_is_peft_model"].values()):
            kwargs["is_peft_model"] = True

        from .....distributed.torch_compile import CompileConfig
        from .....distributed.torch_parallelize import build_parallelize_model

        compile_config = CompileConfig(
            **{field.name: getattr(acc.torch_compile, field.name) for field in fields(CompileConfig)}
        )
        opt = args.optimizer
        muon_expert_zero_comm = bool(opt) and opt.type == "muon" and opt.muon_expert_zero_comm
        weights_path = {name: runtime.weights_path for name, runtime in modules.items()}

        logger.info_rank0(f"OmniModelRuntime: wrapping composed OmniModel (fsdp_scope='model') over {list(modules)}.")
        # ``OmniTrainer._setup`` registered ``base`` from the same top-level accelerator.
        with use_parallel_state("base"):
            self.model = build_parallelize_model(
                self.model,
                init_device=acc.init_device,
                weights_path=weights_path,
                should_skip_hf_weight_load=False,
                enable_reshard_after_forward=acc.fsdp_config.reshard_after_forward,
                mixed_precision=acc.fsdp_config.mixed_precision,
                enable_gradient_checkpointing=False,
                basic_modules=wrap_units,
                enable_reentrant=acc.gradient_checkpointing.enable_reentrant,
                early_stop=acc.gradient_checkpointing.early_stop,
                enable_forward_prefetch=acc.fsdp_config.forward_prefetch,
                enable_fsdp_offload=acc.fsdp_config.offload,
                fsdp_offload_pin_memory=acc.fsdp_config.offload_pin_memory,
                broadcast_model_weights_from_rank0=args.broadcast_model_weights_from_rank0,
                ep_sharded_stream_load=args.ep_sharded_stream_load,
                max_load_broadcast_size=acc.fsdp_config.max_load_broadcast_size,
                muon_expert_zero_comm=muon_expert_zero_comm,
                compile_config=compile_config,
                **kwargs,
            )

        if not for_inference:
            for runtime in modules.values():
                runtime.build_after_omni_model_wrap()

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
        """Scope ``module_name``'s :class:`ParallelState` as current, as its runtime defines it."""
        runtime = self.module_runtimes.get(module_name)
        return runtime._scoped() if runtime is not None else nullcontext()

    def forward(
        self,
        batch: dict[str, Any],
        *,
        profiler: GraphProfiler | None = None,
    ) -> dict[str, Any]:
        """Run the training DAG through the composed model's FSDP ``forward``.

        ``profiler`` defaults to :attr:`step_profiler` — the trainer only has to
        open the trace window (:meth:`OmniTrainer.init_graph_profile`) once per step.

        Must go through ``self.model(batch)`` (``nn.Module.__call__``), not
        ``self.model.forward(...)``, so FSDP2 root pre-forward hooks unshard
        leftover params before the graph calls each child.

        The model walks the graph; :class:`TrainNodeRunner` — passed in as
        ``node_runner``, fresh per step because it labels node transitions — is
        what makes each node VeOmni-aware (unwrap, ParallelState scope, profile).
        """
        profiler = profiler if profiler is not None else self._step_profiler
        runner = TrainNodeRunner(profiler=profiler, scope_fn=self.module_context)

        def run_node(module: Any, node: Any, batch: dict[str, Any]) -> None:
            runner(self.get_module(node.module), node, batch)

        return self.model(batch, node_runner=run_node)

    def clip_grad_norm(self) -> float:
        """Clip every module's grads and return the whole-model norm.

        Each :class:`ModuleRuntime` clips its own params against **its own**
        ``optimizer.max_grad_norm`` (every OmniModule carries its own optimizer
        config); the returned norm is the L2 combination of the pre-clip module
        norms. Takes no threshold, since no single one applies. With no module
        runtimes the norm is 0.0.
        """
        if not self.module_runtimes:
            return 0.0
        optimizer = self.omni_model_runtime_args.optimizer
        return veomni_omni_model_clip_grad_norm(
            self.module_runtimes,
            optimizer.max_grad_norm,
            grad_clip_scope=optimizer.grad_clip_scope,
        )

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
        modules = {name: self.get_module(name) for name in model._module_names}
        model.generation_graph.validate_modules({name: model.get_module(name) for name in model._module_names})
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
                # Per node, as in ``OmniModel.generate``: a later node emitting
                # ``generated`` would overwrite this one's in the shared ``ctx``.
                self._collect_generated(ctx, profiler, label="generated")
            total_steps += 1
            fired = model.generation_graph.maybe_transition(ctx)
            if fired is not None and profiler is not None:
                profiler.record(f"transition: {fired.from_state} -> {fired.to_state} [{fired.condition}]")

        model._emit_progress(total_steps)

        if not model.generation_graph.is_done():
            for name, raw in self.named_omni_modules():
                out = raw.finalize(ctx=ctx)
                if not isinstance(out, dict):
                    raise TypeError(f"{type(raw).__name__}.finalize must return a dict, got {type(out).__name__}.")
                ctx.update(out)
                self._collect_generated(ctx, profiler, label=f"finalize:{name} | generated")

        return list(model._generated)

    def _collect_generated(self, ctx: dict[str, Any], profiler: GraphProfiler | None, *, label: str) -> None:
        """Drain ``ctx["generated"]`` via :meth:`OmniModel._collect_generated`, tracing what it kept."""
        generated = self.model._generated
        before = len(generated)
        self.model._collect_generated(ctx)
        if profiler is not None and len(generated) > before:
            profiler.record(f"{label}:{generated[-1]['type']}")

    def get_module(self, name: str) -> Any:
        """Module ``name`` as the graph runs it: wrapped as its runtime wrapped it.

        :meth:`OmniModel.get_module` returns the bare module the composed model
        holds; a DDP / LoRA wrapper lives on the module's runtime instead.
        """
        module_runtime = self.module_runtimes.get(name)
        return module_runtime.model if module_runtime is not None else self.model.get_module(name)

    def save_pretrained(self, save_directory: str | os.PathLike, **kwargs: Any) -> None:
        """Save the omni-root HF layout (config + graphs + module sidecars).

        Each module's sidecars are its runtime's :attr:`ModuleRuntime.model_assets`. Weights are
        written from the main process alone, so weight export only suits
        unsharded modules (eager, DDP); it strips DDP but keeps a LoRA wrapper,
        whose ``save_pretrained`` writes the adapter. Training exports sharded
        weights per module via ``OmniTrainer.save_hf_or_lora`` and calls this
        with ``save_module_weights=False``.
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
            module_runtime = self.module_runtimes[name]
            save_module_subdirectory(
                name,
                module_runtime.model,
                save_directory,
                assets=module_runtime.model_assets,
                save_module_weights=save_module_weights,
                **module_save_kwargs,
            )

        model.config.save_pretrained(save_directory)

    def save_model_assets(self) -> None:
        """Write the omni-root HF layout (config + graphs + module sidecars, no weights).

        Under an HF ``model_path`` the export is the source's own layout instead,
        so the assets are the source's non-weight files.
        """
        import torch.distributed as dist

        if self.train_args is None:
            raise ValueError(
                "OmniModelRuntime.save_model_assets needs a training runtime (built with train_args=...)."
            )
        if self.train_args.global_rank == 0:
            save_directory = self.train_args.checkpoint.model_assets_dir
            if self.hf_source is not None:
                copy_source_non_weight_files(self.hf_source.path, save_directory)
            else:
                self.save_pretrained(save_directory, save_module_weights=False)
            logger.info_rank0(f"OmniModelRuntime: saved OmniModel assets to {save_directory}.")
        if dist.is_initialized():
            dist.barrier()

    def metric_meter_collect(self) -> dict[str, MetricMeterResult]:
        """Drain each metered module's ``(theoretical_flops, seqlens)`` for this step.

        Modules without :class:`MetricMeterMixin` contribute nothing. The keys
        must be identical on every rank: :class:`~veomni.utils.omni_helper.OmniEnvironMeter`
        packs one value per key into a single all-reduce.
        """
        return {
            name: module_runtime.omni_module.metric_meter_collect()
            for name, module_runtime in self.module_runtimes.items()
            if isinstance(module_runtime.omni_module, MetricMeterMixin)
        }

    def load(self) -> None:
        """Resume every module's DCP checkpoint (no-op for frozen / unconfigured modules)."""
        for module_runtime in self.module_runtimes.values():
            module_runtime.load()

    def save_dcp(self, state: TrainerState) -> None:
        """Write every module's distributed checkpoint (train resume)."""
        for module_runtime in self.module_runtimes.values():
            module_runtime.save_dcp(state)

    def save_hf_or_lora(self, state: TrainerState, stage: str = "step_end") -> None:
        """Export every module's HF weights / LoRA adapter.

        Under an HF ``model_path`` the full-parameter modules export together as
        one checkpoint in the source's own layout (``hf_ckpt/``), which loads
        wherever the source does; LoRA adapters still export per module.
        """
        if self.hf_source is None:
            for module_runtime in self.module_runtimes.values():
                module_runtime.save_hf_or_lora(state, stage=stage)
            return
        self._save_hf_source_layout(state, stage)

    def _save_hf_source_layout(self, state: TrainerState, stage: str) -> None:
        """One merged export in the source layout.

        A frozen module (no checkpoint manager) and a LoRA module's base still
        hold the source weights, so :func:`save_hf_source_checkpoint` copies
        their tensors from the source rather than gathering them.
        """
        trained: dict[str, ModuleRuntime] = {}
        for name, module_runtime in self.module_runtimes.items():
            checkpoint = module_runtime.checkpoint
            if checkpoint is None:
                continue
            if checkpoint.trainable_only:
                checkpoint.save_lora(state, stage=stage)
            else:
                trained[name] = module_runtime
        if not trained:
            return
        for module_runtime in trained.values():
            module_runtime.checkpoint._prepare_export(state, stage)
        save_path = layout.hf_export_dir(next(iter(trained.values())).checkpoint.step_dir(state))
        save_hf_source_checkpoint(
            self.hf_source,
            {name: (rt.model, rt.checkpoint.parallel_state) for name, rt in trained.items()},
            save_path,
        )

    def wait_for_pending_save(self) -> None:
        """Drain every module's in-flight async checkpoint writes."""
        for module_runtime in self.module_runtimes.values():
            module_runtime.wait_for_pending_save()


def build_omni_model_runtime(
    omni_model_runtime_args: OmniModelRuntimeArguments,
    *,
    train_args: OmniTrainingArguments | None = None,
    for_inference: bool = False,
) -> OmniModelRuntime:
    """Compose a VeOmni-managed model from a resolved :class:`OmniModelRuntimeArguments`.

    Args:
        omni_model_runtime_args: The resolved model section (modules, graphs, global accelerator).
        train_args: The job's ``train:`` config section (``OmniArguments.train``), not a
            train/infer switch — ``for_inference`` is that. ``None`` for inference builds.
            Forwarded unchanged to every :class:`ModuleRuntime`, which reads
            ``training_task`` and the shared checkpoint ``save_path``/``output_dir``/``load_path``.
            Each module also gets the methods the training graph calls on it, which decide
            whether an ``offline_training`` run builds it on meta. The graph must call the
            offline endpoint of ``training_task`` and no other.
        for_inference: Build for generation, which skips the optimizer and the training-only checks.
    """
    from ..omni_module.omni_module_runtime import build_omni_module_runtime

    omni_config = omni_model_runtime_args.to_hf_config()
    omni_config.load_checkpoint_sidecars(omni_model_runtime_args.resolved_model_path)
    module_runtime_args = omni_model_runtime_args.modules
    graph_methods = {} if for_inference else _training_graph_methods(omni_config.training_graph)
    if not for_inference and train_args is not None:
        _reject_graph_that_mismatches_training_task(graph_methods, train_args)
    module_runtimes: dict[str, ModuleRuntime] = {}
    for name in omni_config.module_names:
        module_args = module_runtime_args[name]
        module_runtime = build_omni_module_runtime(
            module_args,
            module_name=name,
            module_config=omni_config._module_configs[name],
            train_args=train_args,
            training_graph_methods=graph_methods.get(name, frozenset()),
            for_inference=for_inference,
            global_accelerator=omni_model_runtime_args.accelerator,
            hf_source=omni_config._hf_source,
        )
        module_runtimes[name] = module_runtime
        logger.info_rank0(f"OmniModelRuntime: built ModuleRuntime '{name}' from {module_runtime.weights_path}")

    logger.info_rank0(
        f"OmniModelRuntime: composed OmniModel with {len(module_runtimes)} module(s) ({list(module_runtimes)})."
    )
    if not for_inference:
        _reject_lora_that_matched_nothing(module_runtimes, train_args)
    runtime = OmniModelRuntime(
        OmniModel(omni_config, {name: rt.omni_module for name, rt in module_runtimes.items()}),
        module_runtimes=module_runtimes,
        omni_model_runtime_args=omni_model_runtime_args,
        train_args=train_args,
        hf_source=omni_config._hf_source,
    )
    runtime._parallelize_composed_model(for_inference=for_inference)
    if not for_inference:
        runtime._build_optimizer()
    return runtime


__all__ = ["MultiLRScheduler", "MultiOptimizer", "OmniModelRuntime", "build_omni_model_runtime"]
