"""
OmniModel — composable multi-modal model driven by config-specified graphs.

This file holds the **clean modeling definition** — training graph via
:meth:`OmniModel.forward`, FSM inference via :meth:`OmniModel.generate`, and
checkpoint compose/load/save.  It must import nothing from VeOmni's runtime
(``accelerator`` / ``distributed`` / trainer), at module scope or inside a
function, so this modeling can be lifted into another framework as-is and so
HF ``from_pretrained`` / ``from_config`` keeps working for eager
single-process inference. ``tests/seed_omni/test_graph.py`` asserts this.

``forward`` is the FSDP2 root entry: leftover params unshard on ``__call__``,
then the training graph runs each child eagerly, which is correct for an
unwrapped single-process model.

Architecture
------------
``OmniModel`` carries:

* sub-modules — each graph participant is a direct attribute from the registry.
* ``training_graph`` — :class:`TrainingGraph` (DAG), or ``None`` until one is loaded.
* ``generation_graph`` — :class:`GenerationGraph` (FSM), or ``None`` until one is loaded.

Inference
---------
``generate(request, generation_kwargs)`` loops the FSM.  Each node is optional
``pre_generate`` → endpoint → optional ``post_generate``, mirroring the training
walk.  Stop when ``is_done()`` or ``max_new_tokens`` is reached.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterator, Mapping

import torch.distributed as dist
import torch.nn as nn
from transformers import PreTrainedModel

from ...utils import helper  # VeOmni shared logger (rank-0 helpers); not seed_omni-local.
from .configuration_omni import (
    DEFAULT_GENERATION_GRAPH_FILE,
    DEFAULT_TRAINING_GRAPH_FILE,
    OmniConfig,
    pop_omni_kwargs,
)
from .graphs.base import NodeDef
from .graphs.generation_graph import GenerationGraph
from .graphs.training_graph import TrainingGraph
from .modules import OMNI_MODEL_REGISTRY, PretrainedOmniModule
from .modules.module_configuration_base import OmniModuleConfig


logger = helper.create_logger(__name__)

# Must match the ``_loss`` key every OmniModule's ``post_forward`` emits.
LOSS_KEY = "_loss"

# HF hub kwargs forwarded to :meth:`OmniConfig.from_pretrained`.
_CONFIG_LOAD_KWARG_NAMES = frozenset(
    {
        "cache_dir",
        "force_download",
        "local_files_only",
        "proxies",
        "resume_download",
        "revision",
        "subfolder",
        "token",
        "trust_remote_code",
        "mirror",
        "_from_pipeline",
    }
)

# Top-level :class:`OmniConfig` fields overridable at ``OmniModel.from_pretrained`` time.
_OMNI_CONFIG_OVERRIDE_KEYS = frozenset(
    {
        "train_type",
        "infer_type",
        "generation_kwargs",
        "training_graphs",
        "generation_graphs",
        "_module_entries",
    }
)

# What ``PreTrainedModel._from_config`` pops for itself; anything else it is
# handed reaches the model's ``__init__``.
_FROM_CONFIG_KWARG_NAMES = frozenset(
    {
        "dtype",
        "torch_dtype",
        "attn_implementation",
        "experts_implementation",
    }
)


class OmniModel(PreTrainedModel):
    """Pure SeedOmni modeling runtime over already-built sub-modules.

    Parameters
    ----------
    config:
        :class:`OmniConfig` with ``_module_entries`` populated. Graphs are optional:
        a convert that only split modules leaves them empty, and
        :meth:`forward` / :meth:`generate` error until a DAG / FSM is supplied
        (checkpoint sidecar or ``training_graphs`` / ``generation_graphs``
        override). The FSM bound here is the one ``config.infer_type``
        selects, so switching scenario means rebuilding.
    modules:
        ``{module_name: PretrainedOmniModule}`` — the graph participants. Every
        participant must be a :class:`PretrainedOmniModule`: the save path needs
        its ``config`` / ``save_pretrained``, and a graph endpoint is resolved
        against the class's own methods.
    """

    config_class = OmniConfig
    base_model_prefix = "omni"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def __init__(self, config: OmniConfig, modules: Mapping[str, PretrainedOmniModule]):
        super().__init__(config)
        self.config = config

        self._module_names: list[str] = list(config.module_names)
        # The single enforcement point for the participant contract: everything
        # downstream (graph walk, ``get_module``, save) may then assume it.
        for name in self._module_names:
            module = modules[name]
            if not isinstance(module, PretrainedOmniModule):
                raise TypeError(
                    f"OmniModel: sub-module '{name}' is a {type(module).__name__}; "
                    "every graph participant must be a PretrainedOmniModule."
                )
            self.add_module(name, module)
            # ``from_pretrained`` already stored the loaded config. A model built
            # from live modules still has to save those configs.
            module_config = module.config
            if not isinstance(module_config, OmniModuleConfig):
                raise TypeError(
                    f"OmniModel: sub-module '{name}' config is a {type(module_config).__name__}, "
                    "expected a OmniModuleConfig."
                )
            self.config._module_configs.setdefault(name, module_config)

        # ``PreTrainedModel.post_init`` unions children's ``_no_split_modules``
        # (FSDP unit class names) and parallel plans onto the composite model.
        # ``super().__init__`` ran it before the children existed.
        #
        # It also runs ``init_weights()``, whose ``smart_apply`` dispatches each
        # child PreTrainedModel's *own* ``_init_weights`` — real random init for
        # any module not flagged ``_is_hf_initialized``. A child handed to us has
        # already been built and possibly loaded (VeOmni materializes a module
        # before composing, and its loader does not set that flag the way HF's
        # ``from_pretrained`` does), so let the children keep what they hold.
        for name in self._module_names:
            for module in getattr(self, name).modules():
                module._is_hf_initialized = True
        self.post_init()

        # Convert may split modules before any DAG/FSM exists. Build each graph
        # only when the config has one; :meth:`forward` / :meth:`generate` report
        # the absence at train / infer time. A present graph checks endpoints
        # against these modules during its own construction.
        self.training_graph = TrainingGraph(config.training_graph, modules=modules) if config.training_graph else None
        self.generation_graph = (
            GenerationGraph(config.generation_graph, modules=modules) if config.generation_graphs else None
        )

        self._last_printed_state: str | None = None
        self._generated: list[dict[str, Any]] = []
        self._losses: dict[str, Any] = {}

        self.reset()

    def _init_weights(self, module: nn.Module) -> None:
        """Sub-modules own weight init; the composed model has no standalone params."""
        return

    @classmethod
    def from_config(cls, config: OmniConfig | dict[str, Any], **kwargs: Any) -> OmniModel:
        """Build an :class:`OmniModel` from config only (sub-modules without weights)."""
        if not isinstance(config, OmniConfig):
            config = OmniConfig.from_dict(config)
        checkpoint_root = kwargs.pop("checkpoint_root", None) or getattr(config, "_name_or_path", None)
        kwargs = config.apply_omni_kwargs(kwargs)
        modules = cls._load_modules(
            config,
            checkpoint_root=checkpoint_root,
            load_weights=False,
            **kwargs,
        )
        return cls(config, modules)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *model_args: Any,
        **kwargs: Any,
    ) -> OmniModel:
        """Load an :class:`OmniModel` and every declared sub-module from a split checkpoint.

        ``pretrained_model_name_or_path`` is the omni root (``config.json`` +
        graph YAML sidecars + one subfolder per module).

        Remaining kwargs are forwarded to **every** sub-module's
        ``from_pretrained`` as global load options (e.g. ``torch_dtype``,
        ``device_map``). Per-module ``model_config`` and ``attn_implementation``
        are already on the module config loaded by :class:`OmniConfig`.

        Pass ``config=`` to load the weights under an already-resolved
        :class:`OmniConfig`. That object is used as given.

        When no ``config`` is passed, caller overwrites go into
        :meth:`OmniConfig.from_pretrained` and are merged before each module
        config loads: root ``ops_implementation`` / ``xxx_implementation``, a
        module entry's ``model_path`` / ``ops_implementation`` /
        ``model_config`` / ``processor_config``, and the top-level fields
        ``infer_type``, ``generation_kwargs``, ``training_graphs``,
        ``generation_graphs``, ``_module_entries``.
        """
        config = kwargs.pop("config", None)
        config_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in _CONFIG_LOAD_KWARG_NAMES}
        config_overrides = {key: kwargs.pop(key) for key in list(kwargs) if key in _OMNI_CONFIG_OVERRIDE_KEYS}
        omni_kwargs = pop_omni_kwargs(kwargs)
        if config is None:
            config = OmniConfig.from_pretrained(
                pretrained_model_name_or_path,
                **config_kwargs,
                **config_overrides,
                **omni_kwargs,
            )
        else:
            overlay = sorted({**config_overrides, **omni_kwargs})
            if not isinstance(config, OmniConfig) or overlay:
                raise ValueError(
                    "Pass config as an OmniConfig with its fields already resolved, "
                    f"got {type(config).__name__} and overlay kwargs {overlay}."
                )

        # Same root :meth:`OmniConfig.from_pretrained` used to load each module
        # config: the directory the config remembers, else the path passed here.
        checkpoint_root = config.checkpoint_root or (
            str(pretrained_model_name_or_path) if pretrained_model_name_or_path else None
        )

        modules = cls._load_modules(
            config,
            checkpoint_root=checkpoint_root,
            load_weights=True,
            **kwargs,
        )
        return cls(config, modules)

    @staticmethod
    def _init_only_load_kwargs(load_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Narrow module load options to what ``_from_config`` consumes.

        ``from_pretrained`` turns an unrecognized kwarg into a config override,
        but ``_from_config`` hands everything it does not pop straight to the
        model's ``__init__``. Forwarding a global weight-loading option such as
        ``device_map`` — which :meth:`build_foundation_model` documents and
        ``tasks/omni`` passes — would raise a ``TypeError`` there, and placement
        means nothing on a path that loads no weights.
        """
        return {key: value for key, value in load_kwargs.items() if key in _FROM_CONFIG_KWARG_NAMES}

    @classmethod
    def _load_modules(
        cls,
        config: OmniConfig,
        *,
        checkpoint_root: str | os.PathLike | None,
        load_weights: bool,
        **kwargs: Any,
    ) -> dict[str, PretrainedOmniModule]:
        """Load each declared module.

        ``load_weights=True`` calls ``from_pretrained`` (checkpoint weights).
        ``load_weights=False`` calls ``_from_config`` (architecture only).

        ``kwargs`` are HF load options here (``torch_dtype``, ``device_map``,
        …). A caller's kernel selections were absorbed into ``config`` by
        :meth:`OmniConfig.apply_omni_kwargs` before this runs, so kernels are
        read from the config like every other layer.
        """

        from ...arguments import OpsImplementationConfig
        from ...ops import apply_ops_config
        from ...ops.config.singleton import get_ops_config
        from ..auto import bind_ops_to_modeling

        base_ops = get_ops_config()
        modules: dict[str, PretrainedOmniModule] = {}
        for name in config.module_names:
            entry = config._module_entries[name]
            module_path = OmniModuleConfig.resolve_path(checkpoint_root, name, entry.get("model_path"))
            module_config = config._module_configs[name]
            mod_cls = OMNI_MODEL_REGISTRY[module_config.model_type]()
            module_ops = OpsImplementationConfig(**dict(module_config.ops_implementation or {}))
            # Copy per module: HF pops keys such as ``torch_dtype`` out of the dict it is given.
            # TODO: make it per module kwargs
            module_kwargs = dict(kwargs)

            # Install this module's kernels for the duration of its own load, so
            # a module's kernels do not depend on its position in
            # ``config.module_names``.
            apply_ops_config(module_ops)

            # After the ops config above was installed.
            bind_ops_to_modeling(mod_cls)
            if load_weights:
                modules[name] = mod_cls.from_pretrained(module_path, config=module_config, **module_kwargs)
            else:
                modules[name] = mod_cls._from_config(module_config, **cls._init_only_load_kwargs(module_kwargs))

        # Don't leave the caller's config holding the last module's override.
        if base_ops is not None:
            apply_ops_config(base_ops)
        return modules

    def _save_module_subdirectory(
        self,
        name: str,
        module: PretrainedOmniModule,
        save_directory: str,
        *,
        save_module_weights: bool,
        **kwargs: Any,
    ) -> None:
        module_dir = os.path.join(save_directory, name)
        module.save_pretrained(module_dir, save_module_weights=save_module_weights, **kwargs)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        is_main_process: bool | None = None,
        state_dict: dict[str, Any] | None = None,
        push_to_hub: bool = False,
        max_shard_size: int | str = "5GB",
        variant: str | None = None,
        token: str | bool | None = None,
        save_peft_format: bool = True,
        save_original_format: bool = True,
        distributed_checkpoint: bool = False,
        *,
        save_module_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        """Write an HF-style omni checkpoint (root config/graphs + module subfolders).

        The positional parameters mirror
        :meth:`transformers.PreTrainedModel.save_pretrained` so an ``OmniModel``
        stays substitutable for a plain HF model. A composite holds no weights
        of its own, so each per-model option is handed to every sub-module's
        own ``save_pretrained`` rather than acted on here; the three that cannot
        be expressed against a directory of sub-checkpoints are rejected.

        Parameters
        ----------
        save_directory:
            Omni checkpoint root. Each module is written under ``<root>/<name>/``.
        is_main_process:
            ``None`` (the default) resolves to "this is rank 0".
        state_dict / push_to_hub / distributed_checkpoint:
            Unsupported — there is no single composite state dict, no hub
            export, and sharded export belongs to the checkpoint manager.
        max_shard_size / variant / token / save_peft_format / save_original_format:
            Forwarded to each sub-module's ``save_pretrained`` when
            ``save_module_weights=True``.
        save_module_weights:
            When ``False``, only each module's ``config.json`` and attached
            assets (processor / tokenizer) are written — used for the initial
            ``model_assets`` export at train begin.

        Rank-0 writes and returns; the other ranks return immediately and this
        does **not** barrier. The caller owns the barrier, because it is the one
        that knows what the other ranks go on to do —
        :meth:`OmniTrainer.save_model_assets` barriers right after. Without one,
        a rank can read a half-written directory.

        For the same reason ``save_module_weights=True`` is for an unsharded
        model only: a sub-module's ``state_dict`` over DTensor parameters is
        collective, so calling it on rank 0 alone would hang. Assets
        (``save_module_weights=False``) are plain host-side objects and are safe
        on a sharded model. Weight export from a sharded model goes through the
        checkpoint manager instead.
        """
        unsupported = [
            name
            for name, given in (
                ("state_dict", state_dict is not None),
                ("push_to_hub", push_to_hub),
                ("distributed_checkpoint", distributed_checkpoint),
            )
            if given
        ]
        if unsupported:
            raise NotImplementedError(
                f"OmniModel.save_pretrained does not support {unsupported}: an omni checkpoint is a "
                "directory of per-module checkpoints, not a single model."
            )

        if is_main_process is None:
            is_main_process = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        if not is_main_process:
            return

        save_directory = str(save_directory)
        os.makedirs(save_directory, exist_ok=True)

        module_save_kwargs = {
            **kwargs,
            "max_shard_size": max_shard_size,
            "variant": variant,
            "token": token,
            "save_peft_format": save_peft_format,
            "save_original_format": save_original_format,
        }
        for name, module in self.named_omni_modules():
            self._save_module_subdirectory(
                name,
                module,
                save_directory,
                save_module_weights=save_module_weights,
                **module_save_kwargs,
            )

        self.config.save_pretrained(save_directory)

    @property
    def modules_dict(self) -> dict[str, PretrainedOmniModule]:
        """Back-compat dict view of the sub-modules."""
        return {name: getattr(self, name) for name in self._module_names}

    def _run_train_node(
        self,
        module: PretrainedOmniModule,
        node: NodeDef,
        batch: dict[str, Any],
    ) -> None:
        """Run one training node — optional ``pre_forward`` → endpoint → optional ``post_forward``."""
        method = node.method
        fn = getattr(module, method, None)
        if fn is None:
            raise AttributeError(f"Node method {type(module).__name__}.{method}() is not implemented.")
        inputs: dict[str, Any] = batch
        pre_forward = getattr(module, "pre_forward", None)
        if pre_forward is not None:
            inputs = pre_forward(method=method, **inputs)
        outputs = fn(**inputs)
        post_forward = getattr(module, "post_forward", None)
        if post_forward is not None:
            outputs = post_forward(method=method, **outputs)
        batch.update(outputs)

    def forward(
        self,
        batch: dict[str, Any],
        *args: Any,
        node_runner: Callable[[PretrainedOmniModule, NodeDef, dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the training DAG; this is the FSDP2 root ``forward``.

        Each node is optional ``pre_forward`` → endpoint → optional ``post_forward``.
        Mixins are not required on a native ``PretrainedOmniModule``. Nested wrap
        units (decoder layers, ``Embedding``, …) unshard on their own
        ``__call__``; leftover params on this module unshard because training
        enters here.

        ``node_runner`` replaces how one node runs; the graph walk, loss
        collection and return value stay here. A runtime that wraps the modules
        passes one to unwrap them and scope each node to its module's mesh;
        without one each node runs as :meth:`_run_train_node`.
        """
        del args, kwargs
        run_node = node_runner if node_runner is not None else self._run_train_node
        if self.training_graph is None:
            raise ValueError(
                "OmniModel.forward: this model has no training graph. Pass `training_graphs` "
                f"(or place `{DEFAULT_TRAINING_GRAPH_FILE}` next to the checkpoint) to train it."
            )

        self.training_graph.reset()
        self._losses.clear()

        for node in self.training_graph.iter_nodes():
            run_node(self.get_module(node.module), node, batch)
            loss = batch.pop(LOSS_KEY, None)
            if loss is not None:
                self._losses[node.name] = loss
        return {"loss": _sum_losses(self._losses), "losses": dict(self._losses)}

    def reset(self) -> None:
        """Clear per-request inference runtime state."""
        if self.generation_graph is not None:
            self.generation_graph.reset()
        self._generated.clear()
        for _, module in self.named_omni_modules():
            module.reset_global_inference_state()

    @staticmethod
    def _normalize_generated(item: Any) -> dict[str, Any] | None:
        if item is None:
            return None
        if isinstance(item, dict) and "type" in item and "value" in item:
            normalized: dict[str, Any] = {"type": item["type"], "value": item["value"]}
            if item.get("meta") is not None:
                normalized["meta"] = item["meta"]
            return normalized
        return None

    def _collect_generated(self, payload: dict[str, Any]) -> None:
        """Drain ``payload["generated"]`` into :attr:`_generated` (one-shot).

        ``payload`` is the FSM ``ctx`` after a node ran, or whatever a module's
        ``finalize`` returned — both carry an artefact under the same key.
        """
        normalized = self._normalize_generated(payload.pop("generated", None))
        if normalized is not None:
            self._generated.append(normalized)

    def _run_generation_node(
        self,
        module: PretrainedOmniModule,
        node: NodeDef,
        ctx: dict[str, Any],
        generation_kwargs: dict[str, Any] | None,
    ) -> None:
        """Run one generation node — optional ``pre_generate`` → endpoint → optional ``post_generate``."""
        method = node.method
        fn = getattr(module, method, None)
        if fn is None:
            raise AttributeError(f"Node method {type(module).__name__}.{method}() is not implemented.")
        inputs = ctx
        pre_generate = getattr(module, "pre_generate", None)
        if pre_generate is not None:
            inputs = pre_generate(method=method, **inputs)
        out = fn(**inputs, generation_kwargs=generation_kwargs)
        if not isinstance(out, dict):
            raise TypeError(f"FSM node '{node.name}'.{method} must return a dict; got {type(out).__name__}.")
        post_generate = getattr(module, "post_generate", None)
        if post_generate is not None:
            out = post_generate(method=method, **out)
        ctx.update(out)

    def _run_module_finalize(self, module: PretrainedOmniModule, ctx: dict[str, Any]) -> None:
        """Run one module's abort-only flush — ``finalize`` → merge into ``ctx``."""
        out = module.finalize(ctx=ctx)
        if not isinstance(out, dict):
            raise TypeError(f"{type(module).__name__}.finalize must return a dict, got {type(out).__name__}.")
        ctx.update(out)

    def _emit_progress(self, total_steps: int) -> None:
        graph = self.generation_graph
        if graph is None:
            return
        if total_steps == 0:
            self._last_printed_state = None
        current = graph.current_state_name
        if current != self._last_printed_state:
            logger.info_rank0(f"[FSM] step {total_steps:>4}: {current}")
            self._last_printed_state = current

    def generate(
        self,
        request: dict[str, Any],
        generation_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run inference using the FSM (eager path).

        Parameters
        ----------
        request:
            Generation request dict — used directly as the initial ``ctx`` and
            mutated in place as the FSM runs. Call :meth:`reset` before each
            request to clear graph / module / artefact state.
        generation_kwargs:
            Per-request knobs merged on top of :attr:`~OmniConfig.generation_kwargs`
            defaults (request keys win). Forwarded to every module's FSM step.
            The framework only reads ``max_new_tokens`` (default 2048).

        Returns
        -------
        list[dict]
            Generated artefacts — ``{"type": ..., "value": ..., "meta": ...}``
            entries collected from the FSM (text replies, images, …).
        """
        if self.generation_graph is None:
            raise ValueError(
                "OmniModel.generate: this model has no generation graph. Pass `generation_graphs` "
                f"(or place `{DEFAULT_GENERATION_GRAPH_FILE}` next to the checkpoint) to run inference."
            )
        ctx: dict[str, Any] = request
        generation_kwargs = self.resolve_generation_kwargs(generation_kwargs)

        max_new_tokens = generation_kwargs.get("max_new_tokens", 2048)
        total_steps = 0
        while not self.generation_graph.is_done() and total_steps < max_new_tokens:
            self._emit_progress(total_steps)
            for node in self.generation_graph.iter_nodes(ctx):
                module = getattr(self, node.module)
                self._run_generation_node(module, node, ctx, generation_kwargs)
                # Per node, not per body pass: ``ctx`` is one shared dict, so a
                # second node emitting ``generated`` overwrites the first's
                # artefact before a pass-level drain could see it.
                self._collect_generated(ctx)
            total_steps += 1
            self.generation_graph.maybe_transition(ctx)

        self._emit_progress(total_steps)

        # ``done``: the last state already finished and its artefacts are in;
        # do not finalize again. Still running means the safety cap aborted a
        # span mid-flight (often mid-image tokens, which cannot be finalized),
        # so this is a best-effort salvage, typically text-only.
        if not self.generation_graph.is_done():
            for _, module in self.named_omni_modules():
                self._run_module_finalize(module, ctx)
                # Drained per module for the same reason as the node loop above.
                self._collect_generated(ctx)

        return list(self._generated)

    def named_omni_modules(self) -> Iterator[tuple[str, PretrainedOmniModule]]:
        """Yield ``(name, module)`` for every graph participant."""
        for name in self._module_names:
            yield name, getattr(self, name)

    def get_module(self, name: str) -> PretrainedOmniModule:
        if name not in self._module_names:
            raise KeyError(f"Module '{name}' not found in OmniModel")
        return getattr(self, name)

    def resolve_generation_kwargs(
        self,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge :attr:`~OmniConfig.generation_kwargs` defaults with per-request overrides."""
        resolved = dict(self.config.generation_kwargs or {})
        resolved.update(generation_kwargs or {})
        return resolved

    def collect_assets(self) -> list[Any]:
        """Collect per-module assets (vision/audio processors, codebooks)."""
        assets: list[Any] = []
        for _, module in self.named_omni_modules():
            assets.extend(module.get_assets())
        return assets


def _sum_losses(losses: dict[str, Any]) -> Any | None:
    if not losses:
        return None
    it = iter(losses.values())
    total = next(it)
    for v in it:
        total = total + v
    return total


__all__ = ["LOSS_KEY", "OmniModel"]
