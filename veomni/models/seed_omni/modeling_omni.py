"""
OmniModel V2 — composable multi-modal model driven by config-specified graphs.

This file holds the **clean modeling definition** — FSM inference via
:meth:`OmniModel.generate` and checkpoint compose/load/save.  It is profiler-free
and parallel-infra-free so it can be loaded with HF ``from_pretrained`` /
``from_config`` and run eager single-process inference without VeOmni.

**Training** requires :class:`~veomni.models.seed_omni.accelerator.omni_model_runtime.OmniModelRuntime`,
which owns the training DAG loop, parallel-state scoping, graph profiling, and
metric metering.

Architecture
------------
``OmniModel`` carries:

* sub-modules           — each graph participant is attached as a **direct
                          attribute** of ``OmniModel``.  In the eager path these
                          are bare hook-bearing modules (typically a concrete
                          ``*ModuleMixin + PreTrainedModel`` class from the
                          registry).  Under VeOmni training they may be
                          FSDP/DDP-wrapped; the graph resolves graph hooks
                          (``pre_forward`` / ``post_forward``) through
                          :func:`~veomni.models.seed_omni.graphs.dispatch.unwrap_graph_module`.
                          Parallel/acceleration build hooks live on
                          :class:`~veomni.models.seed_omni.accelerator.module_runtime.ModuleRuntime`
                          (or a customized runtime subclass), not on
                          :class:`~veomni.models.seed_omni.mixins.modulemixin.ModuleMixin`.
* ``generation_graph``  — :class:`GenerationGraph` (FSM).

Training uses :class:`~veomni.models.seed_omni.accelerator.omni_model_runtime.OmniModelRuntime`
to walk ``training_graph``; the bare :class:`OmniModel` does not implement it.

Inference
---------
``generate(request, generation_kwargs)`` loops the FSM.  Each node calls its
endpoint directly (no pre/post hooks).  Stop when ``is_done()`` or
``max_new_tokens`` is reached.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Mapping

import torch.distributed as dist
import torch.nn as nn
from transformers import PreTrainedModel

from ...utils import helper
from .configuration_omni import OmniConfig
from .graphs.dispatch import unwrap_graph_module, unwrap_module_chain
from .graphs.generation_graph import GenerationGraph
from .graphs.graph import NodeDef
from .graphs.training_graph import TrainingGraph
from .mixins.modulemixin import ModuleMixin
from .modules import OMNI_MODEL_REGISTRY, read_model_type


logger = helper.create_logger(__name__)

# Must match the ``_loss`` key every OmniModule's ``post_forward`` emits.
_LOSS_KEY = "_loss"

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
        "infer_type",
        "generation_kwargs",
        "training_graph",
        "generation_graphs",
        "modules",
    }
)


class OmniModel(PreTrainedModel):
    """Pure SeedOmni modeling runtime over already-built sub-modules.

    Parameters
    ----------
    config:
        :class:`OmniConfig` with ``modules`` / ``training_graph`` /
        ``generation_graphs`` populated. The FSM bound here is the one
        ``config.infer_type`` selects, so switching scenario means rebuilding.
    modules:
        ``{module_name: nn.Module}`` — already constructed graph participants.
        VeOmni trainers/inferencers compose them into :class:`OmniModel` and run
        graph loops via :class:`~veomni.models.seed_omni.accelerator.omni_model_runtime.OmniModelRuntime`.
        Training may attach FSDP/DDP wrappers around each entry;
        graph hooks are resolved via :func:`~veomni.models.seed_omni.graphs.dispatch.unwrap_graph_module`.
    """

    config_class = OmniConfig
    base_model_prefix = "omni"
    main_input_name = "conversation_list"
    supports_gradient_checkpointing = False

    def __init__(self, config: OmniConfig, modules: Mapping[str, nn.Module]):
        super().__init__(config)

        self._module_names: list[str] = list(config.module_names)
        for name in self._module_names:
            self.add_module(name, modules[name])

        self.training_graph = TrainingGraph(config.training_graph)
        self.generation_graph = GenerationGraph(config.generation_graph)

        self._last_printed_state: str | None = None
        self._generated: list[dict[str, Any]] = []

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
        modules = cls._load_modules(
            config,
            checkpoint_root=checkpoint_root,
            pretrained=False,
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
        ``device_map``).  Per-module ``model_config`` and ``ops_implementation``
        persisted in the checkpoint are merged on top via
        :meth:`_build_module_load_kwargs` — module-level ``attn_implementation``
        wins over any global kwarg when both are set.

        Pass ``config=`` to load the weights under an already-resolved
        :class:`OmniConfig` instead of the root ``config.json`` — how
        :class:`~veomni.trainer.omni.omni_inferencer.OmniInferencer` keeps its
        launcher-YAML graph / ``model_config`` overrides on the eager path.

        Top-level :class:`OmniConfig` fields (``infer_type``, ``generation_kwargs``,
        …) may also be passed as kwargs and override the checkpoint defaults.
        """
        config = kwargs.pop("config", None)
        config_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in _CONFIG_LOAD_KWARG_NAMES}
        config_overrides = {key: kwargs.pop(key) for key in list(kwargs) if key in _OMNI_CONFIG_OVERRIDE_KEYS}
        if config is None:
            config = OmniConfig.from_pretrained(
                pretrained_model_name_or_path,
                **config_kwargs,
                **config_overrides,
            )
        elif not isinstance(config, OmniConfig):
            config = OmniConfig.from_dict(config)
            for key, value in config_overrides.items():
                setattr(config, key, value)
        elif config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)

        checkpoint_root = getattr(config, "_name_or_path", None) or str(pretrained_model_name_or_path)
        modules = cls._load_modules(
            config,
            checkpoint_root=checkpoint_root,
            pretrained=True,
            **kwargs,
        )
        return cls(config, modules)

    @staticmethod
    def _build_module_load_kwargs(
        config: OmniConfig,
        name: str,
        base_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge checkpoint overrides and apply persisted ``ops_implementation``."""
        from ...arguments import OpsImplementationConfig
        from ...ops import apply_ops_config

        load_kwargs = {**base_kwargs, **config.module_model_config(name)}
        ops_dict = config.module_ops_implementation(name)
        if not ops_dict:
            return load_kwargs

        ops = OpsImplementationConfig(**ops_dict)
        apply_ops_config(ops)
        if ops.attn_implementation is not None:
            load_kwargs["attn_implementation"] = ops.attn_implementation
        return load_kwargs

    @classmethod
    def _load_modules(
        cls,
        config: OmniConfig,
        *,
        checkpoint_root: str | os.PathLike | None,
        pretrained: bool,
        **kwargs: Any,
    ) -> dict[str, nn.Module]:
        import sys

        from transformers import PretrainedConfig

        from ...ops.config.singleton import get_ops_config
        from ..auto import _bind_veomni_ops

        modules: dict[str, nn.Module] = {}
        for name in config.module_names:
            module_path = config.resolve_module_path(checkpoint_root, name)
            entry = config.modules.get(name)
            load_kwargs = cls._build_module_load_kwargs(config, name, kwargs)
            if isinstance(entry, PretrainedConfig):
                model_type = entry.model_type
                mod_cls = OMNI_MODEL_REGISTRY[model_type]()
                modeling_module = sys.modules.get(mod_cls.__module__)
                if modeling_module is not None and config.module_ops_implementation(name):
                    _bind_veomni_ops(modeling_module, get_ops_config())
                if pretrained:
                    modules[name] = mod_cls.from_pretrained(module_path, config=entry, **load_kwargs)
                else:
                    modules[name] = mod_cls.from_config(entry, **load_kwargs)
                continue
            model_type = read_model_type(module_path)
            mod_cls = OMNI_MODEL_REGISTRY[model_type]()
            modeling_module = sys.modules.get(mod_cls.__module__)
            if modeling_module is not None and config.module_ops_implementation(name):
                _bind_veomni_ops(modeling_module, get_ops_config())
            if pretrained:
                modules[name] = mod_cls.from_pretrained(module_path, **load_kwargs)
            else:
                cfg_cls = mod_cls.config_class
                sub_config = cfg_cls.from_pretrained(module_path)
                modules[name] = mod_cls.from_config(sub_config, **config.module_model_config(name))
        return modules

    @staticmethod
    def _save_module_assets(module: nn.Module, module_dir: str) -> None:
        """Save a module's config plus its processor / tokenizer sidecars (no weights).

        ``module`` may still be DDP-wrapped under VeOmni training (FSDP2 composes
        in place, but ``DistributedDataParallel`` does not forward unknown
        attribute lookups to ``.module``) — unwrap first so ``config`` / processor
        / tokenizer resolve regardless of ``dp_mode``.
        """
        module = unwrap_module_chain(module)
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "save_pretrained"):
            cfg.save_pretrained(module_dir)
        for attr in ("_processor", "_image_processor", "_video_processor", "_tokenizer"):
            asset = getattr(module, attr, None)
            if asset is not None and hasattr(asset, "save_pretrained"):
                asset.save_pretrained(module_dir)

    def _save_module_subdirectory(
        self,
        name: str,
        module: nn.Module,
        save_directory: str,
        *,
        save_module_weights: bool,
        **kwargs: Any,
    ) -> None:
        subfolder = self.config.module_checkpoint_subfolder(name)
        module_dir = os.path.join(save_directory, subfolder)
        os.makedirs(module_dir, exist_ok=True)
        # Processors / tokenizers are not part of a module's ``save_pretrained``, so
        # they are written on both paths: a weights export without them reloads as a
        # model that cannot preprocess its own inputs. Weights go last so the module
        # has the final say over ``config.json``.
        self._save_module_assets(module, module_dir)
        if save_module_weights:
            if not hasattr(module, "save_pretrained"):
                raise TypeError(
                    f"OmniModel.save_pretrained: sub-module '{name}' ({type(module).__name__}) "
                    "has no save_pretrained()."
                )
            module.save_pretrained(module_dir, **kwargs)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        *,
        save_module_weights: bool = True,
        safe_serialization: bool = True,
        max_shard_size: int | str = "5GB",
        is_main_process: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Write an HF-style omni checkpoint (root config/graphs + module subfolders).

        Parameters
        ----------
        save_directory:
            Omni checkpoint root. Each module is written under
            ``<root>/<subfolder>/`` where ``subfolder`` comes from
            :meth:`OmniConfig.module_checkpoint_subfolder`.
        save_module_weights:
            When ``False``, only each module's ``config.json`` and attached
            assets (processor / tokenizer) are written — used for the initial
            ``model_assets`` export at train begin.
        safe_serialization / max_shard_size:
            Forwarded to each sub-module's ``save_pretrained`` when
            ``save_module_weights=True``.
        """
        if is_main_process is None:
            is_main_process = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        if not is_main_process:
            return

        save_directory = str(save_directory)
        os.makedirs(save_directory, exist_ok=True)

        module_save_kwargs = {
            **kwargs,
            "safe_serialization": safe_serialization,
            "max_shard_size": max_shard_size,
        }
        for name in self._module_names:
            module = getattr(self, name)
            self._save_module_subdirectory(
                name,
                module,
                save_directory,
                save_module_weights=save_module_weights,
                **module_save_kwargs,
            )

        self.config.save_pretrained(save_directory)

    @property
    def modules_dict(self) -> dict[str, nn.Module]:
        """Back-compat dict view of the sub-modules."""
        return {name: getattr(self, name) for name in self._module_names}

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Training is not supported on the bare :class:`OmniModel`.

        Use :class:`~veomni.models.seed_omni.accelerator.omni_model_runtime.OmniModelRuntime`
        (via :class:`~veomni.trainer.omni.omni_trainer.OmniTrainer`) for the training DAG.
        """
        raise NotImplementedError(
            "OmniModel.forward() is not available on the native eager model. "
            "Training requires OmniModelRuntime (see OmniTrainer)."
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear per-conversation inference runtime state."""
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

    def _append_generated(self, item: Any) -> None:
        normalized = self._normalize_generated(item)
        if normalized is not None:
            self._generated.append(normalized)

    def _collect_generated(self, ctx: dict[str, Any]) -> None:
        """Drain ``ctx["generated"]`` into :attr:`_generated` (one-shot)."""
        self._append_generated(ctx.pop("generated", None))

    def _run_generation_node(
        self,
        module: nn.Module,
        node: NodeDef,
        ctx: dict[str, Any],
        generation_kwargs: dict[str, Any] | None,
    ) -> None:
        """Run one generation node — eager endpoint call only."""
        method = node.method
        fn = getattr(module, method, None)
        if fn is None:
            raise AttributeError(f"Node method {type(module).__name__}.{method}() is not implemented.")
        out = fn(**ctx, generation_kwargs=generation_kwargs)
        if not isinstance(out, dict):
            raise TypeError(f"FSM node '{node.name}'.{method} must return a dict; got {type(out).__name__}.")
        ctx.update(out)

    def _invoke_module_finalize(self, ctx: dict[str, Any]) -> None:
        """Call ``finalize`` on every graph participant when the safety cap trips."""
        for _, raw in self.named_omni_modules():
            out = raw.finalize(ctx=ctx)
            if not isinstance(out, dict):
                raise TypeError(f"{type(raw).__name__}.finalize must return a dict, got {type(out).__name__}.")
            self._append_generated(out.pop("generated", None))

    def _emit_progress(self, total_steps: int) -> None:
        if total_steps == 0:
            self._last_printed_state = None
        current = self.generation_graph.current_state_name
        if current != self._last_printed_state:
            logger.info_rank0(f"[FSM] step {total_steps:>4}: {current}")
            self._last_printed_state = current

    def generate(
        self,
        request: dict[str, Any],
        generation_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run inference using the FSM (profiler-free eager path).

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
        ctx: dict[str, Any] = request
        generation_kwargs = self.resolve_generation_kwargs(generation_kwargs)

        max_new_tokens = generation_kwargs.get("max_new_tokens", 2048)
        total_steps = 0

        while not self.generation_graph.is_done() and total_steps < max_new_tokens:
            self._emit_progress(total_steps)
            for node in self.generation_graph.iter_nodes(ctx):
                module = getattr(self, node.module)
                self._run_generation_node(module, node, ctx, generation_kwargs)
            total_steps += 1
            self._collect_generated(ctx)
            self.generation_graph.maybe_transition(ctx)

        self._emit_progress(total_steps)

        if not self.generation_graph.is_done():
            self._invoke_module_finalize(ctx)

        return list(self._generated)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def named_omni_modules(self) -> Iterator[tuple[str, nn.Module]]:
        """Yield ``(name, raw)`` for every graph participant.

        ``raw`` is the :class:`ModuleMixin` resolved through
        :func:`~veomni.models.seed_omni.graphs.dispatch.unwrap_graph_module`
        (bare in the eager path; unwrapped from FSDP/DDP/LoRA wrappers under
        VeOmni).  Non-:class:`ModuleMixin` entries are skipped.
        """
        for name in self._module_names:
            module = getattr(self, name)
            if not _is_omni_module(module):
                continue
            yield name, unwrap_graph_module(module, module_name=name)

    def get_module(self, name: str) -> nn.Module:
        if name not in self._module_names:
            raise KeyError(f"Module '{name}' not found in OmniModel")
        return getattr(self, name)

    def resolve_generation_kwargs(
        self,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge :attr:`~OmniConfig.generation_kwargs` defaults with per-request overrides."""
        return merge_generation_kwargs(self.config.generation_kwargs, generation_kwargs)

    def collect_assets(self) -> list[Any]:
        """Collect per-module assets (vision/audio processors, codebooks)."""
        assets: list[Any] = []
        for _, module in self.named_omni_modules():
            assets.extend(module.get_assets())
        return assets


# ── helpers ───────────────────────────────────────────────────────────────────


def merge_generation_kwargs(
    defaults: Mapping[str, Any] | None,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return ``defaults`` updated with ``overrides`` (override keys win)."""
    merged = dict(defaults or {})
    merged.update(overrides or {})
    return merged


def _is_omni_module(mod: nn.Module) -> bool:
    """True when ``mod`` (possibly wrapped) resolves to a :class:`ModuleMixin`."""
    return isinstance(unwrap_module_chain(mod), ModuleMixin)


def _sum_losses(losses: dict[str, Any]) -> Any | None:
    if not losses:
        return None
    it = iter(losses.values())
    total = next(it)
    for v in it:
        total = total + v
    return total


__all__ = ["OmniModel", "_LOSS_KEY", "merge_generation_kwargs"]
