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

"""The model-bound half of a training job.

:class:`VeOmniModelRuntime` owns everything that belongs to *one* model —
build, freeze/LoRA, parallelize + weight load, optimizer, lr-scheduler,
gradient clipping and the device mesh they all read. It deliberately owns
nothing job-bound: no process-group init, no dataloader, no train loop, no
callbacks. A job has exactly one of those; it may have many models.

This split is what lets a single-model trainer and a multi-module omni model
share one build sequence instead of maintaining divergent copies of it.
"""

from contextlib import contextmanager
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from transformers import PretrainedConfig

from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.parallel_state import (
    get_parallel_state_by_name,
    init_parallel_state,
    use_parallel_state,
)
from ..distributed.torch_compile import CompileConfig
from ..distributed.torch_parallelize import build_parallelize_model
from ..optim import build_lr_scheduler, build_optimizer
from ..utils import helper, logging
from ..utils.checkpoint_utils import should_skip_hf_weight_load
from ..utils.model_utils import pretty_print_trainable_parameters
from .auto import build_foundation_model


if TYPE_CHECKING:
    from ..arguments import ModelRuntimeArguments


logger = logging.get_logger(__name__)


def _has_trainable_lora_parameters(module: Optional[torch.nn.Module]) -> bool:
    if module is None:
        return False
    return any(
        param.requires_grad and ({"lora_A", "lora_B"} & set(name.split(".")))
        for name, param in module.named_parameters()
    )


def _resolve_muon_lr(optimizer_cfg) -> float:
    """Resolve Muon LR, inheriting AdamW lr under match_rms_adamw when unset."""
    if optimizer_cfg.muon_lr is not None:
        return float(optimizer_cfg.muon_lr)
    adamw_lr = float(optimizer_cfg.lr)
    if optimizer_cfg.muon_adjust_lr_fn == "match_rms_adamw":
        return adamw_lr
    # original: Moonlight-style ~25x AdamW lr starting point
    return 25.0 * adamw_lr


def _collect_muon_kwargs(optimizer_cfg) -> Dict[str, Any]:
    """Pull Muon-specific hyperparameters out of ``OptimizerConfig``."""
    return {
        "lr": _resolve_muon_lr(optimizer_cfg),
        "momentum": optimizer_cfg.muon_momentum,
        "nesterov": optimizer_cfg.muon_nesterov,
        "weight_decay": optimizer_cfg.muon_weight_decay,
        "ns_steps": optimizer_cfg.muon_ns_steps,
        "ns_coefficients": tuple(optimizer_cfg.muon_ns_coefficients),
        "eps": optimizer_cfg.muon_eps,
        "adjust_lr_fn": optimizer_cfg.muon_adjust_lr_fn,
        "ns_implementation": optimizer_cfg.muon_ns_implementation,
        "gram_ns_reset_iterations": tuple(optimizer_cfg.muon_gram_ns_reset_iterations),
        # Resolved against the model in _build_muon_with_adamw, not ctor kwargs.
        "head_group_size": int(optimizer_cfg.muon_head_group_size),
        "head_split_modules": tuple(optimizer_cfg.muon_head_split_modules),
        # Surface for startup summary only; not a DistributedMuon ctor kwarg.
        "expert_zero_comm": bool(optimizer_cfg.muon_expert_zero_comm),
        "adamw_lr": float(optimizer_cfg.lr),
        "muon_lr_explicit": optimizer_cfg.muon_lr is not None,
    }


class VeOmniModelRuntime:
    """One model's training unit: build → freeze/LoRA → parallelize → optimizer → clip.

    Subclassed rather than composed. Every consumer today reaches the model
    through plain attributes (``runtime.model``, ``runtime.optimizer``) and
    several construct their trainer via ``__new__``, bypassing ``__init__``
    entirely; delegating properties would break both. Inheritance keeps
    attribute access identical to hand-written code.

    Subclasses supply three seams describing where their configuration lives:

    * :attr:`model_args` — the :class:`ModelRuntimeArguments` for *this* model.
      A single-model trainer nests it under ``args.model``; an omni module
      *is* one.
    * :attr:`runtime_name` — the key this model's :class:`ParallelState` is
      registered under, so sibling models can hold different meshes.
    * :attr:`checkpoint_load_path` — job-level resume path, the one piece of
      non-model state the build needs (see :attr:`skip_hf_weight_load`).

    Standalone use needs none of that::

        runtime = VeOmniModelRuntime(args.model)
        runtime.register_parallel_state()
        runtime.build_model()
        runtime.build_parallelized_model()
        runtime.build_optimizer()
    """

    model: Optional[torch.nn.Module] = None
    model_config: PretrainedConfig = PretrainedConfig()
    optimizer: Optional[Optimizer] = None
    lr_scheduler: Optional[LRScheduler] = None

    def __init__(
        self,
        args: "ModelRuntimeArguments",
        *,
        name: str = "base",
        checkpoint_load_path: Optional[str] = None,
    ):
        self._model_args = args
        self._runtime_name = name
        self._resume_load_path = checkpoint_load_path

    @property
    def model_args(self) -> "ModelRuntimeArguments":
        """The model fields, accelerator and optimizer config driving this model."""
        return self._model_args

    @property
    def runtime_name(self) -> str:
        """Registry key of this model's :class:`ParallelState`."""
        return self._runtime_name

    @property
    def checkpoint_load_path(self) -> Optional[str]:
        """Job-level DCP resume path, or ``None`` when this is a fresh run."""
        return self._resume_load_path

    def register_parallel_state(self, name: Optional[str] = None) -> None:
        """Build this model's device mesh and register it under :attr:`runtime_name`.

        The process group itself is job-bound and must already be initialised;
        this only derives the model's own mesh from its accelerator config.
        """
        accelerator = self.model_args.accelerator
        init_parallel_state(
            dp_size=accelerator.dp_size,
            dp_replicate_size=accelerator.dp_replicate_size,
            dp_shard_size=accelerator.dp_shard_size,
            tp_size=accelerator.tp_size,
            pp_size=accelerator.pp_size,
            cp_size=accelerator.cp_size,
            ulysses_size=accelerator.ulysses_size,
            extra_parallel_sizes=accelerator.extra_parallel_sizes,
            extra_parallel_placement_innermost=accelerator.extra_parallel_placement_innermost,
            extra_parallel_names=accelerator.extra_parallel_names,
            dp_mode=accelerator.fsdp_config.fsdp_mode,
            async_enabled=accelerator.enable_async,
            name=name if name is not None else self.runtime_name,
        )

    @property
    def parallel_state(self):
        """This model's :class:`ParallelState`, looked up by name on every access.

        A read-only lookup rather than a stored handle: the registry stays the
        single source of truth, so a state re-registered mid-job is picked up
        instead of silently serving a stale mesh.
        """
        return get_parallel_state_by_name(self.runtime_name)

    @contextmanager
    def scoped(self):
        """Make this model's :class:`ParallelState` current for the duration.

        Every build step below reads the ambient state via
        ``get_parallel_state()``, so each enters this itself. Callers never
        have to know the model's mesh, and nesting the same name is a no-op.
        """
        with use_parallel_state(self.runtime_name):
            yield

    def build_model(self) -> None:
        """Meta-init the model from its config via the registry-aware loader."""
        args = self.model_args
        logger.info_rank0("Build model")
        self.model = build_foundation_model(
            config_path=args.foundation_config_path,
            weights_path=args.model_path,
            torch_dtype="float32" if args.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
            init_device=args.accelerator.init_device,
            ops_implementation=args.ops_implementation,
            config_kwargs=args.model_config,
        )
        self.model_config = self.model.config

    def customized_setup_lora(self, lora_config: Dict) -> Optional[torch.nn.Module]:
        """Optional hook for a model that wraps its own LoRA adapters.

        Returning a module uses it verbatim; ``None`` (the default) runs the
        generic :meth:`setup_lora` path.
        """
        customized = getattr(self.model, "customized_setup_lora", None)
        if callable(customized):
            return customized(lora_config)
        return None

    def setup_lora(self) -> None:
        """Wrap :attr:`model` with the PEFT-free :class:`veomni.lora.VeOmniLoraModel`.

        A single native path handles both dense ``nn.Linear`` LoRA
        (``lora_modules`` / ``target_modules``) and MoE expert LoRA
        (``target_parameters``, wrapper flavour selected by
        ``share_expert_lora``). On resume (``lora_config['lora_adapter']`` set)
        the wrappers are rebuilt from the on-disk ``adapter_config.json`` (MoE
        mode lives in its ``veomni_lora`` block); otherwise a fresh adapter is
        initialised from the yaml config. Either way the actual adapter
        *weights* are streamed in later during parallelization
        (``build_parallelize_model`` with ``adapter_path``).

        Recognised ``lora_config`` keys (in addition to ``rank`` / ``alpha`` /
        ``lora_adapter`` / ``is_trainable``): ``lora_modules`` (aka
        ``target_modules``), ``target_parameters``, ``share_expert_lora``,
        ``use_rslora``, ``lora_dropout``, ``bias``, ``exclude_modules``,
        ``rank_pattern``, ``alpha_pattern``, ``modules_to_save`` — see
        :class:`veomni.lora.VeOmniLoraConfig`.

        Fused-MoE models (Qwen3-MoE family) may list the semantic expert module
        names ``gate_proj`` / ``up_proj`` / ``down_proj`` in ``lora_modules``;
        these are auto-mapped to the model's fused expert ``target_parameters``
        (see :func:`veomni.lora.resolve_fused_moe_lora_targets`). Dense models
        keep those names as ordinary ``nn.Linear`` LoRA targets.
        """
        lora_config = self.model_args.lora_config
        if not bool(lora_config):
            return

        customized_model = self.customized_setup_lora(lora_config)
        if customized_model is not None:
            # The trainable-adapter check below looks for ``lora_A``/``lora_B``
            # parameter names, which a bespoke wrapper has no reason to use.
            # An override owns its own validation.
            self.model = customized_model
            return

        from ..lora import VeOmniLoraConfig, VeOmniLoraModel, resolve_fused_moe_lora_targets

        lora_adapter_path = lora_config.get("lora_adapter", None)
        if lora_adapter_path is not None:
            logger.info_rank0(f"Wrapping model with VeOmniLoraModel from {lora_adapter_path}.")
            self.model = VeOmniLoraModel.from_pretrained(
                self.model,
                lora_adapter_path,
                is_trainable=lora_config.get("is_trainable", True),
            )
        else:
            # Rewrite semantic MoE module names onto fused expert parameters
            # before building the config (no-op for dense models / plain configs).
            resolved_config = resolve_fused_moe_lora_targets(self.model, lora_config)
            cfg = VeOmniLoraConfig.from_yaml(resolved_config)
            logger.info_rank0(f"Initialising VeOmni LoRA adapter from scratch: {cfg}.")
            self.model = VeOmniLoraModel(self.model, cfg)

        if not _has_trainable_lora_parameters(self.model):
            raise ValueError(
                "LoRA configuration produced no trainable adapters. Select at least one Linear or MoE target."
            )

    def freeze_model(self) -> None:
        """Apply LoRA and report what is left trainable."""
        self.setup_lora()
        pretty_print_trainable_parameters(self.model)
        helper.print_device_mem_info("VRAM usage after building model")

    @property
    def skip_hf_weight_load(self) -> bool:
        """Whether the initial HF weight materialization can be skipped.

        A full non-LoRA resume already carries model weights, so materializing
        the HF checkpoint only to overwrite it costs a second memory peak that
        can OOM large MoE jobs. LoRA resumes still need the HF base.
        """
        return should_skip_hf_weight_load(self.checkpoint_load_path, self.model_args.lora_config)

    def customized_build_parallelize_model(self, *, weights_path: Optional[str]) -> Optional[torch.nn.Module]:
        """Optional hook owning parallelize + weight load end to end.

        Returning a module uses it verbatim — the override then owns FSDP/DDP
        wrap, weight load, param offload, gradient checkpointing and mixed
        precision. ``None`` (the default) runs the generic path below. Exists
        for models the generic GPU-materializing loader has no hook for, such
        as a MoE backbone streaming EP-sharded experts to CPU.
        """
        del weights_path
        return None

    def build_parallelized_model(self) -> None:
        """FSDP2/DDP-wrap the model and load its weights.

        The wrap preserves ``requires_grad`` (the shard inherits it) and the
        loader writes weights in place, so a freeze applied in
        :meth:`freeze_model` survives and is not re-asserted here.
        """
        args = self.model_args

        customized_model = self.customized_build_parallelize_model(weights_path=args.model_path)
        if customized_model is not None:
            self.model = customized_model
            return

        kwargs: Dict[str, Any] = {}
        cpu_load_param_name = None
        if hasattr(self.model, "get_parallel_plan"):
            cpu_load_param_name = getattr(self.model.get_parallel_plan(), "cpu_load_param_name", None)
        kwargs["cpu_load_param_name"] = cpu_load_param_name
        if bool(args.lora_config):
            kwargs["adapter_path"] = args.lora_config.get("lora_adapter", None)
            kwargs["is_peft_model"] = True

        muon_expert_zero_comm = args.optimizer.type == "muon" and args.optimizer.muon_expert_zero_comm

        if args.fqn_to_index_mapping is not None:
            kwargs["fqn_to_index_mapping"] = args.fqn_to_index_mapping
        if args.accelerator.chunk_mbs_config.enable:
            kwargs["chunk_mbs_config"] = args.accelerator.chunk_mbs_config

        skip_hf_weight_load = self.skip_hf_weight_load
        if skip_hf_weight_load:
            logger.info_rank0(
                f"Checkpoint resume enabled (load_path={self.checkpoint_load_path}); "
                "skipping HF weight materialization before checkpoint restore."
            )

        self.model = build_parallelize_model(
            self.model,
            init_device=args.accelerator.init_device,
            weights_path=args.model_path,
            should_skip_hf_weight_load=skip_hf_weight_load,
            enable_reshard_after_forward=args.accelerator.fsdp_config.reshard_after_forward,
            mixed_precision=args.accelerator.fsdp_config.mixed_precision,
            enable_gradient_checkpointing=args.accelerator.gradient_checkpointing.enable,
            basic_modules=list(set(getattr(self.model, "_no_split_modules", None) or []) | set(args.basic_modules)),
            enable_reentrant=args.accelerator.gradient_checkpointing.enable_reentrant,
            early_stop=args.accelerator.gradient_checkpointing.early_stop,
            enable_forward_prefetch=args.accelerator.fsdp_config.forward_prefetch,
            enable_fsdp_offload=args.accelerator.fsdp_config.offload,
            fsdp_offload_pin_memory=args.accelerator.fsdp_config.offload_pin_memory,
            broadcast_model_weights_from_rank0=args.accelerator.broadcast_model_weights_from_rank0,
            ep_sharded_stream_load=args.accelerator.ep_sharded_stream_load,
            max_load_broadcast_size=args.accelerator.fsdp_config.max_load_broadcast_size,
            muon_expert_zero_comm=muon_expert_zero_comm,
            compile_config=CompileConfig(
                **{field.name: getattr(args.accelerator.torch_compile, field.name) for field in fields(CompileConfig)}
            ),
            **kwargs,
        )
        self.model.train()

    def build_optimizer(self) -> None:
        """Build the optimizer over this model's still-trainable params.

        A distributed optimizer (Muon) reads ``get_parallel_state()`` at build
        time, so it must resolve to *this* model's mesh.
        """
        opt = self.model_args.optimizer
        self.optimizer = build_optimizer(
            self.model,
            lr=opt.lr,
            weight_decay=opt.weight_decay,
            fused=True,
            optimizer_type=opt.type,
            no_decay_modules=opt.no_decay_modules,
            no_decay_params=opt.no_decay_params,
            muon_kwargs=_collect_muon_kwargs(opt),
        )

    def build_lr_scheduler(self, total_steps: int) -> None:
        """Build the lr-scheduler over ``total_steps``.

        Takes the step count rather than reading it off a training config: it
        is only known once the dataset has been sized, which is job-bound.
        """
        opt = self.model_args.optimizer
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            train_steps=total_steps,
            lr=opt.lr,
            lr_min=opt.lr_min,
            lr_decay_style=opt.lr_decay_style,
            lr_decay_ratio=opt.lr_decay_ratio,
            lr_warmup_ratio=opt.lr_warmup_ratio,
            lr_start=opt.lr_start,
        )

    def clip_grad_norm(self, max_norm: Optional[float] = None, norm_type: float = 2.0):
        """Clip this model's gradients under its own parallelism and return the norm."""
        if max_norm is None:
            max_norm = self.model_args.optimizer.max_grad_norm
        with self.scoped():
            return veomni_clip_grad_norm(self.model, max_norm, norm_type)
