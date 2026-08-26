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

from dataclasses import fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import PretrainedConfig

from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.parallel_state import (
    get_parallel_state_by_name,
    use_parallel_state,
)
from ..utils import helper, logging


if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer

    from ..arguments import ModelRuntimeArguments, TrainingArguments
    from ..data.chat_template import ChatTemplate
    from ..trainer.callbacks import TrainerState
    from .checkpoint_manager import ModelCheckpointManager


logger = logging.get_logger(__name__)


def _has_trainable_lora_parameters(module: Optional[torch.nn.Module]) -> bool:
    if module is None:
        return False
    return any(
        param.requires_grad and ({"lora_A", "lora_B"} & set(name.split(".")))
        for name, param in module.named_parameters()
    )


class VeOmniModelRuntime:
    """One model's training unit: build → freeze/LoRA → parallelize → optimizer → clip.

    This is a *model handle*, not a trainer base class. A trainer holds one
    (``trainer.model``) the way :class:`~veomni.trainer.omni.omni_trainer.OmniTrainer`
    holds an ``OmniModelRuntime``, and drives it as ``self.model.build_model()``,
    ``self.model.save_dcp(...)``. The wrapped :class:`torch.nn.Module`
    lives at :attr:`model`; every other module API is forwarded by
    :meth:`__getattr__`, so ``trainer.model.parameters()`` and
    ``trainer.model.config`` keep working while the handle itself owns the
    training lifecycle.

    Handing the *handle* to code that demands a real ``nn.Module`` would fail an
    ``isinstance`` check, so the runtime owns every such boundary itself. Reading
    and writing checkpoints is one: :meth:`load`, :meth:`save_dcp` and
    :meth:`save_hf_or_lora` say *what* this model persists, the
    :class:`~veomni.models.checkpoint_manager.ModelCheckpointManager` at
    :attr:`checkpoint` says *how*, and *when* stays in the trainer callbacks.

    Constructing one *builds* it — ``VeOmniModelRuntime(args)`` comes back with a
    parallelized model and its optimizer, because everything that build touches
    (meta-init, FSDP2/TP/EP wrap, weight load, EP-aware optimizer) reads the
    ambient :class:`ParallelState`, and the only place that can be scoped once,
    for every caller, is inside the runtime that owns the mesh. A trainer is then
    free of it: ``self.model = self.build_model_runtime()`` and the job-level
    steps that follow need no scope of their own.

    Construction takes this model's *own* arguments — not the job's — plus the
    :attr:`model_name` its :class:`ParallelState` registers under, so a job
    composing several models hands each one its own slice and its own mesh
    without any of them having to know how to find itself inside a larger
    config. ``train`` comes alongside because a handful of decisions are
    genuinely job-wide: where checkpoints are written, and whether a resume path
    makes the initial HF weight load redundant (see :attr:`skip_hf_weight_load`).
    ``chat_template_name`` likewise — the job picks it (``data.chat_template``),
    but only the runtime holds the preprocessor to build it from.

    A model whose build differs (a VLM freezing its tower, a DiT carrying a
    condition model) subclasses this and overrides the step that differs, rather
    than the trainer resequencing the build from outside.

    The lr scheduler is the one piece deliberately left out: it needs
    ``total_steps``, which is only known once the dataset has been built, so the
    trainer calls :meth:`build_lr_scheduler` later.
    """

    args: "ModelRuntimeArguments"
    model_name: str
    model: Optional[torch.nn.Module] = None
    model_config: PretrainedConfig = PretrainedConfig()
    train: "TrainingArguments"
    optimizer: Optional["Optimizer"] = None
    lr_scheduler: Optional["LRScheduler"] = None
    # Class-level so __getattr__ never fires for them before the build runs.
    # ``checkpoint`` is only ``None`` during that build: __init__ installs one
    # unconditionally, so everything downstream may reach through it.
    checkpoint: Optional["ModelCheckpointManager"] = None
    tokenizer: Optional[Any] = None
    processor: Optional[Any] = None
    chat_template_name: Optional[str] = None
    chat_template: Optional["ChatTemplate"] = None
    model_assets: List[Any] = []

    def __init__(
        self,
        args: "ModelRuntimeArguments",
        model_name: str = "base",
        *,
        train: "TrainingArguments",
        chat_template_name: Optional[str] = None,
    ):
        self.args = args
        self.model_name = model_name
        self.train = train
        self.chat_template_name = chat_template_name
        self.setup()
        with use_parallel_state(self.model_name):
            self.build_model()
            self.freeze_model()
            self.build_parallelized_model()
            self.build_optimizer()
        self.build_model_assets()
        self.build_checkpoint()

    # ── Base model runtime functions ────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        """Forward unshadowed :class:`torch.nn.Module` APIs to the wrapped model."""
        model = object.__getattribute__(self, "model")
        if model is None:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}, and its model is not built yet.")
        return getattr(model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped model's forward.

        Python looks dunders up on the type, so :meth:`__getattr__` never sees
        ``trainer.model(**batch)`` — it has to be spelled out.
        """
        return self.model(*args, **kwargs)

    # ── Model runtime property accessors ────────────────────────────────

    def setup(self) -> None:
        """Build this model's device mesh and register it under :attr:`model_name`.

        The process group itself is job-bound and must already be initialised by
        :meth:`BaseTrainer.setup_distributed`; this only derives the model's own
        mesh from its accelerator config, which is why sibling models in one job
        can hold different ones.
        """
        from ..distributed.parallel_state import init_parallel_state_from_accelerator

        init_parallel_state_from_accelerator(self.args.accelerator, self.model_name)

    @property
    def parallel_state(self):
        """This model's :class:`ParallelState`, looked up by name on every access.

        A read-only lookup rather than a stored handle: the registry stays the
        single source of truth, so a state re-registered mid-job is picked up
        instead of silently serving a stale mesh.
        """
        return get_parallel_state_by_name(self.model_name)

    @property
    def skip_hf_weight_load(self) -> bool:
        """Whether the initial HF weight materialization can be skipped.

        A full non-LoRA resume already carries model weights, so materializing
        the HF checkpoint only to overwrite it costs a second memory peak that
        can OOM large MoE jobs. LoRA resumes still need the HF base.
        """
        from ..utils.checkpoint_utils import should_skip_hf_weight_load

        return should_skip_hf_weight_load(self.train.checkpoint.load_path, self.args.lora_config)

    # ── Model runtime build functions ────────────────────────────────

    def build_model(self) -> None:
        """Meta-init the model from its config via the registry-aware loader."""
        from .auto import build_foundation_model

        args = self.args
        logger.info_rank0("Build model")
        self.model = build_foundation_model(
            config_path=args.config_path,
            weights_path=args.model_path,
            torch_dtype="float32" if args.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
            init_device=args.accelerator.init_device,
            ops_implementation=args.ops_implementation,
            config_kwargs=args.model_config,
        )
        self.model_config = self.model.config

    def build_model_assets(self) -> None:
        """Load the preprocessor this model reads its inputs through.

        Also assembles :attr:`model_assets`, the sidecars an export writes beside
        this model's weights, and :attr:`chat_template` when the job named one.
        The config is always among the sidecars; the preprocessor joins it if
        there was one to load. The chat template is absent from that list by
        design: it is a *choice about the data*, not a property of the
        checkpoint, and an export writes what the checkpoint is.

        ``processor_config`` overrides what the repository ships, the way
        ``model_config`` does for the architecture — a pixel budget, say. It is
        a job-level knob rather than a runtime hook because the value belongs to
        the run, not to the model class.

        Which preprocessor it is follows from what the checkpoint actually
        holds, not from a declaration the model makes about itself.
        ``AutoProcessor`` falls back to ``AutoTokenizer`` when a repository has
        no processor to offer, so one load answers both cases — and the answer
        is trustworthy in a way a declaration is not, since a multimodal
        repository missing its ``preprocessor_config.json`` would return a bare
        tokenizer either way.

        :attr:`processor` therefore stays ``None`` unless a real processor came
        back, which is what the rest of the job reads as "this model sees more
        than text". Its tokenizer is taken from inside it rather than loaded a
        second time, so the object the data pipeline reads through is the same
        object that gets exported. A model that reads no text at all (a DiT over
        latents) overrides this to load nothing.

        A path with no preprocessor to load is not fatal here — a toy config
        used to exercise the training loop on synthetic batches has none, and
        never asks for one. The warning names the path, and a job that does read
        text fails where it reads it.

        The template is the third thing a model needs before it can read text:
        the tokenizer says how a string becomes ids, the processor how pixels
        do, and the template how a *conversation* becomes a training sample —
        including the assistant-only label mask that no jinja can express. A
        trainer therefore never assembles one; it reads :attr:`chat_template`
        the way it reads :attr:`tokenizer`. Stays ``None`` when the job names
        none (plaintext, diffusion, a Qwen-Omni model that formats through its
        processor) and, with a warning, when a name *was* given but nothing
        loaded to build it from. A job that needs a template and named none
        fails where it uses one, in the data transform.
        """
        from transformers.processing_utils import ProcessorMixin

        from .auto import build_processor

        self.model_assets = [self.model_config]

        path = self.args.tokenizer_path
        try:
            loaded = build_processor(path, **(self.args.processor_config or {}))
        except Exception as e:  # noqa: BLE001 — surfaced later by whoever reads text
            logger.warning_once(f"{type(self).__name__}: no preprocessor loaded from {path}: {e}.")
            loaded = None

        if loaded is not None:
            if isinstance(loaded, ProcessorMixin):
                self.processor = loaded
                self.tokenizer = loaded.tokenizer
            else:
                self.tokenizer = loaded
            self.model_assets.append(loaded)

        if not self.chat_template_name:
            return

        preprocessor = self.processor or self.tokenizer
        if preprocessor is None:
            logger.warning_once(
                f"{type(self).__name__}: chat template {self.chat_template_name!r} was requested but no "
                "preprocessor loaded to build it from; leaving it unset."
            )
            return

        from ..data.chat_template import build_chat_template

        self.chat_template = build_chat_template(self.chat_template_name, preprocessor)

    def build_parallelized_model(self) -> None:
        """FSDP2/DDP-wrap the model and load its weights.

        The wrap preserves ``requires_grad`` (the shard inherits it) and the
        loader writes weights in place, so a freeze applied in
        :meth:`freeze_model` survives and is not re-asserted here.
        """
        args = self.args

        # Customized parallelize model.
        customized_parallelize_model_function = getattr(self.model, "build_parallelize_model", None)
        if callable(customized_parallelize_model_function):
            parallelized_model = customized_parallelize_model_function(
                weights_path=self.args.model_path, args=self.args
            )
            if parallelized_model is not None:
                self.model = parallelized_model
                logger.info_rank0("Built customized parallelized model.")
                return

        # Default parallelize model.
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

        skip_hf_weight_load = self.skip_hf_weight_load
        if skip_hf_weight_load:
            logger.info_rank0(
                f"Checkpoint resume enabled (load_path={self.train.checkpoint.load_path}); "
                "skipping HF weight materialization before checkpoint restore."
            )

        from ..distributed.torch_compile import CompileConfig
        from ..distributed.torch_parallelize import build_parallelize_model

        compile_config = CompileConfig(
            **{field.name: getattr(args.accelerator.torch_compile, field.name) for field in fields(CompileConfig)}
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
            compile_config=compile_config,
            **kwargs,
        )
        self.model.train()

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
        lora_config = self.args.lora_config
        if not bool(lora_config):
            return

        # Customized LoRA model setup.
        customized_setup_lora_function = getattr(self.model, "setup_lora", None)
        if callable(customized_setup_lora_function):
            customized_lora_model = customized_setup_lora_function(lora_config)
            if customized_lora_model is not None:
                self.model = customized_lora_model
                logger.info_rank0("Setup customized LoRA model.")
                return

        # Default LoRA model setup.
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
        """Let the model freeze itself, apply LoRA, and report what is left trainable.

        Order matters: LoRA runs second because it is authoritative — it freezes
        every parameter it did not target, so a model-level policy applied
        afterwards would be fighting it.
        """
        # Customized freeze model.
        freeze_model_function = getattr(self.model, "freeze_model", None)
        if callable(freeze_model_function):
            freeze_model_function()

        self.setup_lora()

        from ..utils.model_utils import pretty_print_trainable_parameters

        pretty_print_trainable_parameters(self.model)
        helper.print_device_mem_info("VRAM usage after building model")

    # ── Model runtime optimizer & lr_scheduler build functions ────────────────────────────────
    def build_optimizer(self, param_groups: Optional[List[Dict[str, Any]]] = None) -> None:
        """Build the optimizer over this model's still-trainable params.

        A distributed optimizer (Muon) reads ``get_parallel_state()`` at build
        time, so it must resolve to *this* model's mesh.

        Args:
            param_groups: Split the parameters across groups with their own
                hyperparameters — a VLM giving its vision tower a separate lr,
                say. Which parameters belong together is knowledge the model's
                trainer has, so it is passed in rather than derived here.
        """
        opt = self.args.optimizer
        from ..optim import build_optimizer

        self.optimizer = build_optimizer(
            self.model,
            lr=opt.lr,
            weight_decay=opt.weight_decay,
            fused=True,
            optimizer_type=opt.type,
            param_groups=param_groups,
            no_decay_modules=opt.no_decay_modules,
            no_decay_params=opt.no_decay_params,
            optimizer_config=opt,
        )

    def build_lr_scheduler(self, total_steps: int) -> None:
        """Build the lr-scheduler over ``total_steps``.

        Takes the step count rather than reading it off a training config: it
        is only known once the dataset has been sized, which is job-bound.
        """
        opt = self.args.optimizer
        from ..optim import build_lr_scheduler

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

    def clip_grad_norm(self, max_norm: Optional[float] = None):
        """Clip this model's gradients under its own parallelism and return the norm."""
        if max_norm is None:
            max_norm = self.args.optimizer.max_grad_norm
        with use_parallel_state(self.model_name):
            return veomni_clip_grad_norm(self.model, max_norm)

    # ── Model runtime checkpoint build functions ────────────────────────────────
    def build_checkpoint(self) -> None:
        """Attach the component that checkpoints this model.

        Part of the build like every other step: a model that came back from
        construction unable to save itself would be half-built.
        """
        from .checkpoint_manager import ModelCheckpointManager

        self.checkpoint = ModelCheckpointManager(self, self.train.checkpoint)

    def load(self) -> None:
        """Restore this model and its optimizer from the configured load path.

        A composed model overrides this to fan out over its modules; the caller
        above only ever asks the model to load itself.
        """
        self.checkpoint.load()

    def save_dcp(self, state: "TrainerState") -> None:
        """Write this model's resumable checkpoint for ``state.global_step``."""
        self.checkpoint.save_dcp(state)

    def save_hf_or_lora(self, state: "TrainerState", stage: str = "step_end") -> None:
        """Export this model in whichever format it was trained in."""
        self.checkpoint.save_hf_or_lora(state, stage=stage)

    def save_model_assets(self) -> None:
        """Write the tokenizer/processor/config sidecars that an export needs."""
        import torch.distributed as dist

        from .module_utils import save_model_assets as write_model_assets

        if self.train.global_rank == 0:
            write_model_assets(self.train.checkpoint.model_assets_dir, self.model_assets)
        dist.barrier()
