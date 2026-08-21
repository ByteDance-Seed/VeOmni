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

from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

import torch

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import MainCollator, build_chat_template, build_data_transform
from ..distributed.parallel_state import get_parallel_state, use_parallel_state
from ..distributed.torch_compile import (
    CompileConfig,
    mark_compile_step_begin,
    validate_compile_model,
    validate_compile_runtime,
)
from ..models import build_foundation_model, build_processor
from ..models.model_runtime import _collect_muon_kwargs
from ..optim import build_optimizer
from ..utils import helper
from ..utils.device import get_device_type, synchronize
from ..utils.loss_utils import count_loss_token, reduce_global_loss_token
from ..utils.model_utils import pretty_print_trainable_parameters
from .base import BaseTrainer, VeOmniIter


logger = helper.create_logger(__name__)
MAX_PIXELS = 768 * 28 * 28


def _get_vlm_visual_module(model):
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        if base_model is not model:
            return _get_vlm_visual_module(base_model)

    # Qwen-VL wrappers are not consistent across transformers versions:
    # older releases may expose `visual` directly on the conditional model
    # for backward compatibility, while newer ones only keep `model.visual`.
    visual = getattr(model, "visual", None)
    if visual is not None:
        return visual

    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        return getattr(inner_model, "visual", None)

    return None


@dataclass
class VLMTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether to freeze ViT parameters during full tuning; ignored when LoRA is enabled."},
    )
    freeze_audio_tower: bool = field(
        default=False,
        metadata={"help": "Whether to freeze audio tower parameters during full tuning; ignored with LoRA."},
    )
    vit_lr: float = field(
        default=1e-6,
        metadata={"help": "Maximum learning rate for vit parameters."},
    )


@dataclass
class VLMMDataArguments(DataArguments):
    supports_torch_compile = True
    mm_configs: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for multimodal input."},
    )


@dataclass
class VLMMModelArguments(ModelArguments):
    encoder_data_balance: Optional[bool] = field(
        default=False, metadata={"help": "Whether to balance encoder data for qwen3-vl model"}
    )
    encoder_data_balance_sorting_algo: Optional[str] = field(
        default="post_mbs_balancing_greedy_without_pad",
        metadata={
            "help": "The sorting algorithm of encoder data balance. All viable algorithms are defined in "
            "veomni/utils/data_balance/balance_sorting_algo.py, SORTING_ALGO_FUNC"
        },
    )


@dataclass
class VeOmniVLMArguments(VeOmniArguments):
    model: "VLMMModelArguments" = field(default_factory=VLMMModelArguments)
    data: "VLMMDataArguments" = field(default_factory=VLMMDataArguments)
    train: "VLMTrainingArguments" = field(default_factory=VLMTrainingArguments)


class VLMTrainer:
    def __init__(self, args: VeOmniVLMArguments):
        # BaseTrainer.__init__ is NOT called here; we call its private
        # helpers one-by-one so the sequence is explicit.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base._setup()  # registers ParallelState("base") before seed

        # All build steps read the current ParallelState via ``get_parallel_state()``
        # (meta-init, FSDP2/EP wrap + weight load, optimizer, SP data pipeline), so
        # scope the whole build under this trainer's own state. No-op for the
        # single-model case; keeps each module building over its own mesh once
        # multiple modules build separately.
        with use_parallel_state("base"):
            # rewrite build model to support data balancing
            self.build_model()
            self._validate_torch_compile()

            # rewrite freeze_model_module to support freeze multimodal encoder, etc.
            self.freeze_model()

            # rewrite build_model_assets to support chat_template and processor for multimodal datasets
            self._build_model_assets()

            # rewrite build_data_transform to support multimodal transform
            self._build_data_transform()

            self.base._build_dataset()

            # rewrite build_collate_fn to support multimodal collate_fn
            self._build_collate_fn()

            self.base._build_dataloader()
            self.base.build_parallelized_model()

            # rewrite build_optimizer to support different lr param groups
            self.build_optimizer()

            self.base.build_lr_scheduler()
            self.base._build_training_context()
            self.base._init_callbacks()

    def build_model(self):
        args: VeOmniVLMArguments = self.base.args
        logger.info_rank0("Build model")
        self.base.model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=args.model.model_path,
            torch_dtype="float32" if args.model.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
            init_device=args.model.accelerator.init_device,
            encoder_data_balance=args.model.encoder_data_balance,
            encoder_data_balance_sorting_algo=args.model.encoder_data_balance_sorting_algo,
            ops_implementation=args.model.ops_implementation,
            config_kwargs=args.model.model_config,
        )
        self.base.model_config = self.base.model.config

    def _validate_torch_compile(self):
        args: VeOmniVLMArguments = self.base.args
        if not args.model.accelerator.torch_compile.enable:
            return

        accelerator = args.model.accelerator
        compile_config = CompileConfig(
            **{
                field.name: getattr(args.model.accelerator.torch_compile, field.name)
                for field in fields(CompileConfig)
            }
        )
        validate_compile_model(
            self.base.model,
            compile_config,
            sequence_parallel_enabled=accelerator.ulysses_size > 1 or accelerator.cp_size > 1,
            async_enabled=accelerator.enable_async,
        )
        parallel_state = get_parallel_state()
        validate_compile_runtime(
            compile_config,
            device_type=get_device_type(),
            fsdp_enabled=parallel_state.fsdp_enabled,
            fsdp_mode=parallel_state.dp_mode,
            any_extra_parallel_enabled=parallel_state.any_extra_parallel_enabled,
            enable_reshard_after_forward=accelerator.fsdp_config.reshard_after_forward,
        )

    def freeze_model(self):
        args: VeOmniVLMArguments = self.base.args
        model_config = self.base.model_config
        lora_enabled = bool(args.model.lora_config)
        if model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            self.base.model.disable_talker()

        # VLMTrainer composes BaseTrainer instead of calling its constructor, so
        # it must opt into the shared LoRA setup explicitly. The wrapper freezes
        # all base weights and re-enables only matched adapter parameters.
        if lora_enabled:
            self.base.setup_lora()

        is_omni = model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe")
        visual = self.base.model.thinker.visual if is_omni else _get_vlm_visual_module(self.base.model)

        # LoRA setup is authoritative for trainability. It already freezes every
        # untargeted parameter, so the legacy tower flags apply only to full tuning.
        if not lora_enabled:
            if args.train.freeze_vit:
                if is_omni:
                    self.base.model.thinker.visual.requires_grad_(False)
                    # Preserve the existing full-tuning policy: freeze the
                    # visual backbone while continuing to train the merger.
                    self.base.model.thinker.visual.merger.requires_grad_(True)
                else:
                    # Resolve both flat and nested visual-module layouts to cover
                    # both the plain `model.visual` shape and Qwen3.5-VL's nested
                    # layout.
                    if visual is None:
                        raise AttributeError(f"Cannot find visual module for model_type={model_config.model_type}.")
                    visual.requires_grad_(False)

            if args.train.freeze_audio_tower and is_omni:
                self.base.model.thinker.audio_tower.requires_grad_(False)
                # Qwen2.5-Omni uses audio_tower.proj; Qwen3-Omni-MoE uses audio_tower.proj1.
                audio_proj = (
                    getattr(self.base.model.thinker.audio_tower, "proj1", None)
                    or self.base.model.thinker.audio_tower.proj
                )
                audio_proj.requires_grad_(True)

        pretty_print_trainable_parameters(self.base.model)
        helper.print_device_mem_info("VRAM usage after building model")

    def _build_model_assets(self):
        args: VeOmniVLMArguments = self.base.args
        self.base.processor = build_processor(args.model.tokenizer_path, max_pixels=MAX_PIXELS)
        if self.base.model_config.model_type not in ("qwen2_5_omni", "qwen3_omni_moe"):
            self.base.chat_template = build_chat_template(args.data.chat_template, self.base.processor)
            self.base.model_assets = [self.base.processor, self.base.chat_template]
        else:
            self.base.chat_template = None
            self.base.model_assets = [self.base.processor]

    def _build_data_transform(self):
        args: VeOmniVLMArguments = self.base.args
        model_type = self.base.model_config.model_type

        self.base.data_transform = build_data_transform(
            model_type,
            processor=self.base.processor,
            chat_template=self.base.chat_template,
            position_id_func=self.base.model.get_position_id_func(),
            **args.data.mm_configs,
        )

    def _build_collate_fn(self):
        model = self.base.model
        # The model owns its modality-specific collate topology — mirrors
        # get_position_id_func. Both hooks are optional capabilities: text
        # models / pipelines that don't wire them simply fall back (the ViT
        # forward keeps its in-forward derivation; see multimodal_metadata.md).
        #   * get_extra_collate_infos() — extra collate rules (e.g. omni audio
        #     feature tensors); replaces the former model_type hardcode here.
        #   * get_metadata_collate_func() — picklable CPU-side hook the collator
        #     runs after SP padding to derive multimodal_metadata.
        get_extra_infos = getattr(model, "get_extra_collate_infos", None)
        data_collate_info = get_extra_infos() if get_extra_infos is not None else {}
        get_metadata_func = getattr(model, "get_metadata_collate_func", None)
        metadata_collate_func = get_metadata_func() if get_metadata_func is not None else None

        seq_classification = self.base.args.data.data_type == "classification"
        pad_to_length = self.base.args.train.pad_to_length
        self.base.collate_fn = MainCollator(
            pad_to_length=pad_to_length,
            seq_classification=seq_classification,
            data_collate_info=data_collate_info,
            metadata_collate_func=metadata_collate_func,
        )

    def build_optimizer(self):
        args: VeOmniVLMArguments = self.base.args

        vit_params, other_params = [], []
        for name, param in self.base.model.named_parameters():
            if param.requires_grad:
                if "visual" in name:
                    vit_params.append(param)
                else:
                    other_params.append(param)

        # Only create groups that have trainable params. An empty visual group
        # has no optimizer state under DCP and would raise
        # KeyError: 'betas' on the first step after resume. VLMRLTrainer
        # inherits this method, so the guard covers both trainers.
        param_groups = []
        if vit_params:
            param_groups.append({"params": vit_params, "lr": args.train.vit_lr})
        if other_params:
            param_groups.append({"params": other_params, "lr": args.model.optimizer.lr})

        self.base.optimizer = build_optimizer(
            self.base.model,
            lr=args.model.optimizer.lr,
            weight_decay=args.model.optimizer.weight_decay,
            fused=True,
            optimizer_type=args.model.optimizer.type,
            param_groups=param_groups,
            no_decay_modules=args.model.optimizer.no_decay_modules,
            no_decay_params=args.model.optimizer.no_decay_params,
            muon_kwargs=_collect_muon_kwargs(args.model.optimizer),
        )

    def on_train_begin(self):
        self.base.on_train_begin()

    def on_train_end(self):
        self.base.on_train_end()

    def on_epoch_begin(self):
        self.base.on_epoch_begin()

    def on_epoch_end(self):
        self.base.on_epoch_end()

    def on_step_begin(self, micro_batches=None):
        self.base.on_step_begin(micro_batches=micro_batches)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
        self.base.on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    def train_step(
        self,
        data_iterator: Any,
    ) -> Dict[str, float]:
        self.base.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)

        self.on_step_begin(micro_batches=micro_batches)

        # Forward and backward for each micro batch
        self.base.sync_before_train_step()

        total_loss = 0.0
        total_loss_dict = defaultdict(int)

        # token num for fixed_ce_loss in postforward
        self.base.micro_batches_token_len = count_loss_token(micro_batches)
        self.base.global_micro_batches_token_len = reduce_global_loss_token(self.base.micro_batches_token_len)
        num_micro_steps = len(micro_batches)
        # forward and backward pass with gradient_accumulationsteps
        for micro_step, micro_batch in enumerate(micro_batches):
            mark_compile_step_begin(getattr(self.base.model, "_veomni_compile_uses_cuda_graphs", False))
            self.base.model_reshard(micro_step, num_micro_steps)
            self.base._configure_hsdp_allreduce(micro_step, num_micro_steps)
            loss: torch.Tensor
            loss_dict: Dict[str, torch.Tensor]
            # token num for fixed_ce_loss in postforward
            self.base.micro_batch_token_len = count_loss_token(micro_batch)
            loss, loss_dict = self.base.forward_backward_step(micro_batch)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item()

        # Gradient clipping (reads FSDP/EP groups from current ParallelState)
        grad_norm = self.base.clip_grad_norm()

        # Optimizer and scheduler step
        self.base.optimizer.step()
        self.base.lr_scheduler.step()
        self.base.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

    def train(self):
        args: VeOmniVLMArguments = self.base.args
        self.on_train_begin()
        logger.info(
            f"Rank{args.train.local_rank} Start training. "
            f"Start step: {self.base.start_step}. "
            f"Train steps: {args.train_steps}. "
            f"Start epoch: {self.base.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(self.base.start_epoch, args.train.num_train_epochs):
            if hasattr(self.base.train_dataloader, "set_epoch"):
                self.base.train_dataloader.set_epoch(epoch)
            self.base.state.epoch = epoch

            self.on_epoch_begin()

            # Create a batch generator
            self.base.data_iterator = VeOmniIter(
                self.base.train_dataloader, use_background_prefetcher=args.data.dataloader.use_background_prefetcher
            )

            for _ in range(self.base.start_step, args.train_steps):
                try:
                    self.train_step(self.base.data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.on_epoch_end()

            self.base.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
            if args.data.dataloader.use_background_prefetcher:
                self.base.data_iterator.stop()

        self.on_train_end()

        if args.data.dataloader.use_background_prefetcher:
            self.base.data_iterator.stop()

        synchronize()

        self.base.destroy_distributed()
