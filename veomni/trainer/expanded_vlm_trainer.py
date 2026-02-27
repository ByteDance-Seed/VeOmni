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

"""
Expanded VLMTrainer using composition over inheritance.

Compared with the text trainer this file demonstrates:
  - VLM-specific model build (_build_vlm_model) that passes extra kwargs to
    build_foundation_model (encoder_data_balance, etc.)
  - Processor / multimodal chat template as model assets
  - VIT parameter groups (separate lr for visual encoder)
  - Freeze logic for ViT / audio tower
  - VLM-specific data transform dispatch (qwen2.5-vl, qwen3-vl, qwen-omni)
  - on_step_begin injects an extra VLM-specific action (encoder balance reset)
    without touching any shared base code

All lifecycle hooks remain explicit, mirroring expanded_text_trainer.py.
"""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import build_dataloader, build_dataset, build_multimodal_chat_template
from ..data.multimodal.data_transform import (
    process_sample_qwen2_5_vl,
    process_sample_qwen3_vl,
    process_sample_qwen_omni,
)
from ..models import build_foundation_model, build_processor
from ..trainer.callbacks.base import TrainerState
from ..trainer.callbacks.checkpoint_callback import CheckpointerCallback, HuggingfaceCkptCallback
from ..trainer.callbacks.evaluate_callback import EvaluateCallback
from ..trainer.callbacks.trace_callback import (
    EnvironMeterCallback,
    ProfileTraceCallback,
    TqdmCallback,
    WandbTraceCallback,
)
from ..utils import helper
from ..utils.device import synchronize
from ..utils.loss_utils import count_loss_token
from ..utils.model_utils import pretty_print_trainable_parameters
from .base import BaseTrainer


logger = helper.create_logger(__name__)
MAX_PIXELS = 768 * 28 * 28

_OMNI_MODEL_TYPES = ("qwen2_5_omni", "qwen3_omni_moe")


# ------------------------------------------------------------------ #
# Argument extensions (same as in vlm_trainer.py)                     #
# ------------------------------------------------------------------ #


@dataclass
class VLMTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(default=False, metadata={"help": "Freeze ViT parameters."})
    freeze_audio_tower: bool = field(default=False, metadata={"help": "Freeze audio tower parameters."})
    vit_lr: float = field(default=1e-6, metadata={"help": "LR for ViT parameters."})


@dataclass
class VLMDataArguments(DataArguments):
    mm_configs: Optional[Dict] = field(default_factory=dict, metadata={"help": "Config for multimodal input."})


@dataclass
class VLMModelArguments(ModelArguments):
    encoder_data_balance: Optional[bool] = field(default=False)
    encoder_data_balance_sorting_algo: Optional[str] = field(
        default="post_mbs_balancing_greedy_without_pad",
    )


@dataclass
class VLMArguments(VeOmniArguments):
    model: "VLMModelArguments" = field(default_factory=VLMModelArguments)
    data: "VLMDataArguments" = field(default_factory=VLMDataArguments)
    train: "VLMTrainingArguments" = field(default_factory=VLMTrainingArguments)


# ------------------------------------------------------------------ #
# Trainer                                                              #
# ------------------------------------------------------------------ #


class VLMTrainer:
    """
    Vision-Language Model trainer (Qwen2.5-VL / Qwen3-VL / Qwen-Omni).

    Holds BaseTrainer as a member.  Every build step and every lifecycle hook
    is called explicitly in __init__ and fit(), so there is no implicit logic
    hidden in a parent class.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, args: VLMArguments):
        self.args: VLMArguments = args
        logger.info_rank0(json.dumps(asdict(self.args), indent=2))

        # Allocate the infrastructure component without calling its __init__
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args
        self.base.start_epoch = 0
        self.base.start_step = 0
        self.base.train_steps = 0

        # ---- Ordered init sequence ---- #
        self.base._setup()

        # VLM model build (overrides base: extra kwargs for encoder balance)
        self._build_vlm_model()

        # Data — same pattern as text trainer; VLM-specific collate info
        self._build_data()

        self.base._build_parallelized_model()

        # Optimizer — VLM needs separate param groups for ViT
        self._build_optimizer_and_scheduler()

        self.base._build_training_context()
        self._init_callbacks()

        self.state = TrainerState()

    # ------------------------------------------------------------------ #
    # VLM-specific build helpers                                           #
    # ------------------------------------------------------------------ #

    def _build_vlm_model(self):
        """
        Build the VLM foundation model with extra VLM-specific arguments,
        then attach model assets and freeze requested submodules.
        """
        args: VLMArguments = self.args
        logger.info_rank0("Build VLM model")

        self.base.model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=args.model.model_path,
            torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
            attn_implementation=args.model.attn_implementation,
            moe_implementation=args.model.moe_implementation,
            init_device=args.train.init_device,
            config_kwargs=self.base.build_model_config_kwargs(),
            # VLM-specific
            encoder_data_balance=args.model.encoder_data_balance,
            encoder_data_balance_sorting_algo=args.model.encoder_data_balance_sorting_algo,
        )
        self.base.model_config = self.base.model.config

        # Build processor & optional chat template
        self._build_vlm_model_assets()

        # Freeze submodules before parallelism
        self._freeze_modules()

        pretty_print_trainable_parameters(self.base.model)
        helper.print_device_mem_info("VRAM usage after building VLM model")

    def _build_vlm_model_assets(self):
        """Build processor (and optionally multimodal chat template)."""
        args: VLMArguments = self.args
        self.processor = build_processor(args.model.tokenizer_path, max_pixels=MAX_PIXELS)

        self.base.model_assets = [self.base.model_config]

        if self.base.model_config.model_type in _OMNI_MODEL_TYPES:
            # Omni models: no separate chat template, processor is enough
            self.chat_template = None
            self.base.model_assets.append(self.processor)
        else:
            self.chat_template = build_multimodal_chat_template(args.data.chat_template, self.processor.tokenizer)
            self.base.model_assets.extend([self.processor, self.chat_template])

    def _freeze_modules(self):
        """Freeze ViT / audio tower according to training args."""
        args: VLMArguments = self.args
        model_type = self.base.model_config.model_type

        if model_type in _OMNI_MODEL_TYPES:
            self.base.model.disable_talker()

        if args.train.freeze_vit:
            if model_type in _OMNI_MODEL_TYPES:
                self.base.model.thinker.visual.requires_grad_(False)
                self.base.model.thinker.visual.merger.requires_grad_(True)
            else:
                self.base.model.visual.requires_grad_(False)

        if args.train.freeze_audio_tower and model_type in _OMNI_MODEL_TYPES:
            self.base.model.thinker.audio_tower.requires_grad_(False)
            self.base.model.thinker.audio_tower.proj.requires_grad_(True)

    def _build_data_collate_info(self) -> dict:
        model_type = self.base.model_config.model_type
        if model_type in _OMNI_MODEL_TYPES:
            return {
                "audio_feature_lengths": (0, False, None, None),
                "input_features": (0, True, 0, 1),
                "audio_mask": (-1, False, 0, 1),
            }
        return {}

    def _build_data_transform(self):
        args: VLMArguments = self.args
        model_type = self.base.model_config.model_type

        if model_type in ("qwen3_vl", "qwen3_vl_moe"):
            process_fn = process_sample_qwen3_vl
            position_id_func = self.base.model.get_position_id_func()
        elif model_type in ("qwen2_5_vl", "qwen2_vl"):
            process_fn = process_sample_qwen2_5_vl
            position_id_func = self.base.model.get_position_id_func()
        elif model_type in _OMNI_MODEL_TYPES:
            process_fn = process_sample_qwen_omni
            position_id_func = self.base.model.thinker.get_position_id_func()
        else:
            raise NotImplementedError(f"Unsupported model type: {model_type}.")

        return partial(
            process_fn,
            processor=self.processor,
            chat_template=self.chat_template,
            position_id_func=position_id_func,
            **args.data.mm_configs,
        )

    def _build_data(self):
        logger.info_rank0("Build VLM data")
        args: VLMArguments = self.args
        data_transform = self._build_data_transform()

        train_dataset = build_dataset(
            dataset_name=args.data.dataset_name,
            transform=data_transform,
            seed=args.train.seed,
            **asdict(args.data),
        )

        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type == "mapping":
            dataset_length = dataset_length / args.train.data_parallel_size
        args.compute_train_steps(dataset_length)
        self.base.train_steps = args.train_steps

        self.base.train_dataloader = build_dataloader(
            dataloader_type=args.data.dataloader_type,
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train_steps,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz=args.train.dyn_bsz,
            dyn_bsz_buffer_size=args.data.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
            seed=args.train.seed,
            build_collate_fn=True,
            collate_fn_kwargs={
                "data_collate_info": self._build_data_collate_info(),
                "pad_to_length": args.train.pad_to_length,
                "seq_classification": args.data.data_type == "classification",
            },
        )

    def _build_optimizer_and_scheduler(self):
        """Build optimizer with separate ViT / LLM param groups."""
        from ..optim import build_lr_scheduler, build_optimizer

        args: VLMArguments = self.args
        vit_params, other_params = [], []
        for name, param in self.base.model.named_parameters():
            if param.requires_grad:
                if "visual" in name:
                    vit_params.append(param)
                else:
                    other_params.append(param)

        param_groups = [
            {"params": vit_params, "lr": args.train.vit_lr},
            {"params": other_params, "lr": args.train.lr},
        ]

        self.base.optimizer = build_optimizer(
            self.base.model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=True,
            optimizer_type=args.train.optimizer,
            param_groups=param_groups,
        )
        self.base.lr_scheduler = build_lr_scheduler(
            self.base.optimizer,
            train_steps=args.train_steps * args.train.num_train_epochs,
            lr=args.train.lr,
            lr_min=args.train.lr_min,
            lr_decay_style=args.train.lr_decay_style,
            lr_decay_ratio=args.train.lr_decay_ratio,
            lr_warmup_ratio=args.train.lr_warmup_ratio,
            lr_start=args.train.lr_start,
        )

    # ------------------------------------------------------------------ #
    # Callback initialisation                                              #
    # ------------------------------------------------------------------ #

    def _init_callbacks(self):
        self.environ_meter_cb = EnvironMeterCallback(self.base)
        self.tqdm_cb = TqdmCallback(self.base)
        self.wandb_cb = WandbTraceCallback(self.base)
        self.profile_cb = ProfileTraceCallback(self.base)
        self.checkpointer_cb = CheckpointerCallback(self.base)
        self.hf_ckpt_cb = HuggingfaceCkptCallback(self.base)
        self.evaluate_cb = EvaluateCallback(self.base)

    # ------------------------------------------------------------------ #
    # Explicit lifecycle methods                                           #
    # ------------------------------------------------------------------ #

    def on_train_begin(self):
        self.checkpointer_cb.on_train_begin(self.state)
        self.hf_ckpt_cb.on_train_begin(self.state)
        self.wandb_cb.on_train_begin(self.state)
        self.profile_cb.on_train_begin(self.state)

    def on_train_end(self):
        self.hf_ckpt_cb.on_train_end(self.state)

    def on_epoch_begin(self):
        self.tqdm_cb.on_epoch_begin(self.state)

    def on_epoch_end(self):
        self.tqdm_cb.on_epoch_end(self.state)
        self.checkpointer_cb.on_epoch_end(self.state)
        self.hf_ckpt_cb.on_epoch_end(self.state)
        self.evaluate_cb.on_epoch_end(self.state)

    def on_step_begin(self, micro_batches: List[Dict[str, Any]]):
        # VLM-specific: encoder balance pre-step reset could go here
        self.environ_meter_cb.on_step_begin(self.state, micro_batches=micro_batches)

    def on_step_end(self, loss: float, loss_dict: Dict[str, float], grad_norm: float):
        self.environ_meter_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.tqdm_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.wandb_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.profile_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.checkpointer_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.hf_ckpt_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.evaluate_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #

    def fit(self):
        args = self.args
        base = self.base

        self.on_train_begin()

        logger.info(
            f"Rank{args.train.local_rank} Start training. "
            f"Start step: {base.start_step}. "
            f"Train steps: {args.train_steps}. "
            f"Start epoch: {base.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(base.start_epoch, args.train.num_train_epochs):
            if hasattr(base.train_dataloader, "set_epoch"):
                base.train_dataloader.set_epoch(epoch)
            self.state.epoch = epoch

            self.on_epoch_begin()

            data_iterator = iter(base.train_dataloader)

            for _ in range(base.start_step, args.train_steps):
                try:
                    self._train_step(data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                    break

            self.on_epoch_end()

            base.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        self.on_train_end()

        synchronize()
        base._destroy_distributed()

    def _train_step(self, data_iterator: Any):
        args = self.args
        base = self.base
        self.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)

        self.on_step_begin(micro_batches)

        synchronize()

        total_loss = 0.0
        total_loss_dict = defaultdict(int)

        base.micro_batches_token_len = count_loss_token(micro_batches)
        num_micro_steps = len(micro_batches)

        for micro_step, micro_batch in enumerate(micro_batches):
            base._model_reshard(micro_step, num_micro_steps)
            base.micro_batch_token_len = count_loss_token(micro_batch)
            loss, loss_dict = base.forward_backward_step(micro_batch)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item()

        from ..distributed.clip_grad_norm import veomni_clip_grad_norm

        grad_norm = veomni_clip_grad_norm(base.model, args.train.max_grad_norm)

        base.optimizer.step()
        base.lr_scheduler.step()
        base.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)
