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
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

import torch

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import MainCollator, build_multimodal_chat_template
from ..data.multimodal.data_transform import (
    process_sample_qwen2_5_vl,
    process_sample_qwen3_vl,
    process_sample_qwen_omni,
)
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..models import build_foundation_model, build_processor
from ..optim import build_optimizer
from ..utils import helper
from ..utils.device import synchronize
from ..utils.loss_utils import count_loss_token
from ..utils.model_utils import pretty_print_trainable_parameters
from .base import BaseTrainer


logger = helper.create_logger(__name__)
MAX_PIXELS = 768 * 28 * 28


@dataclass
class VLMTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    freeze_audio_tower: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the audio tower parameters."},
    )
    vit_lr: float = field(
        default=1e-6,
        metadata={"help": "Maximum learning rate for vit parameters."},
    )


@dataclass
class VLMMDataArguments(DataArguments):
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

        self.base._setup()

        # rewrite build model to support data balancing
        self._build_model()

        # rewrite freeze_model_module to support freeze multimodal encoder, etc.
        self._freeze_model_module()

        # rewrite build_model_assets to support chat_template and processor for multimodal datasets
        self._build_model_assets()

        # rewrite build_data_transform to support multimodal transform
        self._build_data_transform()

        self.base._build_dataset()

        # rewrite build_collate_fn to support multimodal collate_fn
        self._build_collate_fn()

        self.base._build_dataloader()
        self.base._build_parallelized_model()

        # rewrite build_optimizer to support different lr param groups
        self._build_optimizer()

        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()

    def _build_model(self):
        args: VeOmniVLMArguments = self.base.args
        logger.info_rank0("Build model")
        self.base.model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=args.model.model_path,
            torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
            attn_implementation=args.model.attn_implementation,
            moe_implementation=args.model.moe_implementation,
            init_device=args.train.init_device,
            encoder_data_balance=args.model.encoder_data_balance,
            encoder_data_balance_sorting_algo=args.model.encoder_data_balance_sorting_algo,
        )
        self.base.model_config = self.base.model.config

    def _freeze_model_module(self):
        args: VeOmniVLMArguments = self.base.args
        model_config = self.base.model_config
        if model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            self.base.model.disable_talker()

        if args.train.freeze_vit:
            if model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
                self.base.model.thinker.visual.requires_grad_(False)
                self.base.model.thinker.visual.merger.requires_grad_(True)
            else:
                self.base.model.visual.requires_grad_(False)

        if args.train.freeze_audio_tower and model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            self.base.model.thinker.audio_tower.requires_grad_(False)
            self.base.model.thinker.audio_tower.proj.requires_grad_(True)

        pretty_print_trainable_parameters(self.base.model)
        helper.print_device_mem_info("VRAM usage after building model")

    def _build_model_assets(self):
        args: VeOmniVLMArguments = self.base.args
        self.base.processor = build_processor(args.model.tokenizer_path, max_pixels=MAX_PIXELS)
        if self.base.model_config.model_type not in ("qwen2_5_omni", "qwen3_omni_moe"):
            self.base.chat_template = build_multimodal_chat_template(
                args.data.chat_template, self.base.processor.tokenizer
            )
            self.base.model_assets = [self.base.processor, self.base.chat_template]
        else:
            self.base.chat_template = None
            self.base.model_assets = [self.base.processor]

    def _build_data_transform(self):
        args: VeOmniVLMArguments = self.base.args

        model_type = self.base.model_config.model_type

        if model_type in ("qwen3_vl", "qwen3_vl_moe"):
            process_function = process_sample_qwen3_vl
            position_id_func = self.base.model.get_position_id_func()
        elif model_type in ("qwen2_5_vl", "qwen2_vl"):
            process_function = process_sample_qwen2_5_vl
            position_id_func = self.base.model.get_position_id_func()
        elif model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            process_function = process_sample_qwen_omni
            position_id_func = self.base.model.thinker.get_position_id_func()
        else:
            raise NotImplementedError(f"Unsupported model type: {model_type}.")
        self.base.data_transform = partial(
            process_function,
            processor=self.base.processor,
            chat_template=self.base.chat_template,
            position_id_func=position_id_func,
            **args.data.mm_configs,
        )

    def _build_collate_fn(self):
        if self.base.model_config.model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
            data_collate_info = {
                "audio_feature_lengths": (0, False, None, None),
                "input_features": (0, True, 0, 1),
                "audio_mask": (-1, False, 0, 1),
            }
        else:
            data_collate_info = {}
        seq_classification = self.base.args.data.data_type == "classification"
        pad_to_length = self.base.args.train.pad_to_length
        self.base.collate_fn = MainCollator(
            pad_to_length=pad_to_length,
            seq_classification=seq_classification,
            data_collate_info=data_collate_info,
        )

    def _build_optimizer(self):
        args: VeOmniVLMArguments = self.base.args

        vit_params, other_params = [], []
        for name, param in self.base.model.named_parameters():
            if param.requires_grad:
                if "visual" in name:
                    vit_params.append(param)
                else:
                    other_params.append(param)

        param_groups = [{"params": vit_params, "lr": args.train.vit_lr}, {"params": other_params, "lr": args.train.lr}]

        # Build optimizer
        self.base.optimizer = build_optimizer(
            self.base.model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=True,
            optimizer_type=args.train.optimizer,
            param_groups=param_groups,
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
        args: VeOmniVLMArguments = self.base.args
        self.base.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)

        self.on_step_begin(micro_batches=micro_batches)

        # Forward and backward for each micro batch
        synchronize()

        total_loss = 0.0
        total_loss_dict = defaultdict(int)

        # token num for fixed_ce_loss in postforward
        self.base.micro_batches_token_len = count_loss_token(micro_batches)
        num_micro_steps = len(micro_batches)
        # forward and backward pass with gradient_accumulationsteps
        for micro_step, micro_batch in enumerate(micro_batches):
            self.base.model_reshard(micro_step, num_micro_steps)
            loss: torch.Tensor
            loss_dict: Dict[str, torch.Tensor]
            # token num for fixed_ce_loss in postforward
            self.base.micro_batch_token_len = count_loss_token(micro_batch)
            loss, loss_dict = self.base.forward_backward_step(micro_batch)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item()

        # Gradient clipping
        grad_norm = veomni_clip_grad_norm(self.base.model, args.train.max_grad_norm)

        # Optimizer and scheduler step
        self.base.optimizer.step()
        self.base.lr_scheduler.step()
        self.base.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

    def fit(self):
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
            data_iterator = iter(self.base.train_dataloader)

            for _ in range(self.base.start_step, args.train_steps):
                try:
                    self.train_step(data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                    break

            self.on_epoch_end()

            self.base.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        self.on_train_end()

        synchronize()

        self.base.destroy_distributed()
