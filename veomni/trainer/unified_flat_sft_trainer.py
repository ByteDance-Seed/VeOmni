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
Flat SFT Trainer — a single class covering both text and VLM training.

Design goal
-----------
No class hierarchy.  All logic is in one place, with explicit conditional
branches where text and VLM genuinely diverge.  A reader can follow the
entire training lifecycle from top to bottom in this file without jumping
to parent classes or tracing overrides.

The constructor receives `is_vlm: bool` which drives every branching
decision.  Every branch is labelled at the point it appears, so "what
does VLM do differently here?" is always a one-line grep away.

Branching map (all divergences, where they appear)
---------------------------------------------------
  is_vlm
    ├── _build_model_and_assets()
    │     ├── True  → build_foundation_model with encoder-balance kwargs
    │     │           + build_processor + optional multimodal_chat_template
    │     │           (sub-branch: model_type in _OMNI_MODEL_TYPES)
    │     └── False → base._build_model() + build_tokenizer
    │                 (sub-branch: data_type == "plaintext" vs "conversation")
    │
    ├── _freeze_modules()
    │     ├── True  → disable_talker / freeze ViT / freeze audio tower
    │     │           (sub-branches: model_type, freeze_vit, freeze_audio_tower flags)
    │     └── False → no-op
    │
    ├── _build_data_transform()
    │     ├── True  → dispatch on model_type (qwen3_vl / qwen2_5_vl / omni)
    │     └── False → dispatch on data_type  (plaintext / conversation)
    │
    ├── _build_data_collate_info()
    │     ├── True  → Omni audio collate descriptors (if model_type in _OMNI)
    │     └── False → {}
    │
    └── _build_optimizer_and_scheduler()
          ├── True  → split ViT / LLM param groups with separate lr
          └── False → single param group via base helper
"""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import build_dataloader, build_dataset
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
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
from .base import BaseTrainer


logger = helper.create_logger(__name__)

_MAX_PIXELS = 768 * 28 * 28
_OMNI_MODEL_TYPES = ("qwen2_5_omni", "qwen3_omni_moe")


# ── VLM argument extensions ──────────────────────────────────────────────── #
# Kept in this file so VLMArguments is importable from a single place.


@dataclass
class VLMTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(default=False)
    freeze_audio_tower: bool = field(default=False)
    vit_lr: float = field(default=1e-6)


@dataclass
class VLMDataArguments(DataArguments):
    mm_configs: Optional[Dict] = field(default_factory=dict)


@dataclass
class VLMModelArguments(ModelArguments):
    encoder_data_balance: Optional[bool] = field(default=False)
    encoder_data_balance_sorting_algo: Optional[str] = field(default="post_mbs_balancing_greedy_without_pad")


@dataclass
class VLMArguments(VeOmniArguments):
    model: "VLMModelArguments" = field(default_factory=VLMModelArguments)
    data: "VLMDataArguments" = field(default_factory=VLMDataArguments)
    train: "VLMTrainingArguments" = field(default_factory=VLMTrainingArguments)


# ── Trainer ───────────────────────────────────────────────────────────────── #


class FlatSFTTrainer:
    """
    Single-class SFT trainer for both text and VLM modalities.

    Pass is_vlm=True for any vision-language model (Qwen2.5-VL, Qwen3-VL,
    Qwen-Omni, …); pass is_vlm=False for plain-text LLM training.

    Use VLMArguments when is_vlm=True, VeOmniArguments when is_vlm=False.
    """

    def __init__(self, args: VeOmniArguments, *, is_vlm: bool):
        self.args = args
        self.is_vlm = is_vlm
        logger.info_rank0(json.dumps(asdict(self.args), indent=2))

        # BaseTrainer is used as a component (composition, not inheritance).
        # We bypass its __init__ and call each build helper explicitly so the
        # full construction sequence is visible here.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args
        self.base.start_epoch = 0
        self.base.start_step = 0
        self.base.train_steps = 0

        # ── ordered init sequence ──────────────────────────────────────── #
        self.base._setup()
        self._build_model_and_assets()  # diverges: text vs VLM
        self._freeze_modules()  # diverges: VLM only
        self._build_data()  # shared scaffold; transform diverges
        self.base._build_parallelized_model()
        self._build_optimizer_and_scheduler()  # diverges: VLM splits ViT lr
        self.base._build_training_context()
        self._init_callbacks()
        self.state = TrainerState()

    # ─────────────────────────────────────────────────────────────────────── #
    #  Build helpers                                                           #
    # ─────────────────────────────────────────────────────────────────────── #

    def _build_model_and_assets(self) -> None:
        if self.is_vlm:
            self._build_vlm_model_and_assets()
        else:
            self._build_text_model_and_assets()

    def _build_text_model_and_assets(self) -> None:
        from ..data import build_chat_template
        from ..models import build_tokenizer

        args = self.args
        self.base._build_model()

        self.tokenizer = build_tokenizer(args.model.tokenizer_path)
        self.base.model_assets = [self.base.model_config]

        if args.data.data_type == "plaintext":
            self.base.model_assets.append(self.tokenizer)
        else:  # "conversation" / sft
            self.chat_template = build_chat_template(args.data.chat_template, self.tokenizer)
            self.base.model_assets.append(self.chat_template)

    def _build_vlm_model_and_assets(self) -> None:
        from ..data import build_multimodal_chat_template
        from ..models import build_foundation_model, build_processor
        from ..utils.model_utils import pretty_print_trainable_parameters

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
            # VLM-specific: encoder data balance for qwen3-vl
            encoder_data_balance=args.model.encoder_data_balance,
            encoder_data_balance_sorting_algo=args.model.encoder_data_balance_sorting_algo,
        )
        self.base.model_config = self.base.model.config

        self.processor = build_processor(args.model.tokenizer_path, max_pixels=_MAX_PIXELS)
        self.base.model_assets = [self.base.model_config]

        if self.base.model_config.model_type in _OMNI_MODEL_TYPES:
            # Omni: no separate chat template; processor covers tokenization
            self.chat_template = None
            self.base.model_assets.append(self.processor)
        else:
            self.chat_template = build_multimodal_chat_template(args.data.chat_template, self.processor.tokenizer)
            self.base.model_assets.extend([self.processor, self.chat_template])

        pretty_print_trainable_parameters(self.base.model)
        helper.print_device_mem_info("VRAM usage after building VLM model")

    def _freeze_modules(self) -> None:
        if not self.is_vlm:
            return  # text training: nothing to freeze

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

    def _build_data_transform(self):
        if self.is_vlm:
            return self._build_vlm_data_transform()
        else:
            return self._build_text_data_transform()

    def _build_text_data_transform(self):
        from ..data.data_transform import process_pretrain_example, process_sft_example

        args = self.args
        if args.data.data_type == "plaintext":
            return partial(
                process_pretrain_example,
                tokenizer=self.tokenizer,
                max_seq_len=args.data.max_seq_len,
                text_keys=args.data.text_keys,
            )
        elif args.data.data_type == "conversation":
            return partial(
                process_sft_example,
                chat_template=self.chat_template,
                max_seq_len=args.data.max_seq_len,
                text_keys=args.data.text_keys,
            )
        else:
            raise NotImplementedError(f"Unsupported text data type: {args.data.data_type}.")

    def _build_vlm_data_transform(self):
        from ..data.multimodal.data_transform import (
            process_sample_qwen2_5_vl,
            process_sample_qwen3_vl,
            process_sample_qwen_omni,
        )

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
            raise NotImplementedError(f"Unsupported VLM model type: {model_type}.")

        return partial(
            process_fn,
            processor=self.processor,
            chat_template=self.chat_template,
            position_id_func=position_id_func,
            **args.data.mm_configs,
        )

    def _build_data_collate_info(self) -> dict:
        if not self.is_vlm:
            return {}
        model_type = self.base.model_config.model_type
        if model_type in _OMNI_MODEL_TYPES:
            return {
                "audio_feature_lengths": (0, False, None, None),
                "input_features": (0, True, 0, 1),
                "audio_mask": (-1, False, 0, 1),
            }
        return {}

    def _build_data(self) -> None:
        logger.info_rank0("Build data")
        args = self.args
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

    def _build_optimizer_and_scheduler(self) -> None:
        if not self.is_vlm:
            # Text: single param group, delegate to base
            self.base._build_optimizer_and_scheduler()
            return

        # VLM: separate lr for ViT visual encoder vs the rest
        from ..optim import build_lr_scheduler, build_optimizer

        args: VLMArguments = self.args
        vit_params, other_params = [], []
        for name, param in self.base.model.named_parameters():
            if param.requires_grad:
                (vit_params if "visual" in name else other_params).append(param)

        self.base.optimizer = build_optimizer(
            self.base.model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=True,
            optimizer_type=args.train.optimizer,
            param_groups=[
                {"params": vit_params, "lr": args.train.vit_lr},
                {"params": other_params, "lr": args.train.lr},
            ],
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

    # ─────────────────────────────────────────────────────────────────────── #
    #  Callbacks                                                               #
    # ─────────────────────────────────────────────────────────────────────── #

    def _init_callbacks(self) -> None:
        self.environ_meter_cb = EnvironMeterCallback(self.base)
        self.tqdm_cb = TqdmCallback(self.base)
        self.wandb_cb = WandbTraceCallback(self.base)
        self.profile_cb = ProfileTraceCallback(self.base)
        self.checkpointer_cb = CheckpointerCallback(self.base)
        self.hf_ckpt_cb = HuggingfaceCkptCallback(self.base)
        self.evaluate_cb = EvaluateCallback(self.base)

    # ─────────────────────────────────────────────────────────────────────── #
    #  Explicit lifecycle hooks                                                #
    # ─────────────────────────────────────────────────────────────────────── #

    def on_train_begin(self) -> None:
        self.checkpointer_cb.on_train_begin(self.state)  # load ckpt if configured
        self.hf_ckpt_cb.on_train_begin(self.state)  # save model assets to output dir
        self.wandb_cb.on_train_begin(self.state)  # init W&B run
        self.profile_cb.on_train_begin(self.state)  # start profiler

    def on_train_end(self) -> None:
        self.hf_ckpt_cb.on_train_end(self.state)  # save final HF weights

    def on_epoch_begin(self) -> None:
        self.tqdm_cb.on_epoch_begin(self.state)  # open tqdm bar

    def on_epoch_end(self) -> None:
        self.tqdm_cb.on_epoch_end(self.state)  # close tqdm bar
        self.checkpointer_cb.on_epoch_end(self.state)  # maybe save distributed ckpt
        self.hf_ckpt_cb.on_epoch_end(self.state)  # maybe save HF ckpt
        self.evaluate_cb.on_epoch_end(self.state)  # maybe run eval

    def on_step_begin(self, micro_batches: List[Dict[str, Any]]) -> None:
        self.environ_meter_cb.on_step_begin(self.state, micro_batches=micro_batches)

    def on_step_end(self, loss: float, loss_dict: Dict[str, float], grad_norm: float) -> None:
        self.environ_meter_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.tqdm_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.wandb_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.profile_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.checkpointer_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.hf_ckpt_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.evaluate_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    # ─────────────────────────────────────────────────────────────────────── #
    #  Training loop                                                           #
    # ─────────────────────────────────────────────────────────────────────── #

    def fit(self) -> None:
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

    def _train_step(self, data_iterator: Any) -> None:
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

        grad_norm = veomni_clip_grad_norm(base.model, args.train.max_grad_norm)

        base.optimizer.step()
        base.lr_scheduler.step()
        base.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)
