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
Expanded TextTrainer using composition over inheritance.

Design goals:
  1. Explicit callback dispatch: each on_* method lists exactly which callbacks
     fire at that lifecycle stage, making the training loop self-documenting.
  2. Composition over inheritance: BaseTrainer is a member, not a parent class.
     All init/build steps are called explicitly, so the reader can see the full
     construction order in one place without chasing overrides across a class
     hierarchy.
  3. Flat fit loop: the training loop lives entirely in this file; no hidden
     super() calls that can silently change behaviour in a subclass.
"""

import json
from collections import defaultdict
from dataclasses import asdict
from functools import partial
from typing import Any, Dict, List

from ..arguments import VeOmniArguments
from ..data import build_chat_template, build_dataloader, build_dataset
from ..data.data_transform import process_pretrain_example, process_sft_example
from ..models import build_tokenizer
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


class TextTrainer:
    """
    Text-only SFT / pretrain trainer.

    Owns the full init sequence and fit loop so that every step is visible
    in this file.  BaseTrainer is used as a *component* that provides shared
    infrastructure (distributed setup, model parallelism, optimizer, etc.).
    """

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, args: VeOmniArguments):
        self.args: VeOmniArguments = args
        logger.info_rank0(json.dumps(asdict(self.args), indent=2))

        # ---- Instantiate the infrastructure component ---- #
        # BaseTrainer.__init__ is NOT called here; we call its private
        # helpers one-by-one so the sequence is explicit.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        # Shared state that callbacks and train_step read from self.base
        self.base.start_epoch = 0
        self.base.start_step = 0
        self.base.train_steps = 0

        # ---- Ordered init sequence ---- #
        self.base._setup()

        # Model (text: no processor, just tokenizer)
        self.base._build_model()
        self._build_text_model_assets()  # text-specific: tokenizer / chat_template

        # Data  ← text-specific; we own this method entirely
        self._build_data()

        self.base._build_parallelized_model()
        self.base._build_optimizer_and_scheduler()
        self.base._build_training_context()

        # Callbacks  ← constructed after everything else so they can
        # access self.base.model, self.base.train_dataloader, etc.
        self._init_callbacks()

        self.state = TrainerState()

    # ------------------------------------------------------------------ #
    # Text-specific build helpers                                          #
    # ------------------------------------------------------------------ #

    def _build_text_model_assets(self):
        """Build tokenizer (and optionally chat template) for text models."""
        args = self.args
        self.tokenizer = build_tokenizer(args.model.tokenizer_path)
        self.base.model_assets = [self.base.model_config]

        if args.data.data_type == "plaintext":
            self.base.model_assets.append(self.tokenizer)
        else:
            self.chat_template = build_chat_template(args.data.chat_template, self.tokenizer)
            self.base.model_assets.append(self.chat_template)

    def _build_data_transform(self):
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
            raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    def _build_data(self):
        """Build text dataset and dataloader (replaces BaseTrainer._build_data)."""
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

        # Expose on self.base so that CheckpointerCallback / EnvironMeterCallback
        # can find it via trainer.train_dataloader
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
                "data_collate_info": {},
                "pad_to_length": args.train.pad_to_length,
                "seq_classification": args.data.data_type == "classification",
            },
        )

    # ------------------------------------------------------------------ #
    # Callback initialisation                                              #
    # ------------------------------------------------------------------ #

    def _init_callbacks(self):
        """
        Instantiate every callback that participates in the text training loop.

        Keeping them as named attributes (rather than a bare list) makes it
        trivial to see which callbacks exist and to override individual ones
        in a subclass.
        """
        # Trace callbacks
        self.environ_meter_cb = EnvironMeterCallback(self.base)
        self.tqdm_cb = TqdmCallback(self.base)
        self.wandb_cb = WandbTraceCallback(self.base)
        self.profile_cb = ProfileTraceCallback(self.base)

        # Checkpoint callbacks
        self.checkpointer_cb = CheckpointerCallback(self.base)
        self.hf_ckpt_cb = HuggingfaceCkptCallback(self.base)

        # Evaluation callback
        self.evaluate_cb = EvaluateCallback(self.base)

    # ------------------------------------------------------------------ #
    # Explicit lifecycle methods                                           #
    # The reader can see at a glance which callbacks fire at each stage.   #
    # ------------------------------------------------------------------ #

    def on_train_begin(self):
        # Load distributed checkpoint (if load_checkpoint_path is set)
        self.checkpointer_cb.on_train_begin(self.state)
        # Save model config / tokenizer / chat_template to output dir
        self.hf_ckpt_cb.on_train_begin(self.state)
        # Initialise W&B run
        self.wandb_cb.on_train_begin(self.state)
        # Start profiler (if enabled)
        self.profile_cb.on_train_begin(self.state)

    def on_train_end(self):
        # Save final HF weights (if save_hf_weights=True)
        self.hf_ckpt_cb.on_train_end(self.state)

    def on_epoch_begin(self):
        # Render tqdm progress bar for this epoch
        self.tqdm_cb.on_epoch_begin(self.state)

    def on_epoch_end(self):
        # Close tqdm bar
        self.tqdm_cb.on_epoch_end(self.state)
        # Possibly save distributed checkpoint at epoch boundary
        self.checkpointer_cb.on_epoch_end(self.state)
        # Possibly save HF checkpoint at epoch boundary
        self.hf_ckpt_cb.on_epoch_end(self.state)
        # Possibly run evaluation at epoch boundary
        self.evaluate_cb.on_epoch_end(self.state)

    def on_step_begin(self, micro_batches: List[Dict[str, Any]]):
        # Accumulate token / sample counts for MFU / throughput metrics
        self.environ_meter_cb.on_step_begin(self.state, micro_batches=micro_batches)

    def on_step_end(self, loss: float, loss_dict: Dict[str, float], grad_norm: float):
        # Compute + aggregate step metrics (MFU, throughput, loss, lr …)
        self.environ_meter_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        # Update tqdm postfix
        self.tqdm_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        # Log to W&B
        self.wandb_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        # Advance profiler step
        self.profile_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        # Possibly save distributed checkpoint
        self.checkpointer_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        # Possibly save HF checkpoint
        self.hf_ckpt_cb.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        # Possibly run evaluation
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
        """Single optimizer step (potentially multiple micro-batches)."""
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
