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
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from ..arguments import VeOmniDPOArguments
from ..data import build_chat_template, build_data_transform
from ..data.data_collator import DataCollator
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..models import build_tokenizer
from ..utils import helper
from ..utils.constants import IGNORE_INDEX
from ..utils.device import synchronize
from .base_dpo_trainer import BaseDPOTrainer


logger = helper.create_logger(__name__)

_DPO_PAD_VALUES = {
    "input_ids": 0,
    "attention_mask": 0,
    "labels": IGNORE_INDEX,
}


@dataclass
class DPOCollator(DataCollator):
    """Collator for DPO preference data.

    Each sample from the dataset has shape ``[2, L]`` (row 0 = chosen, row 1 =
    rejected), with standard keys ``input_ids``, ``attention_mask``, ``labels``.

    This collator pads a list of B such samples to a common length and
    concatenates them along the batch dimension, producing tensors of shape
    ``[2*B, max_L]``.  The first B rows are chosen, the last B rows are rejected,
    matching the layout expected by ``BaseDPOTrainer.concatenated_forward``.
    """

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {}
        for key in features[0].keys():
            tensors = [f[key] for f in features]  # each [2, L_i]
            pad_value = _DPO_PAD_VALUES.get(key, 0)
            max_len = max(t.shape[-1] for t in tensors)
            padded = [F.pad(t, (0, max_len - t.shape[-1]), value=pad_value) for t in tensors]
            # cat along dim=0: [2, L], [2, L], ... → [2*B, max_L]
            batch[key] = torch.cat(padded, dim=0)
        return batch


class TextDPOTrainer:
    """Text DPO trainer that composes BaseDPOTrainer with TextTrainer-style init."""

    base: BaseDPOTrainer

    def __init__(self, args: VeOmniDPOArguments):
        self.base = BaseDPOTrainer.__new__(BaseDPOTrainer)
        self.base.args = args

        self.base._setup()
        self.base._build_model()
        self.base._freeze_model_module()

        self._build_model_assets()
        self._build_data_transform()

        self.base._build_dataset()
        self._build_collate_fn()
        self.base._build_dataloader()
        self.base._build_parallelized_model()
        self.base._build_optimizer()
        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()

        self.base._build_reference_model()

    def _build_model_assets(self):
        args: VeOmniDPOArguments = self.base.args
        model_config = self.base.model_config
        self.base.tokenizer = build_tokenizer(args.model.tokenizer_path)
        self.base.chat_template = build_chat_template(args.data.chat_template, self.base.tokenizer)
        self.base.model_assets = [model_config, self.base.chat_template]

    def _build_data_transform(self):
        args: VeOmniDPOArguments = self.base.args
        self.base.data_transform = build_data_transform(
            "dpo",
            tokenizer=self.base.tokenizer,
            chat_template=self.base.chat_template,
            max_seq_len=args.data.max_seq_len,
        )

    def _build_collate_fn(self):
        self.base.collate_fn = DPOCollator()

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

    def train_step(self, data_iterator: Any) -> Dict[str, float]:
        args: VeOmniDPOArguments = self.base.args
        self.base.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)

        self.on_step_begin(micro_batches=micro_batches)

        synchronize()

        total_loss = 0.0
        total_loss_dict: Dict[str, float] = defaultdict(float)

        num_micro_steps = len(micro_batches)
        for micro_step, micro_batch in enumerate(micro_batches):
            self.base.model_reshard(micro_step, num_micro_steps)
            loss, loss_dict = self.base.forward_backward_step(micro_batch)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item()

        grad_norm = veomni_clip_grad_norm(self.base.model, args.train.optimizer.max_grad_norm)

        self.base.optimizer.step()
        self.base.lr_scheduler.step()
        self.base.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

    def train(self):
        args: VeOmniDPOArguments = self.base.args
        self.on_train_begin()
        logger.info(
            f"Rank{args.train.local_rank} Start DPO training. "
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

            data_iterator = iter(self.base.train_dataloader)

            for _ in range(self.base.start_step, args.train_steps):
                try:
                    self.train_step(data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.on_epoch_end()

            self.base.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        self.on_train_end()

        synchronize()

        self.base.destroy_distributed()
