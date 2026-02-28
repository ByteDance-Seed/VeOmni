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
Base RL Trainer class for distributed RL.

The main difference between BaseRLTrainer and BaseTrainer is that
BaseTrainer: pack and sp slice data in dataloader worker
RLTrainer:
    1. pack and sp slice data in training loop before forward
    2. postforward to gather outputs to get sample-wise logits
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput

from ..data.data_collator import MainCollator as Preforward
from ..data.data_collator import PostCollator as Postforward
from ..distributed.parallel_state import get_parallel_state
from ..distributed.sequence_parallel import gather_outputs
from .base import BaseTrainer, VeOmniArguments, build_dataloader


class BaseRLTrainer(BaseTrainer):
    def __init__(self, args: VeOmniArguments):
        super().__init__(args)
        self._build_preforward_postforward()

    # post init preforward and postforward hooks
    def _build_preforward_postforward(self):
        """Build preforward and postforward hooks."""
        self.pre_forward = Preforward()
        self.post_forward = Postforward()

    # rewrite: do not build collate_fn in dataloader, as we pack and sp slice data in training loop in preforward
    def _build_dataloader(self):
        """Do not build collate_fn for RL trainer."""
        args: VeOmniArguments = self.args
        self.train_dataloader = build_dataloader(
            dataloader_type=args.data.dataloader_type,
            dataset=self.train_dataset,
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
            build_collate_fn=False,
        )

    def preforward(self, micro_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        micro_batch = self.pre_forward(micro_batch)
        return super().preforward(micro_batch)

    def postforward(self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]) -> None:
        """Postprocess model outputs after forward pass."""
        outputs = self.post_forward(outputs, micro_batch)

        logits = outputs.logits
        labels = micro_batch["labels"]

        logits = torch.cat(logits, dim=0)

        if get_parallel_state().sp_enabled:
            labels = gather_outputs(labels, gather_dim=-1, group=get_parallel_state().sp_group)
            labels = labels[:, : logits.shape[0]]  # unpad sp_pad
        else:
            labels = nn.functional.pad(labels, (0, 1), value=-100)
            labels = labels[..., 1:].contiguous()

        logits = logits.float()
        shift_labels = labels.view(-1)

        loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="mean")

        outputs.loss = loss
        outputs.logits = logits  # logits_list
        return super().postforward(outputs, micro_batch)
