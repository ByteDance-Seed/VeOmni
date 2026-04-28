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
from typing import Any, Dict, List, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from ..arguments import MixedPrecisionConfig, VeOmniArguments
from ..data import build_chat_template, build_data_transform
from ..data.data_collator import PostCollator
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..distributed.parallel_state import get_parallel_state
from ..distributed.sequence_parallel import gather_outputs
from ..distributed.torch_parallelize import build_parallelize_model
from ..models import build_foundation_model, build_tokenizer
from ..ops.batch_invariant_ops import set_batch_invariant_mode
from ..utils import helper, logging
from ..utils.constants import IGNORE_INDEX
from ..utils.device import synchronize
from .base import BaseTrainer


logger = logging.get_logger(__name__)

_NON_MODEL_KEYS = {"labels"}


# ================================ DPO Arguments ======================================


@dataclass
class DPOConfig:
    """dpo.* — Direct Preference Optimization hyperparameters."""

    beta: float = field(
        default=0.1,
        metadata={"help": "Temperature parameter for the DPO loss. Controls deviation from the reference model."},
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "Label smoothing for DPO loss. Non-zero values assume noisy preference labels."},
    )
    reference_free: bool = field(
        default=False,
        metadata={"help": "If True, ignore the reference model and use an implicit uniform reference."},
    )
    loss_type: Literal["sigmoid", "ipo"] = field(
        default="sigmoid",
        metadata={"help": "DPO loss variant: 'sigmoid' for standard DPO, 'ipo' for Identity Preference Optimization."},
    )
    average_log_prob: bool = field(
        default=False,
        metadata={"help": "If True, average log probs per token instead of summing."},
    )
    refer_model_precision: Literal["float32", "bfloat16"] = field(
        default="bfloat16",
        metadata={"help": "Precision of the reference model."},
    )


@dataclass
class VeOmniDPOArguments(VeOmniArguments):
    """Root config for DPO training — extends VeOmniArguments with DPO hyperparameters."""

    dpo_config: DPOConfig = field(default_factory=DPOConfig)


class TextDPOTrainer:
    """Text DPO trainer that composes BaseTrainer with DPO-specific logic."""

    base: BaseTrainer
    reference_model: PreTrainedModel

    def __init__(self, args: VeOmniDPOArguments):
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base._setup()
        self.base._build_model()
        self.base._freeze_model_module()

        self._build_model_assets()
        self._build_data_transform()

        self.base._build_dataset()
        self.base._build_collate_fn()
        self.base._build_dataloader()
        self._build_postforward()
        self.base._build_parallelized_model()
        self.base._build_optimizer()
        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()

        self._build_reference_model()

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

    def _build_postforward(self):
        self.post_forward = PostCollator()
        self.sp_enabled = get_parallel_state().sp_enabled

    def _build_reference_model(self):
        """Build and freeze a reference model with the same architecture and FSDP sharding."""
        args: VeOmniDPOArguments = self.base.args
        logger.info_rank0("Building frozen reference model for DPO")

        self.reference_model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=args.model.model_path,
            torch_dtype=args.dpo_config.refer_model_precision,
            attn_implementation=args.model.ops_implementation.attn_implementation,
            init_device=args.train.init_device,
        )

        self.reference_model.requires_grad_(False)

        self.reference_model = build_parallelize_model(
            self.reference_model,
            init_device=args.train.init_device,
            weights_path=args.model.model_path,
            enable_full_shard=args.train.accelerator.fsdp_config.full_shard,
            enable_reshard_after_forward=args.train.accelerator.fsdp_config.reshard_after_forward,
            mixed_precision=MixedPrecisionConfig(enable=False),  # In reference model, we will not use mixed precision
            enable_gradient_checkpointing=False,
            enable_fsdp_offload=args.train.accelerator.fsdp_config.offload,
            basic_modules=list(
                set(getattr(self.reference_model, "_no_split_modules", None) or []) | set(args.model.basic_modules)
            ),
            enable_reentrant=False,
            enable_forward_prefetch=args.train.accelerator.fsdp_config.forward_prefetch,
        )
        self.reference_model.eval()
        helper.print_device_mem_info("VRAM usage after building reference model")

    @staticmethod
    def dpo_loss(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        beta: float,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        reference_free: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the DPO/IPO loss for a batch of policy and reference model log probabilities.

        Returns:
            (losses, chosen_rewards, rejected_rewards) -- each of shape (batch_size,).
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if loss_type == "ipo":
            losses = (logits - 1 / (2 * beta)) ** 2
        else:
            losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
            )

        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(self, model: nn.Module, micro_batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single forward pass on the packed batch containing chosen+rejected pairs.

        Each DPO sample contributes two consecutive sequences (chosen then
        rejected) to the packed sequence.  ``PostCollator`` gathers SP outputs and
        splits logits per-sequence; even-indexed sequences are chosen and
        odd-indexed are rejected.

        Returns:
            (chosen_logps, rejected_logps) each of shape ``(B,)``.
        """
        model_inputs = {k: v for k, v in micro_batch.items() if k not in _NON_MODEL_KEYS}
        outputs = model(**model_inputs, use_cache=False)

        # PackingPostCollator.split expects 2-D logits [packed_L, V];
        # the model returns [1, packed_L, V] so squeeze the batch dim.
        outputs.logits = outputs.logits.squeeze(0)
        outputs = self.post_forward(outputs, micro_batch)
        logits_list = outputs.logits  # list of 2*B tensors, each [L_i, V]
        seq_lens = [lg.shape[0] for lg in logits_list]

        if self.sp_enabled:
            # Labels already globally shifted by SequenceParallelCollator;
            # gather back and split by sequence lengths directly.
            all_labels = gather_outputs(micro_batch["labels"], gather_dim=-1, group=get_parallel_state().sp_group)
            all_labels = all_labels.view(-1)[: sum(seq_lens)]
            labels_list = list(all_labels.split(seq_lens))
        else:
            # Shift labels per-sequence
            all_labels = micro_batch["labels"].view(-1)
            offset = 0
            labels_list = []
            for sl in seq_lens:
                seq_labels = all_labels[offset : offset + sl]
                labels_list.append(F.pad(seq_labels[1:], (0, 1), value=IGNORE_INDEX))
                offset += sl

        average_log_prob = getattr(self.base.args, "dpo_config", None) and self.base.args.dpo_config.average_log_prob
        all_logps: List[torch.Tensor] = []
        for seq_logits, seq_labels in zip(logits_list, labels_list):
            loss_mask = seq_labels != IGNORE_INDEX
            safe_labels = seq_labels.clamp(min=0)
            per_token_logps = torch.gather(
                seq_logits.float().log_softmax(-1), dim=1, index=safe_labels.unsqueeze(1)
            ).squeeze(1)
            if average_log_prob:
                logp = (per_token_logps * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            else:
                logp = (per_token_logps * loss_mask).sum()
            all_logps.append(logp)

        all_logps_t = torch.stack(all_logps)
        return all_logps_t[0::2], all_logps_t[1::2]

    def forward_backward_step(
        self, micro_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        args: VeOmniDPOArguments = self.base.args
        dpo_config = args.dpo_config

        micro_batch = self.base.preforward(micro_batch)

        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps = self.concatenated_forward(self.reference_model, micro_batch)

        with self.base.model_fwd_context, set_batch_invariant_mode(args.train.enable_batch_invariant_mode):
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.base.model, micro_batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=dpo_config.beta,
            label_smoothing=dpo_config.label_smoothing,
            loss_type=dpo_config.loss_type,
            reference_free=dpo_config.reference_free,
        )

        loss = losses.mean()

        reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
        loss_dict: Dict[str, torch.Tensor] = {
            "dpo_loss": loss.detach(),
            "chosen_rewards": chosen_rewards.mean().detach(),
            "rejected_rewards": rejected_rewards.mean().detach(),
            "reward_accuracy": reward_accuracies.detach(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().detach(),
        }

        with self.base.model_bwd_context, set_batch_invariant_mode(args.train.enable_batch_invariant_mode):
            loss.backward()

        del micro_batch
        return loss, loss_dict

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
            loss, loss_dict = self.forward_backward_step(micro_batch)

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
