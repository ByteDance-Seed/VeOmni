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
Base DPO Trainer class for Direct Preference Optimization.

Extends BaseTrainer with:
    1. A frozen reference model for computing reference log probabilities
    2. Concatenated forward pass (chosen + rejected in one batch)
    3. DPO preference loss computation
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from ..arguments import VeOmniDPOArguments
from ..distributed.torch_parallelize import build_parallelize_model
from ..models import build_foundation_model
from ..ops.batch_invariant_ops import set_batch_invariant_mode
from ..utils import helper, logging
from ..utils.constants import IGNORE_INDEX
from .base import BaseTrainer


logger = logging.get_logger(__name__)


class BaseDPOTrainer(BaseTrainer):
    """Base trainer for DPO that handles reference model management and DPO loss."""

    reference_model: PreTrainedModel

    def __init__(self, args: VeOmniDPOArguments):
        super().__init__(args)

    def _build_reference_model(self):
        """Build and freeze a reference model with the same architecture and FSDP sharding."""
        args: VeOmniDPOArguments = self.args
        logger.info_rank0("Building frozen reference model for DPO")

        self.reference_model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=args.model.model_path,
            torch_dtype=args.dpo_config.refer_model_precision,
            attn_implementation=args.model.ops_implementation.attn_implementation,
            moe_implementation=args.model.ops_implementation.moe_implementation,
            init_device=args.train.init_device,
        )

        self.reference_model.requires_grad_(False)

        self.reference_model = build_parallelize_model(
            self.reference_model,
            init_device=args.train.init_device,
            weights_path=args.model.model_path,
            enable_full_shard=args.train.accelerator.fsdp_config.full_shard,
            enable_reshard_after_forward=args.train.accelerator.fsdp_config.reshard_after_forward,
            enable_mixed_precision=args.train.enable_mixed_precision,
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
            (losses, chosen_rewards, rejected_rewards) — each of shape (batch_size,).
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

    @staticmethod
    def get_batch_logps(
        logits: torch.Tensor,
        labels: torch.Tensor,
        average_log_prob: bool = False,
    ) -> torch.Tensor:
        """Compute per-sample log probabilities from model logits and labels.

        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len) with IGNORE_INDEX for masked positions
            average_log_prob: if True, return mean log-prob per valid token; else sum.

        Returns:
            (batch_size,) log probabilities
        """
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != IGNORE_INDEX

        labels[labels == IGNORE_INDEX] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
        return (per_token_logps * loss_mask).sum(-1)

    def _concatenate_inputs(self, micro_batch: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], int]:
        """Concatenate chosen and rejected inputs along batch dim for a single forward pass.

        Returns:
            (concatenated_batch, num_chosen) where num_chosen is the batch size of chosen samples.
        """
        chosen_ids = micro_batch["chosen_input_ids"]
        rejected_ids = micro_batch["rejected_input_ids"]

        if chosen_ids.dim() == 1:
            chosen_ids = chosen_ids.unsqueeze(0)
            rejected_ids = rejected_ids.unsqueeze(0)
            micro_batch["chosen_attention_mask"] = micro_batch["chosen_attention_mask"].unsqueeze(0)
            micro_batch["rejected_attention_mask"] = micro_batch["rejected_attention_mask"].unsqueeze(0)
            micro_batch["chosen_labels"] = micro_batch["chosen_labels"].unsqueeze(0)
            micro_batch["rejected_labels"] = micro_batch["rejected_labels"].unsqueeze(0)

        num_chosen = chosen_ids.shape[0]
        max_len = max(chosen_ids.shape[1], rejected_ids.shape[1])

        def _pad(tensor: torch.Tensor, length: int, value: int) -> torch.Tensor:
            if tensor.shape[1] >= length:
                return tensor
            pad_size = length - tensor.shape[1]
            return F.pad(tensor, (0, pad_size), value=value)

        cat_input_ids = torch.cat([_pad(chosen_ids, max_len, 0), _pad(rejected_ids, max_len, 0)], dim=0)
        cat_attention_mask = torch.cat(
            [
                _pad(micro_batch["chosen_attention_mask"], max_len, 0),
                _pad(micro_batch["rejected_attention_mask"], max_len, 0),
            ],
            dim=0,
        )
        cat_labels = torch.cat(
            [
                _pad(micro_batch["chosen_labels"], max_len, IGNORE_INDEX),
                _pad(micro_batch["rejected_labels"], max_len, IGNORE_INDEX),
            ],
            dim=0,
        )

        concatenated = {
            "input_ids": cat_input_ids,
            "attention_mask": cat_attention_mask,
            "labels": cat_labels,
        }
        return concatenated, num_chosen

    def concatenated_forward(self, model: nn.Module, micro_batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single forward pass on concatenated chosen+rejected inputs.

        Returns:
            (chosen_logps, rejected_logps) each of shape (batch_size,).
        """
        concatenated, num_chosen = self._concatenate_inputs(micro_batch)
        concatenated = {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in concatenated.items()
        }

        outputs = model(
            input_ids=concatenated["input_ids"],
            attention_mask=concatenated["attention_mask"],
            use_cache=False,
        )
        all_logits = outputs.logits.float()

        average_log_prob = getattr(self.args, "dpo_config", None) and self.args.dpo_config.average_log_prob
        all_logps = self.get_batch_logps(all_logits, concatenated["labels"], average_log_prob=average_log_prob)

        chosen_logps = all_logps[:num_chosen]
        rejected_logps = all_logps[num_chosen:]
        return chosen_logps, rejected_logps

    def forward_backward_step(
        self, micro_batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        args: VeOmniDPOArguments = self.args
        dpo_config = args.dpo_config

        micro_batch = self.preforward(micro_batch)

        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps = self.concatenated_forward(self.reference_model, micro_batch)

        with self.model_fwd_context, set_batch_invariant_mode(args.train.enable_batch_invariant_mode):
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.model, micro_batch)

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

        with self.model_bwd_context, set_batch_invariant_mode(args.train.enable_batch_invariant_mode):
            loss.backward()

        del micro_batch
        return loss, loss_dict
