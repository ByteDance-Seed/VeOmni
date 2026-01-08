from typing import Dict

import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput

from ..distributed.parallel_state import get_parallel_state
from ..distributed.sequence_parallel import gather_outputs
from .postforward import Postforward
from .preforward import Preforward
from .vlm_trainer import Arguments, VLMTrainer


class VLMRLTrainer(VLMTrainer):
    def _build_preforward_postforward(self):
        """Build preforward and postforward hooks."""
        args: Arguments = self.args
        self.pre_forward = Preforward(
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            attn_implementation=args.model.attn_implementation,
        )
        self.post_forward = Postforward(
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        )

    def postforward(self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]) -> None:
        """Postprocess model outputs after forward pass."""
        args: Arguments = self.args
        outputs = self.post_forward(outputs, micro_batch)

        logits = outputs.logits
        labels = micro_batch["labels"]

        if args.train.rmpad_with_pos_ids:
            logits = torch.cat(logits, dim=0)

            if get_parallel_state().sp_enabled:
                labels = gather_outputs(labels, gather_dim=-1, group=get_parallel_state().sp_group)
                labels = labels[:, : logits.shape[0]]  # unpad sp_pad
            else:
                labels = nn.functional.pad(labels, (0, 1), value=-100)
                labels = labels[..., 1:].contiguous()
        else:
            if get_parallel_state().sp_enabled:
                labels = gather_outputs(labels, gather_dim=-1, group=get_parallel_state().sp_group)
            else:
                labels = nn.functional.pad(labels, (0, 1), value=-100)
                labels = labels[..., 1:].contiguous()
            labels = [label[: len(logit)] for label, logit in zip(labels, logits)]  # unpad padding & sp_pad
            labels = torch.cat(labels, dim=0)

            logits = torch.cat(logits, dim=0)

        logits = logits.float()
        shift_labels = labels.view(-1)

        torch.distributed.barrier()
        loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="mean")

        outputs.loss = loss
        outputs.logits = logits
        return super().postforward(outputs, micro_batch)
