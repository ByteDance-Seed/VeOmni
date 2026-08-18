"""MiniMax H3 training glue: DiTTrainer subclass + fp32-master optimizer.

The base DiTTrainer hardcodes "dit_offline" / "dit_online" data transforms
and builds the VeOmni optimizer.
- data transform comes from YAML (minimax_h3_offline / minimax_h3_online)
- optimizer: fp32 master params with ZeRO-3-style sharding across dp ranks
- lr schedule: EmulatedConstantLR
"""

from __future__ import annotations

import torch

from veomni.data.data_transform import build_data_transform
from veomni.trainer.dit_trainer import DiTTrainer, VeOmniDiTArguments


# fp32-master optimizer (ZeRO-3 semantics): bf16 model params, fp32 master
# params, torch.optim.AdamW (plain, fused=False) on the masters — fp32 state,
# fp32 updates. VeOmni DDP mode runs AdamW directly on bf16 params. This
# wrapper mirrors that: forward/grads stay on bf16 params; the step happens on
# fp32 masters. Masters + AdamW state are sharded across dp ranks (ZeRO-3
# partition) so they fit in NPU memory; after the step each rank broadcasts
# its updated shard so every rank's full bf16 replica is updated.
class FP32MasterOptimizer(torch.optim.AdamW):
    def __init__(self, model, lr, weight_decay):
        from veomni.distributed.parallel_state import get_parallel_state

        self.params = [p for p in model.parameters() if p.requires_grad]
        ps = get_parallel_state()
        self.rank = ps.dp_rank
        self.world_size = ps.dp_size
        self.local_idx = list(range(self.rank, len(self.params), self.world_size))
        local_masters = [self.params[i].detach().float() for i in self.local_idx]
        super().__init__(local_masters, lr=lr, weight_decay=weight_decay)

    def step(self, closure=None):
        from torch import distributed as dist

        from veomni.distributed.parallel_state import get_parallel_state

        masters = self.param_groups[0]["params"]
        for i, m in zip(self.local_idx, masters):
            p = self.params[i]
            if p.grad is not None:
                m.grad = p.grad.float()
        super().step(closure)
        # Free all grads before gathering buffers (peak memory).
        self.zero_grad(set_to_none=True)
        group = get_parallel_state().dp_group
        local_flat = torch.cat([m.detach().reshape(-1) for m in masters])
        # Shard total sizes can differ across ranks (params unevenly sized);
        # allocate the receiver buffer from src's exact shard length instead.
        shard_lens = [
            sum(self.params[i].numel() for i in range(src, len(self.params), self.world_size))
            for src in range(self.world_size)
        ]
        for src in range(self.world_size):
            buf = (
                local_flat
                if src == self.rank
                else torch.empty(shard_lens[src], dtype=torch.float32, device=local_flat.device)
            )
            dist.broadcast(buf, group_src=src, group=group)
            offset = 0
            for i in range(src, len(self.params), self.world_size):
                p = self.params[i]
                numel = p.numel()
                p.data.copy_(buf[offset : offset + numel].view_as(p).to(p.dtype))
                offset += numel

    def zero_grad(self, set_to_none: bool = True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()
        for m in self.param_groups[0]["params"]:
            m.grad = None


class EmulatedConstantLR(torch.optim.lr_scheduler.LRScheduler):
    """ConstantLR stepped to full lr after 2 scheduler steps.

    Plain ConstantLR(total_iters=5) stays at 1/3 lr until step 6 and diverges at
    step 4.
    """

    def get_lr(self):
        factor = 1 / 3 if self.last_epoch < 2 else 1.0
        return [base * factor for base in self.base_lrs]


class MinimaxH3DiTTrainer(DiTTrainer):
    """DiTTrainer subclass for MiniMax H3 FL2VA training.

    Adds on top of the base trainer:
    - YAML-driven data transform selection (base hardcodes dit_offline/dit_online)
    - fp32-master optimizer + emulated ConstantLR schedule
    """

    def __init__(self, args: VeOmniDiTArguments):
        super().__init__(args)
        # Raw video frames (list of 124 float32 tensors) would blow up print_example.
        self.base.LOG_SAMPLE = False
        # The fp32-master optimizer assumes every rank holds the full bf16 replica
        # (ddp mode). Under fsdp2 the model is already sharded across dp ranks and
        # VeOmni's native optimizer handles the DTensor shards directly — replacing
        # it would shard the shards.
        if args.train.training_task != "offline_embedding" and args.train.accelerator.fsdp_config.fsdp_mode != "fsdp2":
            # Replace VeOmni optimizer with fp32-master AdamW, and replicate
            # the lr schedule exactly.
            self.base.optimizer = FP32MasterOptimizer(
                self.base.model,
                lr=args.train.optimizer.lr,
                weight_decay=args.train.optimizer.weight_decay,
            )
            self.base.lr_scheduler = EmulatedConstantLR(self.base.optimizer)

    def _build_data_transform(self):
        args: VeOmniDiTArguments = self.base.args
        if args.data.datasets_type == "minimax_h3_offline":
            self.base.data_transform = build_data_transform("minimax_h3_offline")
        elif args.data.datasets_type == "minimax_h3_online":
            self.base.data_transform = build_data_transform("minimax_h3_online", **args.data.mm_configs)
        else:
            super()._build_data_transform()
