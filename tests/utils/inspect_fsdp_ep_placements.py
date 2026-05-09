# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Diagnostic script for inspecting DTensor placements under FSDP2+EP.

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard

from veomni.arguments.arguments_types import MixedPrecisionConfig
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)


class _ToyMoeExperts(nn.Module):
    """Stand-in for `gate_up_proj` / `down_proj`: a 3D tensor [N_experts, H, I]."""

    def __init__(self, n_experts: int = 8, hidden: int = 32, intermediate: int = 64):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.ones(n_experts, hidden, intermediate))
        self.down_proj = nn.Parameter(torch.ones(n_experts, intermediate, hidden))

    def forward(self) -> torch.Tensor:
        return self.gate_up_proj.sum() + self.down_proj.sum()


class _ToyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_qkv = nn.Linear(32, 96, bias=False)
        self.attn_out = nn.Linear(32, 32, bias=False)
        self.moe = _ToyMoeExperts()

    def forward(self, x):
        return self.attn_out(self.attn_qkv(x)).sum() + self.moe()


class _ToyMoEModel(nn.Module):
    _no_split_modules = ["_ToyDecoderLayer"]

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(128, 32)
        self.decoder = _ToyDecoderLayer()

    def forward(self, x):
        return self.embed_tokens(x).sum() + self.decoder(self.embed_tokens(x))

    def init_weights(self):
        for p in self.parameters():
            with torch.no_grad():
                p.fill_(1.0)

    def get_parallel_plan(self):
        ep_plan = {
            "decoder.moe.gate_up_proj": Shard(0),
            "decoder.moe.down_proj": Shard(0),
        }
        plan = ParallelPlan(extra_parallel_plan={"ep": ep_plan})
        plan.extra_parallel_fsdp_no_shard_module = {"ep": {"decoder.moe"}}
        return plan


def _describe(name: str, p: torch.nn.Parameter) -> str:
    if isinstance(p, DTensor):
        return (
            f"{name}: type=DTensor "
            f"global_shape={tuple(p.shape)} "
            f"placements={p.placements} "
            f"mesh={p.device_mesh} "
            f"local_shape={tuple(p._local_tensor.shape)}"
        )
    extra = ""
    if hasattr(p, "spec_info"):
        si = p.spec_info
        extra = f" spec_info(para={si.para_name!r}, placement={si.placement})"
    return f"{name}: type=Parameter shape={tuple(p.shape)}{extra}"


def main():
    dist.init_process_group(backend=get_dist_comm_backend())
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_type = get_device_type()
    get_torch_device().set_device(f"{device_type}:{local_rank}")

    world_size = dist.get_world_size()
    ep_size = int(os.environ.get("EP_SIZE", "2"))
    assert world_size % ep_size == 0, f"world_size {world_size} must be divisible by EP_SIZE={ep_size}"

    init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        extra_parallel_sizes=(ep_size,),
        extra_parallel_placement_innermost=(False,),
        extra_parallel_names=("ep",),
        dp_mode="fsdp2",
    )

    with torch.device("meta"):
        model = _ToyMoEModel()
    model = build_parallelize_model(
        model,
        init_device="meta",
        weights_path=None,
        enable_full_shard=True,
        mixed_precision=MixedPrecisionConfig(enable=False),
        enable_gradient_checkpointing=False,
        enable_fsdp_offload=False,
        basic_modules=[],
        enable_reentrant=False,
        enable_forward_prefetch=True,
        broadcast_model_weights_from_rank0=False,
        max_load_broadcast_size=int(1e9),
    )

    rank = dist.get_rank()
    if rank == 0:
        print(f"\n=== world_size={world_size}, ep_size={ep_size}, fsdp_per_ep={world_size // ep_size} ===\n")
        for name, p in model.named_parameters():
            print(_describe(name, p))
        print()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
