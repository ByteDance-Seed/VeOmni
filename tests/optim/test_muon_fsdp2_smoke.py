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

"""FSDP2 + ExtraParallel smoke tests for ``optimizer_type="muon"``."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from veomni.arguments.arguments_types import MixedPrecisionConfig
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim import build_optimizer
from veomni.optim.optimizer import MultiOptimizer
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)


SEED = 7777
EP_SIZE = 2
QWEN3_MOE_TOY_CFG = "tests/toy_config/qwen3_moe_toy"


def _distributed_smoke(use_zero_comm: bool) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_type = get_device_type()
    get_torch_device().set_device(f"{device_type}:{local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend())
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size == 4, f"this test expects exactly 4 ranks, got {world_size}"

    init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        extra_parallel_sizes=(EP_SIZE,),
        extra_parallel_placement_innermost=(False,),
        extra_parallel_names=("ep",),
        dp_mode="fsdp2",
    )

    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models import build_foundation_model

    # The eager Qwen3-MoE expert path is not EP-aware; use fused MoE here.
    ops_cfg = OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="fused_triton",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
    )
    # fused_triton MoE requires fp16/bf16 routing weights.
    model = build_foundation_model(
        config_path=QWEN3_MOE_TOY_CFG,
        weights_path=None,
        torch_dtype="bfloat16",
        init_device="meta",
        ops_implementation=ops_cfg,
    )
    model = build_parallelize_model(
        model,
        init_device="meta",
        weights_path=None,
        mixed_precision=MixedPrecisionConfig(enable=False),
        enable_gradient_checkpointing=False,
        basic_modules=[],
        enable_reentrant=False,
        enable_forward_prefetch=True,
        broadcast_model_weights_from_rank0=False,
        max_load_broadcast_size=int(1e9),
        muon_expert_zero_comm=use_zero_comm,
    )

    expected_shard_dim = 0 if use_zero_comm else 1
    sample_name = "model.layers.0.mlp.experts.gate_up_proj"
    p = dict(model.named_parameters()).get(sample_name)
    assert p is not None, f"expected param {sample_name} to exist on the toy Qwen3-MoE v5 model"
    assert isinstance(p, DTensor), f"{sample_name} should be a DTensor under FSDP+EP, got {type(p)}"
    shard_dims = [pl.dim for pl in p.placements if isinstance(pl, Shard)]
    assert shard_dims == [expected_shard_dim], (
        f"{sample_name}: expected Shard({expected_shard_dim}), got placements={p.placements}; "
        f"use_zero_comm={use_zero_comm}"
    )

    optimizer = build_optimizer(
        model,
        lr=1e-4,
        weight_decay=0.01,
        fused=True,
        optimizer_type="muon",
        no_decay_modules=[],
        no_decay_params=[],
        muon_kwargs={
            "lr": 2e-2,
            "weight_decay": 0.0,
            "momentum": 0.9,
            "nesterov": True,
            "ns_steps": 5,
            "ns_coefficients": (3.4445, -4.7750, 2.0315),
            "eps": 1e-7,
            "adjust_lr_fn": "match_rms_adamw",
        },
    )
    assert isinstance(optimizer, MultiOptimizer)

    assert "muon_ep" in optimizer.optimizers_dict, (
        f"expected muon_ep sub-optimizer for EP-resident expert weights; "
        f"got optimizer keys: {sorted(optimizer.optimizers_dict.keys())}"
    )

    if rank == 0:
        print(
            f"[rank0] MultiOptimizer keys (use_zero_comm={use_zero_comm}): {sorted(optimizer.optimizers_dict.keys())}"
        )

    device = torch.device(f"{device_type}:{local_rank}")

    snap = {
        name: (p.full_tensor() if isinstance(p, DTensor) else p).detach().cpu().clone()
        for name, p in model.named_parameters()
    }

    inputs = torch.randint(0, 1000, (2, 8), device=device)
    out = model(input_ids=inputs, labels=inputs)
    loss = out.loss
    assert loss is not None and torch.isfinite(loss), f"loss is None or not finite: {loss}"
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    moved = 0
    for name, p in model.named_parameters():
        new = (p.full_tensor() if isinstance(p, DTensor) else p).detach().cpu()
        if not torch.equal(new, snap[name]):
            moved += 1
    assert moved > 0, "no parameter changed after a single optimizer step"

    if rank == 0:
        print(
            f"[rank0] smoke OK (use_zero_comm={use_zero_comm}): "
            f"loss={loss.item():.4f}, moved {moved}/{len(snap)} params"
        )

    dist.barrier()
    dist.destroy_process_group()


def _has_devices(n: int) -> bool:
    try:
        return get_torch_device().device_count() >= n
    except Exception:
        return False


def _torchrun_cmd(nproc: int, port: int, use_zero_comm: bool) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        f"--master_port={port}",
        os.path.abspath(__file__),
        f"--zero-comm={int(use_zero_comm)}",
    ]


@pytest.mark.skipif(not _has_devices(4), reason="device_count should be >= 4")
def test_smoke_default_backend_4gpu():
    cmd = _torchrun_cmd(nproc=4, port=29711, use_zero_comm=False)
    env = os.environ.copy()
    env.setdefault("NCCL_DEBUG", "WARN")
    result = subprocess.run(cmd, env=env, check=True)
    assert result.returncode == 0


@pytest.mark.skipif(not _has_devices(4), reason="device_count should be >= 4")
def test_smoke_zero_comm_backend_4gpu():
    cmd = _torchrun_cmd(nproc=4, port=29712, use_zero_comm=True)
    env = os.environ.copy()
    env.setdefault("NCCL_DEBUG", "WARN")
    result = subprocess.run(cmd, env=env, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-comm", type=int, default=0)
    args = parser.parse_args()
    _distributed_smoke(use_zero_comm=bool(args.zero_comm))
