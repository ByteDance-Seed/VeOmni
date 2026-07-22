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

"""FSDP2 and FSDP2+EP parity tests for ``DistributedMuon``."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard

from veomni.distributed.parallel_state import init_parallel_state
from veomni.optim.muon import DistributedMuon
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)


SEED = 4242
EP_SIZE = 2  # world_size = 4 -> ep_fsdp_size = 4 // 2 = 2.
QWEN3_MOE_TOY_CFG = "tests/toy_config/qwen3_moe_toy"


def _stable_hash(s: str) -> int:
    """Process-independent 31-bit hash for per-FQN gradient seeds."""
    digest = hashlib.md5(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _full_grad(fqn: str, full_shape, step: int) -> torch.Tensor:
    """Deterministic full-shape gradient for a single parameter."""
    seed = (SEED + step + _stable_hash(fqn)) & 0x7FFFFFFF
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(full_shape, generator=g, dtype=torch.float32)


def _ep_local_slice(full: torch.Tensor, ep_factor: int, ep_rank: int) -> torch.Tensor:
    """Return this EP rank's dim-0 chunk."""
    chunk = full.shape[0] // ep_factor
    return full[ep_rank * chunk : (ep_rank + 1) * chunk]


def _make_grads(
    model: nn.Module,
    full_shapes: dict,
    step: int,
    device: torch.device,
    ep_rank: int = 0,
    ep_size: int = 1,
) -> None:
    """Attach deterministic gradients matching each param's local sharding."""
    for fqn, p in model.named_parameters():
        if not p.requires_grad or fqn not in full_shapes:
            continue
        full = _full_grad(fqn, full_shapes[fqn], step)
        if isinstance(p, DTensor):
            ep_factor = full.shape[0] // p.shape[0]
            local_full = _ep_local_slice(full, ep_factor, ep_rank) if ep_factor > 1 else full
            full_dt = DTensor.from_local(
                local_full.to(device),
                device_mesh=p.device_mesh,
                placements=[Replicate()] * p.device_mesh.ndim,
                run_check=False,
            )
            p.grad = full_dt.redistribute(device_mesh=p.device_mesh, placements=p.placements)
        else:
            p.grad = full.to(device)


def _full_state_dict(model: nn.Module, ep_group=None) -> dict:
    """Return a ``name -> globally full tensor`` mapping."""
    out: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if isinstance(p, DTensor):
            local_full = p.full_tensor().detach()
            if ep_group is not None and p.ndim == 3:
                gathered = [torch.empty_like(local_full) for _ in range(dist.get_world_size(group=ep_group))]
                dist.all_gather(gathered, local_full.contiguous(), group=ep_group)
                local_full = torch.cat(gathered, dim=0)
            out[name] = local_full.cpu()
        else:
            out[name] = p.detach().cpu()
    return out


def _seed_all(seed: int, device: torch.device) -> None:
    """Re-seed CPU and the active accelerator generator before init."""
    torch.manual_seed(seed)
    device_backend = get_torch_device()
    if device.type == get_device_type() and hasattr(device_backend, "manual_seed_all"):
        device_backend.manual_seed_all(seed)


def _force_reinit(model: nn.Module, seed: int, device: torch.device) -> None:
    """Re-seed RNG and force HF's ``init_weights`` to re-initialize."""
    for module in model.modules():
        if hasattr(module, "_is_hf_initialized"):
            module._is_hf_initialized = False
    _seed_all(seed, device)
    model.init_weights()


class _ToyDenseBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.gelu(self.up(x)))


class _ToyDenseModel(nn.Module):
    """Two stacked Linear blocks; every weight is 2D and Muon-eligible."""

    _no_split_modules = ["_ToyDenseBlock"]

    def __init__(self, hidden: int = 32, intermediate: int = 64):
        super().__init__()
        self.block0 = _ToyDenseBlock(hidden, intermediate)
        self.block1 = _ToyDenseBlock(hidden, intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block1(self.block0(x)).sum()


def _build_dense_model(device: torch.device) -> nn.Module:
    torch.manual_seed(SEED)
    return _ToyDenseModel().to(device)


def _dense_golden_state(full_shapes: dict) -> dict:
    """Single-process reference for the dense FSDP2 path."""
    device = torch.device("cpu")
    model = _build_dense_model(device)
    opt = DistributedMuon(
        list(model.parameters()),
        lr=5e-3,
        weight_decay=0.0,
        momentum=0.9,
        nesterov=True,
        adjust_lr_fn="match_rms_adamw",
    )
    for step in range(2):
        _make_grads(model, full_shapes, step, device)
        opt.step()
        opt.zero_grad()
    return _full_state_dict(model)


def _run_dense() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_type = get_device_type()
    get_torch_device().set_device(f"{device_type}:{local_rank}")
    dist.init_process_group(backend=get_dist_comm_backend())
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"{device_type}:{local_rank}")

    model = _build_dense_model(device)
    full_shapes = {fqn: tuple(p.shape) for fqn, p in model.named_parameters() if p.requires_grad}
    fully_shard(model.block0)
    fully_shard(model.block1)
    fully_shard(model)
    for name, p in model.named_parameters():
        assert isinstance(p, DTensor), f"expected DTensor for {name}, got {type(p)}"

    opt = DistributedMuon(
        list(model.parameters()),
        lr=5e-3,
        weight_decay=0.0,
        momentum=0.9,
        nesterov=True,
        adjust_lr_fn="match_rms_adamw",
    )
    for step in range(2):
        _make_grads(model, full_shapes, step, device)
        opt.step()
        opt.zero_grad()

    fsdp_state = _full_state_dict(model)
    if rank == 0:
        golden = _dense_golden_state(full_shapes)
        assert set(fsdp_state.keys()) == set(golden.keys())
        for k, v in golden.items():
            torch.testing.assert_close(
                fsdp_state[k],
                v,
                atol=1e-4,
                rtol=1e-4,
                msg=f"FSDP2 Muon update for {k!r} diverges from single-device Muon (world_size={world_size}).",
            )
        print(f"[rank0] dense FSDP2 / single-device parity OK across {len(golden)} param(s)")

    dist.barrier()
    dist.destroy_process_group()


def _eager_ops_config():
    """All-eager ``OpsImplementationConfig``."""
    from veomni.arguments.arguments_types import OpsImplementationConfig

    return OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
    )


def _build_qwen3_moe(device: torch.device) -> nn.Module:
    """Build the real Qwen3-MoE toy via ``build_foundation_model``."""
    from veomni.models import build_foundation_model

    return build_foundation_model(
        config_path=QWEN3_MOE_TOY_CFG,
        weights_path=None,
        torch_dtype="float32",
        init_device=device.type,
        ops_implementation=_eager_ops_config(),
    )


def _qwen3_moe_golden_state(device: torch.device, full_shapes: dict) -> dict:
    """Single-process golden Muon step on the full Qwen3-MoE toy."""
    model = _build_qwen3_moe(device)
    _force_reinit(model, SEED, device)

    from veomni.optim.muon import split_muon_adamw_params

    muon_params, _, muon_names, _ = split_muon_adamw_params(model)
    opt = DistributedMuon(
        muon_params,
        lr=5e-3,
        weight_decay=0.0,
        momentum=0.9,
        nesterov=True,
        adjust_lr_fn="match_rms_adamw",
    )
    name_to_param = dict(zip(muon_names, muon_params))
    for fqn, p in name_to_param.items():
        full = _full_grad(fqn, full_shapes[fqn], step=0)
        p.grad = full.to(device)
    opt.step()
    opt.zero_grad()

    out: dict[str, torch.Tensor] = {}
    name_set = set(muon_names)
    for name, p in model.named_parameters():
        if name in name_set:
            out[name] = p.detach().cpu()
    return out


def _run_qwen3_moe(use_zero_comm: bool) -> None:
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
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    ep_fsdp_mesh = ps.extra_parallel_fsdp_device_mesh["ep"]["ep_fsdp"]
    ep_mesh = ps.extra_parallel_fsdp_device_mesh["ep"]["ep"]
    fsdp_mesh = ps.fsdp_mesh

    device = torch.device(f"{device_type}:{local_rank}")

    model = _build_qwen3_moe(device)
    _force_reinit(model, SEED, device)

    full_shapes = {fqn: tuple(p.shape) for fqn, p in model.named_parameters() if p.requires_grad}

    plan = model.get_parallel_plan()
    plan.apply(model, {"ep": ps.extra_parallel_fsdp_device_mesh["ep"]})

    shard_dim = 0 if use_zero_comm else 1
    for layer in model.model.layers:
        fully_shard(
            layer.mlp.experts,
            mesh=ep_fsdp_mesh,
            shard_placement_fn=lambda _p, _d=shard_dim: Shard(_d),
        )
        fully_shard(layer, mesh=fsdp_mesh)
    fully_shard(model, mesh=fsdp_mesh)

    sample_name = "model.layers.0.mlp.experts.gate_up_proj"
    p = dict(model.named_parameters()).get(sample_name)
    if p is not None:  # tolerate v4 vs v5 naming differences
        assert isinstance(p, DTensor), f"{sample_name} should be a DTensor under FSDP+EP"
        shard_dims = [pl.dim for pl in p.placements if isinstance(pl, Shard)]
        assert shard_dims == [shard_dim], (
            f"{sample_name}: expected Shard({shard_dim}), got placements={p.placements}; use_zero_comm={use_zero_comm}"
        )

    from veomni.optim.muon import split_muon_adamw_params

    muon_params, _, muon_names, _ = split_muon_adamw_params(model)
    opt = DistributedMuon(
        muon_params,
        lr=5e-3,
        weight_decay=0.0,
        momentum=0.9,
        nesterov=True,
        adjust_lr_fn="match_rms_adamw",
    )
    name_to_param = dict(zip(muon_names, muon_params))
    ep_rank = ps.extra_parallel_rank("ep")
    for fqn, p in name_to_param.items():
        full = _full_grad(fqn, full_shapes[fqn], step=0)
        if isinstance(p, DTensor):
            ep_factor = full.shape[0] // p.shape[0]
            local_full = _ep_local_slice(full, ep_factor, ep_rank) if ep_factor > 1 else full
            full_dt = DTensor.from_local(
                local_full.to(device),
                device_mesh=p.device_mesh,
                placements=[Replicate()] * p.device_mesh.ndim,
                run_check=False,
            )
            p.grad = full_dt.redistribute(device_mesh=p.device_mesh, placements=p.placements)
        else:
            p.grad = full.to(device)
    opt.step()
    opt.zero_grad()

    ep_group = ep_mesh.get_group()
    fsdp_state: dict[str, torch.Tensor] = {}
    name_set = set(muon_names)
    for name, p in model.named_parameters():
        if name not in name_set:
            continue
        if isinstance(p, DTensor):
            local_full = p.full_tensor().detach()
            if p.ndim == 3:
                gathered = [torch.empty_like(local_full) for _ in range(dist.get_world_size(group=ep_group))]
                dist.all_gather(gathered, local_full.contiguous(), group=ep_group)
                local_full = torch.cat(gathered, dim=0)
            fsdp_state[name] = local_full.cpu()
        else:
            fsdp_state[name] = p.detach().cpu()

    del model, opt, name_to_param, muon_params
    device_backend = get_torch_device()
    if hasattr(device_backend, "empty_cache"):
        device_backend.empty_cache()

    if rank == 0:
        golden = _qwen3_moe_golden_state(device, full_shapes)
        common = set(fsdp_state.keys()) & set(golden.keys())
        assert common, f"no common keys between fsdp ({list(fsdp_state)[:4]}...) and golden ({list(golden)[:4]}...)"
        first_failure = None
        for k in sorted(common):
            a, b = fsdp_state[k], golden[k]
            if a.shape != b.shape:
                first_failure = first_failure or (k, f"shape mismatch: {a.shape} vs {b.shape}")
                continue
            diff = (a - b).abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            ref_max = b.abs().max().item()
            print(
                f"[rank0] {k}: shape={tuple(a.shape)} max|diff|={max_abs:.3e} "
                f"mean|diff|={mean_abs:.3e} max|ref|={ref_max:.3e}"
            )
            if max_abs > 5e-3 + 5e-3 * ref_max and first_failure is None:
                first_failure = (k, f"max|diff|={max_abs:.3e} max|ref|={ref_max:.3e}")
        if first_failure is not None:
            raise AssertionError(
                f"FSDP2+EP Muon update for {first_failure[0]!r} diverges from "
                f"single-device Muon (use_zero_comm={use_zero_comm}): {first_failure[1]}"
            )
        print(
            f"[rank0] Qwen3-MoE FSDP2+EP / single-device parity OK "
            f"(use_zero_comm={use_zero_comm}) across {len(common)} param(s)"
        )

    dist.barrier()
    dist.destroy_process_group()


def _has_devices(n: int) -> bool:
    try:
        return get_torch_device().device_count() >= n
    except Exception:
        return False


def _torchrun_cmd(nproc: int, port: int, mode: str, use_zero_comm: bool) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        f"--master_port={port}",
        os.path.abspath(__file__),
        f"--mode={mode}",
        f"--zero-comm={int(use_zero_comm)}",
    ]


@pytest.mark.skipif(not _has_devices(4), reason="device_count should be >= 4")
def test_dense_4gpu():
    cmd = _torchrun_cmd(nproc=4, port=29611, mode="dense", use_zero_comm=False)
    env = os.environ.copy()
    env.setdefault("NCCL_DEBUG", "WARN")
    result = subprocess.run(cmd, env=env, check=True)
    assert result.returncode == 0


@pytest.mark.skipif(not _has_devices(4), reason="device_count should be >= 4")
def test_qwen3_moe_default_backend_4gpu():
    """Default backend: ``Shard(1)`` on experts + ep_fsdp all-gather in Muon."""
    cmd = _torchrun_cmd(nproc=4, port=29612, mode="moe", use_zero_comm=False)
    env = os.environ.copy()
    env.setdefault("NCCL_DEBUG", "WARN")
    result = subprocess.run(cmd, env=env, check=True)
    assert result.returncode == 0


@pytest.mark.skipif(not _has_devices(4), reason="device_count should be >= 4")
def test_qwen3_moe_zero_comm_backend_4gpu():
    """Zero-comm backend: ``Shard(0)`` on experts + local batched NS."""
    cmd = _torchrun_cmd(nproc=4, port=29613, mode="moe", use_zero_comm=True)
    env = os.environ.copy()
    env.setdefault("NCCL_DEBUG", "WARN")
    result = subprocess.run(cmd, env=env, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "moe"], required=True)
    parser.add_argument("--zero-comm", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "dense":
        _run_dense()
    else:
        _run_qwen3_moe(use_zero_comm=bool(args.zero_comm))
