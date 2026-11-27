# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""``state_offload='cpu'`` must survive the parallelize-then-build-optimizer and DCP paths.

The unit tests in ``test_muon_unit.py`` drive ``DistributedMuon`` directly, so
they cannot catch a broken hand-off through ``build_optimizer`` ->
``_build_muon_with_adamw``, or a broken ``MultiOptimizer`` / DCP save-and-resume
round-trip over sharded DTensor parameters.

These tests build the optimizer the way the trainer does (including over
DTensor-sharded parameters on CPU and, when available, on an accelerator device
mesh) and verify both placement and full DCP save/load parity.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.utils.device import get_device_type


_CPU_PLAN = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture(scope="module")
def cpu_mesh(tmp_path_factory):
    """One gloo rank with a FileStore, enough for DTensor and DCP."""
    owns_pg = not dist.is_initialized()
    if owns_pg:
        store = dist.FileStore(str(tmp_path_factory.mktemp("muon_offload") / "store"), 1)
        dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    mesh = init_device_mesh("cpu", (1,))
    try:
        yield mesh
    finally:
        if owns_pg and dist.is_initialized():
            dist.destroy_process_group()


class _MuonModel(nn.Module):
    """A 2-D projection (Muon) beside a 1-D bias (AdamW), as the split expects."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Parameter(torch.randn(16, 32))
        self.bias = nn.Parameter(torch.randn(32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.proj.T + self.bias


class _ShardedMuonModel(nn.Module):
    """A sharded 2-D projection (Muon) beside a sharded 1-D bias (AdamW)."""

    def __init__(self, mesh) -> None:
        super().__init__()
        self.proj = nn.Parameter(distribute_tensor(torch.randn(8, 16), mesh, [Shard(0)]))
        self.bias = nn.Parameter(distribute_tensor(torch.randn(16), mesh, [Shard(0)]))


def _build(offload: bool, model: nn.Module | None = None):
    from veomni.arguments.arguments_types import OptimizerConfig
    from veomni.optim.optimizer import build_optimizer

    model = _MuonModel() if model is None else model
    config = OptimizerConfig(
        type="muon",
        lr=1e-4,
        muon_lr=1e-4,
        muon_ns_implementation="std",
        state_offload="cpu" if offload else "none",
    )
    optimizer = build_optimizer(model, lr=1e-4, optimizer_type="muon", optimizer_config=config)
    return model, optimizer


@_CPU_PLAN
def test_the_config_reaches_the_muon_half(cpu_mesh):
    """``state_offload`` must survive ``build_optimizer``; a broken hand-off is silent."""
    model, optimizer = _build(offload=True)

    muon = optimizer.optimizers_dict["muon"]
    assert muon.state_offload is True, (
        "the optimizer built but did not offload: `_build_muon_with_adamw` is not receiving the config"
    )
    # The AdamW sibling carries the same placement.
    assert type(optimizer.optimizers_dict["adamw"]).__name__ == "SwapAdamW"

    model.proj.grad = torch.randn_like(model.proj)
    model.bias.grad = torch.randn_like(model.bias)
    optimizer.step()

    assert model.proj in muon._host_momentum
    host_buf = muon._host_momentum[model.proj]
    assert host_buf.device.type == "cpu"
    bound = muon.state[model.proj]["momentum_buffer"]
    assert bound.device.type == "cpu"
    assert bound.data_ptr() == host_buf.data_ptr()


@_CPU_PLAN
def test_offload_is_off_by_default(cpu_mesh):
    """The default path must stay identical to what earlier runs measured."""
    model, optimizer = _build(offload=False)

    muon = optimizer.optimizers_dict["muon"]
    assert muon.state_offload is False
    assert type(optimizer.optimizers_dict["adamw"]).__name__ != "SwapAdamW"

    model.proj.grad = torch.randn_like(model.proj)
    model.bias.grad = torch.randn_like(model.bias)
    optimizer.step()

    assert not muon._host_momentum
    assert "momentum_buffer" in muon.state[model.proj]


@_CPU_PLAN
def test_offloaded_step_matches_a_resident_step(cpu_mesh):
    """Through the full builder path, offload stays numerically identical."""
    off_model, off_opt = _build(offload=True)
    res_model, res_opt = _build(offload=False)

    with torch.no_grad():
        res_model.proj.copy_(off_model.proj)
        res_model.bias.copy_(off_model.bias)

    torch.manual_seed(3)
    for _ in range(2):
        grad_p = torch.randn_like(off_model.proj)
        grad_b = torch.randn_like(off_model.bias)
        for model in (off_model, res_model):
            model.proj.grad = grad_p.clone()
            model.bias.grad = grad_b.clone()
        off_opt.step()
        res_opt.step()
        torch.testing.assert_close(off_model.proj, res_model.proj, atol=0.0, rtol=0.0)
        torch.testing.assert_close(off_model.bias, res_model.bias, atol=0.0, rtol=0.0)


@_CPU_PLAN
@pytest.mark.parametrize(
    "save_offload,load_offload",
    [
        (True, True),
        (True, False),
        (False, True),
    ],
)
def test_sharded_dcp_roundtrip_via_multi_optimizer(cpu_mesh, tmp_path, save_offload, load_offload):
    """Sharded ``state_offload='cpu'`` survives real ``MultiOptimizer`` + DCP save/load.

    Exercises:
    - DTensor rewrap on ``step()`` so ``lerp_`` never mixes plain Tensor and DTensor.
    - CPU-backed DTensor binding in ``self.state[p]["momentum_buffer"]`` so DCP
      exports global shape/placements and ``set_optimizer_state_dict`` unflattens
      every Muon momentum buffer on resume (including cross-flavor offload <-> resident).
    """
    from veomni.checkpoint.dcp_checkpointer import OptimizerState

    torch.manual_seed(17)
    src_model, src_opt = _build(offload=save_offload, model=_ShardedMuonModel(cpu_mesh))

    for step_idx in range(2):
        torch.manual_seed(200 + step_idx)
        src_model.proj.grad = distribute_tensor(torch.randn(8, 16), cpu_mesh, [Shard(0)])
        src_model.bias.grad = distribute_tensor(torch.randn(16), cpu_mesh, [Shard(0)])
        src_opt.step()
        src_opt.zero_grad()

    src_muon = src_opt.optimizers_dict["muon"]
    expected_mom = src_muon.state[src_model.proj]["momentum_buffer"].to_local().clone()
    assert expected_mom.abs().sum().item() > 0.0
    if save_offload:
        assert src_model.proj in src_muon._host_momentum
        assert src_muon._host_momentum[src_model.proj].device.type == "cpu"
        assert not hasattr(src_muon._host_momentum[src_model.proj], "placements")
        exported_buf = src_muon.state[src_model.proj]["momentum_buffer"]
        assert isinstance(exported_buf, DTensor)
        assert exported_buf.placements == src_model.proj.placements
        assert exported_buf.to_local().device.type == "cpu"

    ckpt_dir = tmp_path / f"ckpt_{save_offload}_{load_offload}"
    dcp.save({"optimizer": OptimizerState(src_model, src_opt)}, checkpoint_id=str(ckpt_dir))

    dst_model, dst_opt = _build(offload=load_offload, model=_ShardedMuonModel(cpu_mesh))
    with torch.no_grad():
        dst_model.proj.to_local().copy_(src_model.proj.to_local())
        dst_model.bias.to_local().copy_(src_model.bias.to_local())

    dcp.load({"optimizer": OptimizerState(dst_model, dst_opt, load=True)}, checkpoint_id=str(ckpt_dir))

    dst_muon = dst_opt.optimizers_dict["muon"]
    restored_mom = (
        dst_muon._host_momentum[dst_model.proj]
        if load_offload
        else dst_muon.state[dst_model.proj]["momentum_buffer"].to_local()
    )
    torch.testing.assert_close(restored_mom, expected_mom, atol=0.0, rtol=0.0)

    # Next step after resume must match bit-for-bit across both Muon and AdamW halves.
    torch.manual_seed(300)
    grad_p = torch.randn(8, 16)
    grad_b = torch.randn(16)
    src_model.proj.grad = distribute_tensor(grad_p.clone(), cpu_mesh, [Shard(0)])
    src_model.bias.grad = distribute_tensor(grad_b.clone(), cpu_mesh, [Shard(0)])
    dst_model.proj.grad = distribute_tensor(grad_p.clone(), cpu_mesh, [Shard(0)])
    dst_model.bias.grad = distribute_tensor(grad_b.clone(), cpu_mesh, [Shard(0)])
    src_opt.step()
    dst_opt.step()

    torch.testing.assert_close(dst_model.proj.to_local(), src_model.proj.to_local(), atol=0.0, rtol=0.0)
    torch.testing.assert_close(dst_model.bias.to_local(), src_model.bias.to_local(), atol=0.0, rtol=0.0)


@_CPU_PLAN
def test_ortho_fsdp_group_all2all_overlap_fn_matches_sync(cpu_mesh, monkeypatch):
    """Optional ``overlap_fn`` runs during async all-to-all without changing numerics."""
    from veomni.optim.muon import DEFAULT_NS_COEFFICIENTS, DistributedMuon

    events: list[str] = []

    class _FakeWork:
        def __init__(self, pairs) -> None:
            self._pairs = pairs

        def wait(self) -> None:
            events.append("wait")
            for out_t, in_t in self._pairs:
                out_t.copy_(in_t)

    def _fake_all_to_all(output_list, input_list, group=None, async_op=False):
        if async_op:
            events.append("async_launch")
            for out_t in output_list:
                out_t.fill_(float("nan"))
            return _FakeWork([(out_t, in_t.clone()) for out_t, in_t in zip(output_list, input_list)])
        for out_t, in_t in zip(output_list, input_list):
            out_t.copy_(in_t)
        return None

    monkeypatch.setattr(dist, "all_to_all", _fake_all_to_all)

    torch.manual_seed(29)
    p = nn.Parameter(distribute_tensor(torch.randn(8, 16), cpu_mesh, [Shard(0)]))
    opt = DistributedMuon([p], lr=1e-2, ns_implementation="std")
    upd = distribute_tensor(torch.randn(8, 16), cpu_mesh, [Shard(0)])
    ns_kwargs = {
        "ns_coefficients": DEFAULT_NS_COEFFICIENTS,
        "ns_steps": 5,
        "eps": 1e-7,
        "ns_implementation": "std",
        "gram_ns_reset_iterations": (2,),
        "head_blocks": 1,
        "expert_ns_max_experts": -1,
    }

    sync_out = opt._ortho_fsdp_group_all2all([upd], cpu_mesh, ns_kwargs)
    async_out = opt._ortho_fsdp_group_all2all([upd], cpu_mesh, ns_kwargs, overlap_fn=lambda: events.append("overlap"))

    assert events == ["async_launch", "overlap", "wait", "async_launch", "overlap", "wait"]
    torch.testing.assert_close(async_out[0], sync_out[0], atol=0.0, rtol=0.0)


@pytest.mark.skipif(get_device_type() == "cpu", reason="accelerator backend required")
def test_dcp_roundtrip_on_accelerator_device_mesh(tmp_path):
    """Exercises mismatched-device DTensor rewrap, pinned host memory, and DCP on a real accelerator."""
    from veomni.checkpoint.dcp_checkpointer import OptimizerState

    device_type = get_device_type()
    owns_pg = not dist.is_initialized()
    if owns_pg:
        store = dist.FileStore(str(tmp_path / "store"), 1)
        dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    try:
        mesh = init_device_mesh(device_type, (1,))

        class _DeviceSharded(nn.Module):
            def __init__(self, seed: int) -> None:
                super().__init__()
                torch.manual_seed(seed)
                self.proj = nn.Parameter(
                    DTensor.from_local(torch.randn(8, 16, device=device_type), mesh, [Shard(0)], run_check=False)
                )
                self.bias = nn.Parameter(
                    DTensor.from_local(torch.randn(16, device=device_type), mesh, [Shard(0)], run_check=False)
                )

        src_model, src_opt = _build(offload=True, model=_DeviceSharded(seed=41))
        res_model, res_opt = _build(offload=False, model=_DeviceSharded(seed=41))
        for step_idx in range(2):
            torch.manual_seed(400 + step_idx)
            gp = torch.randn(8, 16, device=device_type)
            gb = torch.randn(16, device=device_type)
            for m in (src_model, res_model):
                m.proj.grad = DTensor.from_local(gp.clone(), mesh, [Shard(0)], run_check=False)
                m.bias.grad = DTensor.from_local(gb.clone(), mesh, [Shard(0)], run_check=False)
            src_opt.step()
            res_opt.step()
            src_opt.zero_grad()
            res_opt.zero_grad()

        torch.testing.assert_close(src_model.proj.to_local(), res_model.proj.to_local(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(src_model.bias.to_local(), res_model.bias.to_local(), atol=1e-5, rtol=1e-5)

        src_muon = src_opt.optimizers_dict["muon"]
        holder = src_muon.state[src_model.proj]["momentum_buffer"]
        assert isinstance(holder, DTensor)
        assert holder.to_local().device.type == "cpu"
        assert holder.to_local().data_ptr() == src_muon._host_momentum[src_model.proj].data_ptr()

        parallel_state = SimpleNamespace(dp_mode="fsdp2")
        ckpt_dir = tmp_path / "device_ckpt"
        dcp.save(
            {"optimizer": OptimizerState(src_model, src_opt, parallel_state=parallel_state)},
            checkpoint_id=str(ckpt_dir),
        )

        dst_model, dst_opt = _build(offload=True, model=_DeviceSharded(seed=99))
        dcp.load(
            {"optimizer": OptimizerState(dst_model, dst_opt, parallel_state=parallel_state, load=True)},
            checkpoint_id=str(ckpt_dir),
        )

        dst_muon = dst_opt.optimizers_dict["muon"]
        torch.testing.assert_close(
            dst_muon._host_momentum[dst_model.proj],
            src_muon._host_momentum[src_model.proj],
            atol=0.0,
            rtol=0.0,
        )
        assert dst_muon.state[dst_model.proj]["momentum_buffer"].to_local().device.type == "cpu"
    finally:
        if owns_pg and dist.is_initialized():
            dist.destroy_process_group()
