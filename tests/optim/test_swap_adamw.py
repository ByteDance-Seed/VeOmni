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

"""Single-process CPU parity tests for ``SwapAdamW``.

The optimizer is exercised over one-rank ``Shard(0)`` DTensors (a DTensor layout
the optimizer sees under FSDP2/EP) and compared step-by-step against
``torch.optim.AdamW`` running on the plain local tensors. This covers the
swap/offload mechanics and the update math without needing an accelerator; the
device-stream path is only taken on an accelerator host.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.optim.swap_adamw import SwapAdamW
from veomni.utils.device import get_device_type


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture(scope="module")
def cpu_mesh(tmp_path_factory):
    owns_pg = not dist.is_initialized()
    if owns_pg:
        store = dist.FileStore(str(tmp_path_factory.mktemp("swap_adamw") / "store"), 1)
        dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    mesh = init_device_mesh("cpu", (1,))
    yield mesh
    if owns_pg and dist.is_initialized():
        dist.destroy_process_group()


def _dtensor(local: torch.Tensor, mesh) -> DTensor:
    return DTensor.from_local(local, mesh, [Shard(0)], run_check=False)


def _swap_param(init: torch.Tensor, mesh) -> nn.Parameter:
    return nn.Parameter(_dtensor(init.clone(), mesh))


def _partition(local: torch.Tensor, mesh) -> DTensor:
    """Build a grad DTensor matching a one-rank ``Shard(0)`` parameter."""
    return _dtensor(local, mesh)


def test_default_construction_resolves_device_dependent_pinning(cpu_mesh):
    # Regression: the pin_memory gate reads the resolved device type, so a
    # construction with all defaults must not raise (CPU -> pinning disabled).
    param = _swap_param(torch.randn(4, 3), cpu_mesh)
    opt = SwapAdamW([param])
    on_accelerator = get_device_type() != "cpu"
    assert opt.pin_memory is on_accelerator
    assert opt._use_streams is on_accelerator


def test_fused_matches_elementwise_on_cpu(cpu_mesh):
    # ``_use_fused`` is the production path (CUDA/NPU); pinning it on CPU exercises
    # the same ``torch._fused_adamw_`` call contract against the elementwise path.
    torch.manual_seed(11)
    shape = (6, 4)
    init = torch.randn(shape)
    fused_param = _swap_param(init, cpu_mesh)
    eager_param = _swap_param(init, cpu_mesh)
    fused_opt = SwapAdamW([fused_param], lr=1e-2, betas=(0.9, 0.95), pin_memory=False)
    eager_opt = SwapAdamW([eager_param], lr=1e-2, betas=(0.9, 0.95), pin_memory=False)
    fused_opt._use_fused = True
    eager_opt._use_fused = False

    for _ in range(3):
        grad = torch.randn(shape)
        fused_param.grad = _partition(grad.clone(), cpu_mesh)
        eager_param.grad = _partition(grad.clone(), cpu_mesh)
        fused_opt.step()
        eager_opt.step()

    torch.testing.assert_close(fused_param.to_local(), eager_param.to_local(), atol=1e-6, rtol=1e-6)
    for key in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            fused_opt._host_states[fused_param][key],
            eager_opt._host_states[eager_param][key],
            atol=1e-6,
            rtol=1e-6,
        )


def test_mixed_dtype_group_fused_matches_torch(cpu_mesh, monkeypatch):
    # One param group mixing fp32 and bf16 params must split into per-dtype
    # fused calls (the torch._fused_adam grouping contract) and match stock
    # elementwise AdamW step by step.
    torch.manual_seed(12)
    init_f, init_b = torch.randn(6, 4), torch.randn(5, 4)

    ref_f, ref_b = nn.Parameter(init_f.clone()), nn.Parameter(init_b.clone().to(torch.bfloat16))
    par_f = _swap_param(init_f, cpu_mesh)
    par_b = nn.Parameter(_swap_param(init_b, cpu_mesh).to(torch.bfloat16))

    ref_opt = torch.optim.AdamW([ref_f, ref_b], lr=1e-2, betas=(0.9, 0.95), foreach=False)
    swap_opt = SwapAdamW([par_f, par_b], lr=1e-2, betas=(0.9, 0.95), pin_memory=False)
    swap_opt._use_fused = True  # exercise the (device, dtype) bucketing in the fused path

    # Pin the bucketing itself: torch's fused kernel also groups by dtype
    # internally, so a mixed call would succeed and the parity checks alone
    # would not catch a regression that drops the split.
    fused_dtypes = []
    real_fused = torch._fused_adamw_

    def recording_fused(*args, **kwargs):
        fused_dtypes.append({t.dtype for t in args[0]})
        return real_fused(*args, **kwargs)

    monkeypatch.setattr(torch, "_fused_adamw_", recording_fused)

    for _ in range(3):
        grad_f, grad_b = torch.randn(6, 4), torch.randn(5, 4)
        ref_f.grad, ref_b.grad = grad_f.clone(), grad_b.clone().to(torch.bfloat16)
        par_f.grad = _partition(grad_f.clone(), cpu_mesh)
        par_b.grad = _partition(grad_b.clone(), cpu_mesh).to(torch.bfloat16)
        ref_opt.step()
        swap_opt.step()

    assert len(fused_dtypes) == 6, fused_dtypes
    assert all(len(bucket) == 1 for bucket in fused_dtypes), fused_dtypes

    torch.testing.assert_close(par_f.to_local(), ref_f.detach(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(par_b.to_local(), ref_b.detach(), atol=1e-2, rtol=1e-2)
    for key in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(swap_opt._host_states[par_f][key], ref_opt.state[ref_f][key], atol=1e-6, rtol=1e-6)


def test_pinned_allocation_degrades_to_pageable(cpu_mesh, monkeypatch):
    # Page-locked host memory is finite; exhaustion at build must degrade with
    # a warning instead of killing the run (the async_offload fallback pattern).
    real_zeros = torch.zeros

    def failing_pinned_zeros(*args, **kwargs):
        if kwargs.get("pin_memory"):
            raise RuntimeError("simulated pinned-memory exhaustion")
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", failing_pinned_zeros)
    param = _swap_param(torch.randn(4, 3), cpu_mesh)
    opt = SwapAdamW([param], pin_memory=False)
    # Simulate the accelerator-side state: pinning requested, allocation fails.
    opt.pin_memory = True
    second = _swap_param(torch.randn(2, 3), cpu_mesh)
    opt.add_param_group({"params": [second]})

    # One-time degradation: the flag flips and the run continues on pageable buffers.
    assert opt.pin_memory is False
    assert torch.count_nonzero(opt._host_states[second]["exp_avg"]).item() == 0
    second.grad = _partition(torch.randn(2, 3), cpu_mesh)
    opt.step()
    assert int(opt.state[second]["step"]) == 1


def test_matches_torch_adamw_step_by_step(cpu_mesh):
    torch.manual_seed(0)
    shape = (6, 4)
    init = torch.randn(shape)

    ref = nn.Parameter(init.clone())
    swapped = _swap_param(init, cpu_mesh)

    ref_opt = torch.optim.AdamW([ref], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False)
    swap_opt = SwapAdamW([swapped], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, pin_memory=False)

    for _ in range(5):
        grad = torch.randn(shape)
        ref.grad = grad.clone()
        swapped.grad = _partition(grad.clone(), cpu_mesh)
        ref_opt.step()
        swap_opt.step()

        torch.testing.assert_close(swapped.to_local(), ref.detach(), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            swap_opt._host_states[swapped]["exp_avg"],
            ref_opt.state[ref]["exp_avg"],
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            swap_opt._host_states[swapped]["exp_avg_sq"],
            ref_opt.state[ref]["exp_avg_sq"],
            atol=1e-6,
            rtol=1e-6,
        )
        assert swap_opt.state[swapped]["step"] == ref_opt.state[ref]["step"]

    assert swapped.to_local().dtype == ref.dtype


def test_device_states_are_freed_after_step(cpu_mesh):
    torch.manual_seed(1)
    init = torch.randn(8, 5)
    param = _swap_param(init, cpu_mesh)
    opt = SwapAdamW([param], pin_memory=False)

    param.grad = _partition(torch.randn(8, 5), cpu_mesh)
    opt.step()

    for key in ("exp_avg", "exp_avg_sq"):
        local = opt._device_states[param][key]
        assert local.untyped_storage().nbytes() == 0, f"{key} device storage was not released"
        assert opt._host_states[param][key].untyped_storage().nbytes() > 0


def test_swap_all_round_trip(cpu_mesh):
    torch.manual_seed(2)
    init = torch.randn(4, 3)
    param = _swap_param(init, cpu_mesh)
    opt = SwapAdamW([param], pin_memory=False)

    param.grad = _partition(torch.randn(4, 3), cpu_mesh)
    opt.step()

    host_before = opt._host_states[param]["exp_avg"].clone()
    opt.swap_all_to_device()
    local = opt._device_states[param]["exp_avg"]
    assert local.untyped_storage().nbytes() > 0
    torch.testing.assert_close(local, host_before, atol=0, rtol=0)

    opt.swap_all_to_host()
    assert opt._device_states[param]["exp_avg"].untyped_storage().nbytes() == 0
    torch.testing.assert_close(opt._host_states[param]["exp_avg"], host_before, atol=0, rtol=0)


def test_amsgrad_matches_torch(cpu_mesh):
    torch.manual_seed(3)
    shape = (5, 3)
    init = torch.randn(shape)

    ref = nn.Parameter(init.clone())
    swapped = _swap_param(init, cpu_mesh)

    ref_opt = torch.optim.AdamW([ref], lr=5e-3, betas=(0.9, 0.95), amsgrad=True, foreach=False)
    swap_opt = SwapAdamW([swapped], lr=5e-3, betas=(0.9, 0.95), amsgrad=True, pin_memory=False)

    for _ in range(3):
        grad = torch.randn(shape)
        ref.grad = grad.clone()
        swapped.grad = _partition(grad.clone(), cpu_mesh)
        ref_opt.step()
        swap_opt.step()

    torch.testing.assert_close(swapped.to_local(), ref.detach(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        swap_opt._host_states[swapped]["max_exp_avg_sq"],
        ref_opt.state[ref]["max_exp_avg_sq"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_maximize_matches_torch(cpu_mesh):
    torch.manual_seed(8)
    shape = (6, 4)
    init = torch.randn(shape)

    ref = nn.Parameter(init.clone())
    swapped = _swap_param(init, cpu_mesh)

    ref_opt = torch.optim.AdamW(
        [ref], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, maximize=True, foreach=False
    )
    swap_opt = SwapAdamW(
        [{"params": [swapped], "weight_decay": 0.1, "maximize": True}],
        lr=1e-2,
        betas=(0.9, 0.95),
        eps=1e-8,
        pin_memory=False,
    )
    # Force the elementwise path so this guards ``_update_param`` even on a host
    # where an accelerator registers ``torch._fused_adamw_``.
    swap_opt._use_fused = False

    for _ in range(5):
        grad = torch.randn(shape)
        ref.grad = grad.clone()
        swapped.grad = _partition(grad.clone(), cpu_mesh)
        ref_opt.step()
        swap_opt.step()

        torch.testing.assert_close(swapped.to_local(), ref.detach(), atol=1e-6, rtol=1e-6)


def test_param_groups_and_missing_grad(cpu_mesh):
    torch.manual_seed(4)
    init_a = torch.randn(4, 2)
    init_b = torch.randn(3, 2)
    init_c = torch.randn(2, 2)

    ref_a, ref_b, ref_c = (nn.Parameter(t.clone()) for t in (init_a, init_b, init_c))
    par_a, par_b, par_c = (_swap_param(t, cpu_mesh) for t in (init_a, init_b, init_c))

    ref_opt = torch.optim.AdamW(
        [{"params": [ref_a], "weight_decay": 0.1}, {"params": [ref_b, ref_c], "weight_decay": 0.0}],
        lr=1e-2,
        betas=(0.9, 0.95),
        foreach=False,
    )
    swap_opt = SwapAdamW(
        [{"params": [par_a], "weight_decay": 0.1}, {"params": [par_b, par_c], "weight_decay": 0.0}],
        lr=1e-2,
        betas=(0.9, 0.95),
        pin_memory=False,
    )

    grad_a = torch.randn(4, 2)
    grad_c = torch.randn(2, 2)
    ref_a.grad = grad_a.clone()
    ref_c.grad = grad_c.clone()
    # ref_b / par_b get no gradient this step and must be left untouched.
    ref_b.grad = None
    par_a.grad = _partition(grad_a.clone(), cpu_mesh)
    par_b.grad = None
    par_c.grad = _partition(grad_c.clone(), cpu_mesh)

    ref_opt.step()
    swap_opt.step()

    torch.testing.assert_close(par_a.to_local(), ref_a.detach(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(par_b.to_local(), ref_b.detach(), atol=0, rtol=0)
    torch.testing.assert_close(par_c.to_local(), ref_c.detach(), atol=1e-6, rtol=1e-6)


def test_add_param_group_matches_eager_groups(cpu_mesh):
    torch.manual_seed(10)
    init_a, init_b = torch.randn(4, 3), torch.randn(2, 3)

    ref_a, ref_b = nn.Parameter(init_a.clone()), nn.Parameter(init_b.clone())
    par_a, par_b = _swap_param(init_a, cpu_mesh), _swap_param(init_b, cpu_mesh)

    ref_opt = torch.optim.AdamW(
        [{"params": [ref_a], "weight_decay": 0.1}, {"params": [ref_b], "weight_decay": 0.0}],
        lr=1e-2,
        foreach=False,
    )
    swap_opt = SwapAdamW([{"params": [par_a], "weight_decay": 0.1}], lr=1e-2, pin_memory=False)
    swap_opt.add_param_group({"params": [par_b], "weight_decay": 0.0})

    grad_a, grad_b = torch.randn(4, 3), torch.randn(2, 3)
    ref_a.grad, ref_b.grad = grad_a.clone(), grad_b.clone()
    par_a.grad = _partition(grad_a.clone(), cpu_mesh)
    par_b.grad = _partition(grad_b.clone(), cpu_mesh)

    ref_opt.step()
    swap_opt.step()

    torch.testing.assert_close(par_a.to_local(), ref_a.detach(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(par_b.to_local(), ref_b.detach(), atol=1e-6, rtol=1e-6)
    assert int(swap_opt.state[par_b]["step"]) == 1


def test_add_param_group_rejects_different_amsgrad(cpu_mesh):
    # The moment-state layout (and the precomputed state byte sizes) is fixed at
    # construction, so a group with a different amsgrad must be rejected.
    first = _swap_param(torch.randn(4, 3), cpu_mesh)
    second = _swap_param(torch.randn(4, 3), cpu_mesh)
    opt = SwapAdamW([first], amsgrad=False, pin_memory=False)
    with pytest.raises(ValueError, match="amsgrad"):
        opt.add_param_group({"params": [second], "amsgrad": True})


def test_load_state_dict_rejects_different_amsgrad(cpu_mesh):
    src = _swap_param(torch.randn(4, 3), cpu_mesh)
    dst = _swap_param(torch.randn(4, 3), cpu_mesh)
    src_opt = SwapAdamW([src], amsgrad=True, pin_memory=False)
    src.grad = _partition(torch.randn(4, 3), cpu_mesh)
    src_opt.step()
    state_dict = src_opt.state_dict()
    dst_opt = SwapAdamW([dst], amsgrad=False, pin_memory=False)
    with pytest.raises(ValueError, match="amsgrad"):
        dst_opt.load_state_dict(state_dict)


def test_load_state_dict_tolerates_missing_param_state(cpu_mesh):
    # The flattened loader represents a param with no saved state as empty dicts
    # (an unused MoE expert); loading keeps the zero state instead of crashing.
    param = _swap_param(torch.randn(4, 3), cpu_mesh)
    opt = SwapAdamW([param], pin_memory=False)
    opt._load_param_state(param, {"exp_avg": {}, "exp_avg_sq": {}, "step": {}})
    assert int(opt.state[param]["step"]) == 0
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.count_nonzero(opt._host_states[param][key]).item() == 0


def test_sparse_checkpoint_resets_omitted_param_state(cpu_mesh):
    """Loading a sparse checkpoint into a reused optimizer must not keep history.

    Stock ``Optimizer.load_state_dict`` installs a fresh state mapping, so a
    parameter the checkpoint omits starts from zero moments. The flattened DCP
    loader represents that absence as an entry whose *values* are empty dicts, so
    the reset has to key off the values, not on the entry being present.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    torch.manual_seed(41)

    class _Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(4, 3))
            self.b = nn.Parameter(torch.randn(4, 3))

    class _Swapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _swap_param(torch.randn(4, 3), cpu_mesh)
            self.b = _swap_param(torch.randn(4, 3), cpu_mesh)

    # Source checkpoint: stock AdamW that only ever stepped ``a`` -> sparse.
    src = _Plain()
    src_opt = torch.optim.AdamW(src.parameters(), lr=1e-2, foreach=False)
    src.a.grad = torch.randn(4, 3)
    src_opt.step()
    assert len(src_opt.state[src.a]) > 0 and len(src_opt.state[src.b]) == 0

    flat = get_optimizer_state_dict(
        src, optimizers=[src_opt], options=StateDictOptions(flatten_optimizer_state_dict=True)
    )
    assert not any(key.startswith("state.b") for key in flat), sorted(flat)

    # Target: a SwapAdamW that already stepped both parameters.
    dst = _Swapped()
    swap_opt = SwapAdamW(list(dst.parameters()), lr=1e-2, pin_memory=False)
    for _ in range(3):
        for param in dst.parameters():
            param.grad = _partition(torch.randn(4, 3), cpu_mesh)
        swap_opt.step()
    assert int(swap_opt.state[dst.b]["step"]) == 3
    assert torch.count_nonzero(swap_opt._host_states[dst.b]["exp_avg"]).item() > 0

    # Load through the same flatten/unflatten path DCP uses.
    set_optimizer_state_dict(
        dst, swap_opt, optim_state_dict=flat, options=StateDictOptions(flatten_optimizer_state_dict=True)
    )

    assert int(swap_opt.state[dst.a]["step"]) == 1  # restored from the checkpoint
    assert int(swap_opt.state[dst.b]["step"]) == 0  # absent from it -> reset
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.count_nonzero(swap_opt._host_states[dst.b][key]).item() == 0

    # The next updates must match stock AdamW loaded from the same checkpoint.
    ref = _Plain()
    with torch.no_grad():
        ref.a.copy_(dst.a.to_local())
        ref.b.copy_(dst.b.to_local())
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-2, foreach=False)
    ref_opt.load_state_dict(src_opt.state_dict())

    for _ in range(2):
        grads = {"a": torch.randn(4, 3), "b": torch.randn(4, 3)}
        dst.a.grad = _partition(grads["a"].clone(), cpu_mesh)
        dst.b.grad = _partition(grads["b"].clone(), cpu_mesh)
        ref.a.grad = grads["a"].clone()
        ref.b.grad = grads["b"].clone()
        swap_opt.step()
        ref_opt.step()
        torch.testing.assert_close(dst.a.to_local(), ref.a.detach(), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(dst.b.to_local(), ref.b.detach(), atol=1e-6, rtol=1e-6)


def test_multi_optimizer_steps_and_frees_storages(cpu_mesh):
    from veomni.optim.optimizer import MultiOptimizer

    torch.manual_seed(5)
    init_a = torch.randn(4, 2)
    init_b = torch.randn(3, 2)
    par_a = _swap_param(init_a, cpu_mesh)
    par_b = _swap_param(init_b, cpu_mesh)

    opt_a = SwapAdamW([{"params": [par_a], "weight_decay": 0.1}], lr=1e-2, pin_memory=False)
    opt_b = SwapAdamW([{"params": [par_b], "weight_decay": 0.0}], lr=1e-2, pin_memory=False)
    multi = MultiOptimizer(nn.Module(), {"ep": opt_a, "non_extra_parallel": opt_b}, ["ep", "non_extra_parallel"])

    par_a.grad = _partition(torch.randn(4, 2), cpu_mesh)
    par_b.grad = _partition(torch.randn(3, 2), cpu_mesh)
    multi.step()
    multi.zero_grad()

    for opt, param in ((opt_a, par_a), (opt_b, par_b)):
        for key in ("exp_avg", "exp_avg_sq"):
            assert opt._device_states[param][key].untyped_storage().nbytes() == 0


def test_state_dict_reads_from_host(cpu_mesh):
    torch.manual_seed(6)
    init = torch.randn(4, 3)
    param = _swap_param(init, cpu_mesh)
    opt = SwapAdamW([param], pin_memory=False)

    param.grad = _partition(torch.randn(4, 3), cpu_mesh)
    opt.step()

    sd = opt.state_dict()
    assert len(sd["state"]) == 1
    (state,) = sd["state"].values()
    for key in ("exp_avg", "exp_avg_sq"):
        holder = state[key]
        local = holder.to_local() if isinstance(holder, DTensor) else holder
        assert local.data_ptr() == opt._host_states[param][key].data_ptr(), (
            f"{key} should be read from the host buffer"
        )


def test_state_dict_load_state_dict_round_trip(cpu_mesh):
    torch.manual_seed(7)
    init = torch.randn(4, 3)

    src_param = _swap_param(init, cpu_mesh)
    src = SwapAdamW([src_param], lr=1e-2, pin_memory=False)
    for _ in range(3):
        src_param.grad = _partition(torch.randn(4, 3), cpu_mesh)
        src.step()

    sd = src.state_dict()

    dst_param = _swap_param(init, cpu_mesh)
    dst = SwapAdamW([dst_param], lr=1e-2, pin_memory=False)
    dst.load_state_dict(sd)

    for key in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            dst._host_states[dst_param][key],
            src._host_states[src_param][key],
            atol=0,
            rtol=0,
        )
    assert int(dst.state[dst_param]["step"]) == int(src.state[src_param]["step"])


def test_device_free_bytes_driver_free_plus_pool_cache(monkeypatch):
    from veomni.optim import swap_adamw

    class _FakeDevice:
        @staticmethod
        def mem_get_info():
            return 10 << 30, 64 << 30

        @staticmethod
        def memory_reserved():
            return 12 << 30

        @staticmethod
        def memory_allocated():
            return 8 << 30

    monkeypatch.setattr(swap_adamw, "get_torch_device", lambda: _FakeDevice)
    param = nn.Parameter(torch.randn(4, 3))
    opt = SwapAdamW([param], pin_memory=False)
    assert opt._device_free_bytes() == 14 << 30


def test_device_free_bytes_falls_back_to_total_minus_allocated(monkeypatch):
    from veomni.optim import swap_adamw

    class _Props:
        total_memory = 64 << 30

    class _FakeDevice:
        @staticmethod
        def mem_get_info():
            raise AttributeError("mem_get_info unavailable")

        @staticmethod
        def get_device_properties(_device_id):
            return _Props()

        @staticmethod
        def memory_allocated():
            return 8 << 30

    monkeypatch.setattr(swap_adamw, "get_torch_device", lambda: _FakeDevice)
    monkeypatch.setattr(swap_adamw, "get_device_id", lambda: 0)
    param = nn.Parameter(torch.randn(4, 3))
    opt = SwapAdamW([param], pin_memory=False)
    assert opt._device_free_bytes() == 56 << 30


def test_could_not_fit_guard_raises_for_oversized_param(cpu_mesh, monkeypatch):
    # A param whose states exceed the batch budget fails with an actionable
    # error instead of a raw device OOM.
    param = _swap_param(torch.randn(4, 3), cpu_mesh)
    opt = SwapAdamW([param], lr=1e-2, pin_memory=False)
    monkeypatch.setattr(opt, "_batch_budget_bytes", lambda: 8)
    param.grad = _partition(torch.randn(4, 3), cpu_mesh)
    with pytest.raises(RuntimeError, match="could not fit any optimizer state"):
        opt.step()

    # Empty shards (uneven EP/FSDP sharding) move zero bytes: they must not
    # disarm the guard for the next real parameter (bytes_moved stays 0).
    empty = _swap_param(torch.empty(0, 3), cpu_mesh)
    real = _swap_param(torch.randn(4, 3), cpu_mesh)
    opt = SwapAdamW([empty, real], lr=1e-2, pin_memory=False)
    monkeypatch.setattr(opt, "_batch_budget_bytes", lambda: 0)
    empty.grad = _partition(torch.empty(0, 3), cpu_mesh)
    real.grad = _partition(torch.randn(4, 3), cpu_mesh)
    with pytest.raises(RuntimeError, match="could not fit any optimizer state"):
        opt.step()


# ---------------------------------------------------------------------------
# rewrap_dtensor_local (veomni.utils.dtensor_utils)
# ---------------------------------------------------------------------------


def test_rewrap_dtensor_local_same_device(cpu_mesh):
    from veomni.utils.dtensor_utils import rewrap_dtensor_local

    local = torch.arange(8.0)
    dt = rewrap_dtensor_local(local, mesh=cpu_mesh, placements=[Shard(0)])
    assert isinstance(dt, DTensor)
    assert dt.to_local().device.type == "cpu"
    assert tuple(dt.shape) == (8,)  # one-rank mesh -> global == local
    assert dt.placements == (Shard(0),)


def test_rewrap_dtensor_local_infers_global_shape(cpu_mesh):
    from veomni.utils.dtensor_utils import rewrap_dtensor_local

    local = torch.arange(6.0)
    inferred = rewrap_dtensor_local(local, mesh=cpu_mesh, placements=[Shard(0)])
    explicit = rewrap_dtensor_local(local, mesh=cpu_mesh, placements=[Shard(0)], shape=torch.Size([6]), stride=(1,))
    assert inferred.shape == explicit.shape == torch.Size([6])
    assert inferred.stride() == explicit.stride() == (1,)


def test_rewrap_dtensor_local_does_not_move_mismatched_local(cpu_mesh):
    """A local that is not on the mesh's device takes the direct-construction
    branch (no ``from_local`` device copy), so it stays where it is."""
    from veomni.utils.dtensor_utils import rewrap_dtensor_local

    local = torch.empty(5, device="meta")
    dt = rewrap_dtensor_local(local, mesh=cpu_mesh, placements=[Shard(0)])
    assert isinstance(dt, DTensor)
    assert dt.to_local().device.type == "meta"
    assert tuple(dt.shape) == (5,)


def test_rewrap_dtensor_local_preserves_non_contiguous_local(cpu_mesh):
    """The mismatched-device branch is a storage-sharing view: an identity
    ``view_as`` never requires contiguity, so strided locals keep their strides."""
    from veomni.utils.dtensor_utils import rewrap_dtensor_local

    local = torch.empty(4, 3, device="meta").t()  # shape (3, 4), stride (1, 3)
    dt = rewrap_dtensor_local(local, mesh=cpu_mesh, placements=[Shard(0)])
    assert dt.to_local().device.type == "meta"
    assert tuple(dt.to_local().shape) == (3, 4)
    assert dt.to_local().stride() == (1, 3)


def test_async_save_snapshots_state_before_later_steps(cpu_mesh, tmp_path):
    """``dcp.async_save`` must capture the state at the save call, not at write time.

    ``_bind_host_states`` hands DCP the *live* host buffers. That is safe because
    torch's ``FileSystemWriter`` is a ``BlockingAsyncStager``: it deep-copies the
    state dict synchronously on the calling thread before the background write
    starts, so steps taken after ``async_save`` returns cannot tear the
    checkpoint. This test pins that invariant: it fails if staging ever becomes
    asynchronous.
    """
    from types import SimpleNamespace

    import torch.distributed.checkpoint as dcp

    from veomni.checkpoint.dcp_checkpointer import OptimizerState

    torch.manual_seed(9)
    parallel_state = SimpleNamespace(dp_mode="fsdp2")
    model = nn.Linear(4, 4)
    opt = SwapAdamW(list(model.parameters()), lr=1e-2, pin_memory=False)

    def step(seed: int) -> None:
        torch.manual_seed(seed)
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        opt.step()

    step(0)
    step(1)
    future = dcp.async_save(
        state_dict={"optimizer": OptimizerState(model, opt, parallel_state=parallel_state)},
        checkpoint_id=tmp_path,
    )
    # The staging copy ran inside async_save, so this captures exactly the saved state.
    snapshot = {
        param: {key: tensor.clone() for key, tensor in state.items()} for param, state in opt._host_states.items()
    }
    snapshot_steps = {param: int(opt.state[param]["step"]) for param in snapshot}

    step(2)  # Keep training: mutates the live host buffers while the write runs.
    step(3)
    future.result()

    dst_model = nn.Linear(4, 4)
    dst_opt = SwapAdamW(list(dst_model.parameters()), lr=1e-2, pin_memory=False)
    dcp.load(
        {"optimizer": OptimizerState(dst_model, dst_opt, parallel_state=parallel_state, load=True)},
        checkpoint_id=tmp_path,
    )
    for src_param, dst_param in zip(model.parameters(), dst_model.parameters()):
        assert int(dst_opt.state[dst_param]["step"]) == snapshot_steps[src_param]
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                dst_opt._host_states[dst_param][key], snapshot[src_param][key], atol=0.0, rtol=0.0
            )
            # The live buffers moved past the snapshot, so the match above is meaningful.
            assert not torch.equal(snapshot[src_param][key], opt._host_states[src_param][key])
