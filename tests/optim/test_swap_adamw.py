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

The optimizer is exercised over one-rank ``Shard(0)`` DTensors (the layout FSDP2
hands it) and compared step-by-step against ``torch.optim.AdamW`` running on the
plain local tensors. This covers the swap/offload mechanics and the update math
without needing an accelerator; the device-stream path is only taken on NPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from veomni.optim.swap_adamw import SwapAdamW


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


def test_multi_optimizer_propagates_swap_flag(cpu_mesh):
    from veomni.optim.optimizer import MultiOptimizer

    torch.manual_seed(5)
    init_a = torch.randn(4, 2)
    init_b = torch.randn(3, 2)
    par_a = _swap_param(init_a, cpu_mesh)
    par_b = _swap_param(init_b, cpu_mesh)

    opt_a = SwapAdamW([{"params": [par_a], "weight_decay": 0.1}], lr=1e-2, pin_memory=False)
    opt_b = SwapAdamW([{"params": [par_b], "weight_decay": 0.0}], lr=1e-2, pin_memory=False)
    multi = MultiOptimizer(nn.Module(), {"ep": opt_a, "non_extra_parallel": opt_b}, ["ep", "non_extra_parallel"])

    assert multi._is_swap_optimizer is True

    par_a.grad = _partition(torch.randn(4, 2), cpu_mesh)
    par_b.grad = _partition(torch.randn(3, 2), cpu_mesh)
    multi.step()
    multi.zero_grad()

    for opt, param in ((opt_a, par_a), (opt_b, par_b)):
        for key in ("exp_avg", "exp_avg_sq"):
            assert opt._device_states[param][key].untyped_storage().nbytes() == 0


def test_checkpoint_guard_rejects_swap_optimizer(cpu_mesh):
    from veomni.checkpoint.dcp_checkpointer import OptimizerState

    init = torch.randn(2, 2)
    param = _swap_param(init, cpu_mesh)
    opt = SwapAdamW([param], pin_memory=False)

    with pytest.raises(RuntimeError, match="does not support checkpoint"):
        OptimizerState._reject_swap_optimizer(opt)
