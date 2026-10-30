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

"""Device-path companion to ``test_swap_adamw.py``.

The CPU file covers the update math, the swap mechanics and the checkpoint
plumbing. This file covers on a real accelerator:

* the step path: copy streams, a budget sized from live free device memory,
  the fused ``torch._fused_adamw_`` kernel and pinned host buffers;
* the checkpoint path: DTensor parameters on a one-rank device mesh, so
  ``state_dict`` exports through ``rewrap_dtensor_local``'s mismatched-device
  branch (a CPU local on an accelerator mesh) and round-trips through
  ``dcp.save``/``dcp.load``.

Multi-rank FSDP2/EP topologies are validated manually (Ascend NPU + FSDP2 +
EP8 on a 30B MoE), not by this file.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from veomni.optim.swap_adamw import SwapAdamW
from veomni.utils.device import FUSED_ADAMW_DEVICE_TYPES, STREAM_DEVICE_TYPES, get_device_type


pytestmark = pytest.mark.skipif(get_device_type() == "cpu", reason="accelerator backend required")


def test_matches_torch_adamw_step_by_step_on_accelerator():
    torch.manual_seed(21)
    device_type = get_device_type()
    shape = (6, 4)
    init = torch.randn(shape, device=device_type)

    ref = nn.Parameter(init.clone())
    swapped = nn.Parameter(init.clone())

    ref_opt = torch.optim.AdamW([ref], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False)
    swap_opt = SwapAdamW([swapped], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    # The module-level skip guarantees an accelerator backend, but only the
    # backends in ``STREAM_DEVICE_TYPES`` take the stream path (others, e.g.
    # MLU, copy synchronously); the fused kernel likewise runs only on the
    # backends that register it.
    assert swap_opt._use_streams == (device_type in STREAM_DEVICE_TYPES)
    assert swap_opt._use_fused == (device_type in FUSED_ADAMW_DEVICE_TYPES)

    for _ in range(5):
        grad = torch.randn(shape, device=device_type)
        ref.grad = grad.clone()
        swapped.grad = grad.clone()
        ref_opt.step()
        swap_opt.step()

        torch.testing.assert_close(swapped.detach(), ref.detach(), atol=1e-5, rtol=1e-5)

    for key in ("exp_avg", "exp_avg_sq"):
        # The swap-out released the device storage, and the host buffer holds
        # the same moments stock keeps on device.
        assert swap_opt._device_states[swapped][key].untyped_storage().nbytes() == 0
        torch.testing.assert_close(
            swap_opt._host_states[swapped][key], ref_opt.state[ref][key].cpu(), atol=1e-5, rtol=1e-5
        )


def test_dcp_roundtrip_on_device_mesh(tmp_path):
    """The production checkpoint path: DTensor parameters on a device mesh, so
    ``state_dict`` exports through ``rewrap_dtensor_local``'s mismatched-device
    branch (a CPU local on an accelerator mesh), and the state round-trips
    through DCP into a fresh optimizer bitwise-equal."""
    from veomni.checkpoint.dcp_checkpointer import OptimizerState

    device_type = get_device_type()
    owns_pg = not dist.is_initialized()
    if owns_pg:
        store = dist.FileStore(str(tmp_path / "store"), 1)
        dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    try:
        mesh = init_device_mesh(device_type, (1,))

        class _Tiny(nn.Module):
            def __init__(self, seed: int):
                super().__init__()
                torch.manual_seed(seed)
                self.weight = nn.Parameter(
                    DTensor.from_local(torch.randn(6, 4, device=device_type), mesh, [Shard(0)], run_check=False)
                )

        src_model = _Tiny(seed=31)
        src_opt = SwapAdamW(list(src_model.parameters()), lr=1e-2)
        for _ in range(3):
            src_model.weight.grad = DTensor.from_local(
                torch.randn(6, 4, device=device_type), mesh, [Shard(0)], run_check=False
            )
            src_opt.step()

        parallel_state = SimpleNamespace(dp_mode="fsdp2")
        dcp.save(
            {"optimizer": OptimizerState(src_model, src_opt, parallel_state=parallel_state)},
            checkpoint_id=tmp_path,
        )

        # The export must be host-resident: after the save, the state holder's
        # local IS the host buffer (the mismatched-device rewrap branch).
        param = src_model.weight
        holder = src_opt.state[param]["exp_avg"]
        assert holder.to_local().device.type == "cpu"
        assert holder.to_local().data_ptr() == src_opt._host_states[param]["exp_avg"].data_ptr()

        dst_model = _Tiny(seed=99)
        dst_opt = SwapAdamW(list(dst_model.parameters()), lr=1e-2)
        dcp.load(
            {"optimizer": OptimizerState(dst_model, dst_opt, parallel_state=parallel_state, load=True)},
            checkpoint_id=tmp_path,
        )

        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                dst_opt._host_states[dst_model.weight][key],
                src_opt._host_states[param][key],
                atol=0.0,
                rtol=0.0,
            )
        assert int(dst_opt.state[dst_model.weight]["step"]) == 3
    finally:
        if owns_pg and dist.is_initialized():
            dist.destroy_process_group()
