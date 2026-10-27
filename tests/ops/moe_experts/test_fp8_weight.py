# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CPU regression coverage for the opaque FSDP expert-weight container."""

import pytest
import torch

from veomni.distributed.offloading import OffloadPolicy, custom_save_on_cpu
from veomni.ops.kernels.moe_experts.shared.fp8_weight import FP8ExpertWeight


@pytest.mark.parametrize("pin_memory", [False, True])
def test_activation_offload_leaves_fsdp_fp8_storage_owned_by_fsdp(pin_memory):
    # Meta storage makes any accidental device copy fail even on CPU-only CI.
    weight = FP8ExpertWeight(
        torch.empty(2, 128, 128, device="meta", dtype=torch.float8_e4m3fn),
        torch.empty(2, 1, 1, device="meta"),
    )
    offload = custom_save_on_cpu(pin_memory=pin_memory)
    packed = offload.pack_hook(weight)
    assert packed[0] is OffloadPolicy.IGNORE
    assert offload.unpack_hook(packed) is weight
    assert offload.cur_gpu_ram_in_mb == 0


def test_activation_offload_still_packs_ordinary_tensors():
    tensor = torch.randn(32, 32)
    offload = custom_save_on_cpu()
    packed = offload.pack_hook(tensor)
    assert packed[0] is OffloadPolicy.OFFLOAD
    torch.testing.assert_close(offload.unpack_hook(packed), tensor)
