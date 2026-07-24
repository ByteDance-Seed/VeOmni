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

"""Distributed numerical-equivalence tests for USP zig-zag ring attention.

The zig-zag ring attention output (and its Q/K/V gradients) must match a
single-device full causal FlashAttention over the same sequence, after applying
the zig-zag block reordering. Requires >= 2 GPUs and flash-attn (FA2).
"""

import sys

import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


if not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import pytest
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests

from veomni.distributed.sequence_parallel.data import zigzag_reorder
from veomni.distributed.sequence_parallel.ring_attention import zigzag_ring_flash_attn_func

from ..ulysses.utils import SequenceParallelTest
from ._ref import FA_OK as _FA_OK
from ._ref import ref_attn_func


class ZigzagRingAttentionTest(SequenceParallelTest):
    def _shard(self, full, group):
        """Take this rank's zig-zag contiguous chunk (dim=1) of ``full``."""
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        reordered = zigzag_reorder(full.detach(), dim=1, cp_size=world)
        chunk = reordered.shape[1] // world
        return reordered[:, rank * chunk : (rank + 1) * chunk].clone()

    @pytest.mark.skipif(not _FA_OK, reason="a flash-attn backend (FA2 or FA4) is required")
    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    def test_matches_full_attention(self):
        group = self._get_process_group()
        world = dist.get_world_size(group)
        dev = get_device_type()

        b, h, d = 1, 8, 64
        seq = 2 * world * 64
        scale = d**-0.5

        q = torch.randn(b, seq, h, d, device=dev, dtype=torch.bfloat16)
        k = torch.randn(b, seq, h, d, device=dev, dtype=torch.bfloat16)
        v = torch.randn(b, seq, h, d, device=dev, dtype=torch.bfloat16)
        dist.broadcast(q, 0)
        dist.broadcast(k, 0)
        dist.broadcast(v, 0)
        g = torch.randn(b, seq, h, d, device=dev, dtype=torch.bfloat16)
        dist.broadcast(g, 0)

        # reference: single-device full causal attention
        qf = q.clone().requires_grad_(True)
        kf = k.clone().requires_grad_(True)
        vf = v.clone().requires_grad_(True)
        ref = ref_attn_func(qf, kf, vf, scale, causal=True)
        ref.backward(g)

        # zig-zag ring: shard, run, compare against the matching reference chunk
        qz = self._shard(q, group).requires_grad_(True)
        kz = self._shard(k, group).requires_grad_(True)
        vz = self._shard(v, group).requires_grad_(True)
        gz = self._shard(g, group)

        out = zigzag_ring_flash_attn_func(qz, kz, vz, softmax_scale=scale, causal=True, group=group)
        out.backward(gz)

        ref_out = self._shard(ref, group)
        ref_dq = self._shard(qf.grad, group)
        ref_dk = self._shard(kf.grad, group)
        ref_dv = self._shard(vf.grad, group)

        torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(qz.grad, ref_dq, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(kz.grad, ref_dk, atol=3e-2, rtol=2e-2)
        torch.testing.assert_close(vz.grad, ref_dv, atol=3e-2, rtol=2e-2)


if __name__ == "__main__":
    run_tests()
