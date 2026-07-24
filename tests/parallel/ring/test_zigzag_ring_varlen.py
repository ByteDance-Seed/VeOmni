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

"""Numerical equivalence test for the packed (varlen) zig-zag ring attention.

Shards several packed documents with per-document zig-zag reordering (the USP
varlen layout) across the ``cp`` group, runs ``zigzag_ring_flash_attn_varlen_func``,
and checks output + gradients match single-device ``flash_attn_varlen_func``.
Runs for world sizes 2 and 4 (needs that many GPUs). Requires flash-attn (FA2).
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
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

from veomni.distributed.sequence_parallel.data import local_cu_seqlens, zigzag_reorder_varlen


try:
    from flash_attn import flash_attn_varlen_func

    _FA_OK = True
except ImportError:
    _FA_OK = False


class _ZigzagRingVarlenTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _init(self):
        store = dist.FileStore(self.file_name, self.world_size)
        get_torch_device().set_device(self.rank)
        c10d.init_process_group(get_dist_comm_backend(), store=store, rank=self.rank, world_size=self.world_size)

    def _run_case(self):
        from veomni.distributed.sequence_parallel.ring_attention import zigzag_ring_flash_attn_varlen_func

        self._init()
        world = self.world_size
        dev = torch.device(f"{get_device_type()}:{self.rank}")
        h, d = 4, 64
        # documents whose lengths are divisible by 2*world (balanced ring req.)
        doc_lens = [2 * world * 7, 2 * world * 3, 2 * world * 5]
        total = sum(doc_lens)
        cu = torch.tensor(
            [0, *[sum(doc_lens[: i + 1]) for i in range(len(doc_lens))]],
            device=dev,
            dtype=torch.int32,
        )
        max_seqlen = max(doc_lens)
        scale = d**-0.5

        torch.manual_seed(0)
        q = torch.randn(total, h, d, device=dev, dtype=torch.bfloat16)
        k = torch.randn(total, h, d, device=dev, dtype=torch.bfloat16)
        v = torch.randn(total, h, d, device=dev, dtype=torch.bfloat16)
        g = torch.randn(total, h, d, device=dev, dtype=torch.bfloat16)
        for t in (q, k, v, g):
            dist.broadcast(t, 0)

        # single-device reference (full packed varlen causal attention)
        qf = q.clone().requires_grad_(True)
        kf = k.clone().requires_grad_(True)
        vf = v.clone().requires_grad_(True)
        ref = flash_attn_varlen_func(qf, kf, vf, cu, cu, max_seqlen, max_seqlen, softmax_scale=scale, causal=True)
        ref.backward(g)

        def shard(t):
            tr = zigzag_reorder_varlen(t.detach(), cu, dim=0, cp_size=world)
            chunk = tr.shape[0] // world
            return tr[self.rank * chunk : (self.rank + 1) * chunk].clone()

        lcu = local_cu_seqlens(cu, world)
        seqlens = lcu[1:] - lcu[:-1]
        local_max = int(seqlens.max().item())

        qz = shard(q).requires_grad_(True)
        kz = shard(k).requires_grad_(True)
        vz = shard(v).requires_grad_(True)
        gz = shard(g)

        out = zigzag_ring_flash_attn_varlen_func(
            qz, kz, vz, lcu, local_max, softmax_scale=scale, causal=True, group=dist.group.WORLD
        )
        out.backward(gz)

        torch.testing.assert_close(out, shard(ref), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(qz.grad, shard(qf.grad), atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(kz.grad, shard(kf.grad), atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(vz.grad, shard(vf.grad), atol=3e-2, rtol=3e-2)


class ZigzagRingVarlenWorld2Test(_ZigzagRingVarlenTest):
    @property
    def world_size(self):
        return 2

    @pytest.mark.skipif(not _FA_OK, reason="flash-attn (FA2) required")
    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    def test_varlen_ring_matches_full(self):
        self._run_case()


class ZigzagRingVarlenWorld4Test(_ZigzagRingVarlenTest):
    @property
    def world_size(self):
        return 4

    @pytest.mark.skipif(not _FA_OK, reason="flash-attn (FA2) required")
    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_varlen_ring_matches_full(self):
        self._run_case()


if __name__ == "__main__":
    run_tests()
