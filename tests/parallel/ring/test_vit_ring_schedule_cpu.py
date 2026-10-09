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

"""CPU (gloo) schedule test for the NPU bidirectional (ViT) ring attention.

The ``torch_npu`` fusion-attention calls are replaced with a float64 torch
reference that honours TND ``actual_seq_*len`` segments, including zero-length
ones, so the ring schedule (per-rank offsets, softmax-stat merge, dK/dV return
trip) runs and is checked on any machine against full bidirectional varlen
attention. Numerical coverage on real hardware lives in ``test_usp_ring_npu.py``.
"""

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

import veomni.distributed.sequence_parallel.ring_attention.npu as ring_npu
from veomni.distributed.sequence_parallel.ring_attention.layout import vit_cp_cu_seqlens, vit_usp_slice


_HEADS = 2
_HEAD_DIM = 8
# vit seq lengths (multiples of the merger unit 4); several seqs have fewer merger
# units than cp ranks so some ranks hold empty shards in the middle of their cu.
_CASES = [
    [20, 4, 28, 8],
    [4, 36, 12],
    [368, 8],
    [16],
]


def _segments(cu):
    return list(zip(cu[:-1], cu[1:]))


def _fake_npu_fa_forward(q, k, v, *, softmax_scale, actual_seq_qlen, actual_seq_kvlen, **_):
    out = torch.zeros_like(q)
    smax = torch.full(q.shape[:2], float("-inf"), dtype=q.dtype)
    ssum = torch.zeros(q.shape[:2], dtype=q.dtype)
    for (q0, q1), (k0, k1) in zip(_segments(actual_seq_qlen), _segments(actual_seq_kvlen)):
        if q1 == q0 or k1 == k0:
            continue  # rows with no key: max=-inf, sum=0, out=0
        scores = torch.einsum("qnd,knd->qnk", q[q0:q1], k[k0:k1]) * softmax_scale
        row_max = scores.amax(-1)
        probs = torch.exp(scores - row_max.unsqueeze(-1))
        row_sum = probs.sum(-1)
        out[q0:q1] = torch.einsum("qnk,knd->qnd", probs / row_sum.unsqueeze(-1), v[k0:k1])
        smax[q0:q1], ssum[q0:q1] = row_max, row_sum
    stats = lambda x: x.unsqueeze(-1).expand(*x.shape, 8).contiguous()  # noqa: E731
    return out, stats(smax), stats(ssum), (0, 0, 0)


def _fake_npu_fa_backward(
    dout, q, k, v, out, softmax_max, softmax_sum, *, softmax_scale, actual_seq_qlen, actual_seq_kvlen, **_
):
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    delta = (dout * out).sum(-1)  # (T, N)
    for (q0, q1), (k0, k1) in zip(_segments(actual_seq_qlen), _segments(actual_seq_kvlen)):
        if q1 == q0 or k1 == k0:
            continue
        scores = torch.einsum("qnd,knd->qnk", q[q0:q1], k[k0:k1]) * softmax_scale
        probs = torch.exp(scores - softmax_max[q0:q1].unsqueeze(-1)) / softmax_sum[q0:q1].unsqueeze(-1)
        dv[k0:k1] += torch.einsum("qnk,qnd->knd", probs, dout[q0:q1])
        dprobs = torch.einsum("qnd,knd->qnk", dout[q0:q1], v[k0:k1])
        dscores = probs * (dprobs - delta[q0:q1].unsqueeze(-1))
        dq[q0:q1] += torch.einsum("qnk,knd->qnd", dscores, k[k0:k1]) * softmax_scale
        dk[k0:k1] += torch.einsum("qnk,qnd->knd", dscores, q[q0:q1]) * softmax_scale
    return dq, dk, dv


def _reference(q, k, v, vit_seqlens, scale):
    outs, start = [], 0
    for seqlen in vit_seqlens:
        sl = slice(start, start + seqlen)
        scores = torch.einsum("qnd,knd->nqk", q[sl], k[sl]) * scale
        outs.append(torch.einsum("nqk,knd->qnd", torch.softmax(scores, dim=-1), v[sl]))
        start += seqlen
    return torch.cat(outs)


class _VitRingScheduleTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _run_all_cases(self):
        dist.init_process_group(
            "gloo", store=dist.FileStore(self.file_name, self.world_size), rank=self.rank, world_size=self.world_size
        )
        ring_npu._require_torch_npu = lambda: None
        ring_npu._npu_fa_forward = _fake_npu_fa_forward
        ring_npu._npu_fa_backward = _fake_npu_fa_backward
        for vit_seqlens in _CASES:
            self._check(vit_seqlens)
        dist.destroy_process_group()

    def _check(self, vit_seqlens):
        world, rank = self.world_size, self.rank
        total = sum(vit_seqlens)
        scale = _HEAD_DIM**-0.5
        generator = torch.Generator().manual_seed(total)
        q, k, v, dout = (
            torch.randn(total, _HEADS, _HEAD_DIM, generator=generator, dtype=torch.float64) for _ in range(4)
        )

        q_ref, k_ref, v_ref = (t.clone().requires_grad_(True) for t in (q, k, v))
        ref = _reference(q_ref, k_ref, v_ref, vit_seqlens, scale)
        ref.backward(dout)

        def shard(t, pad_value=0):
            return vit_usp_slice(t, vit_seqlens, 0, world, rank, 1, 0, pad_value=pad_value)

        global_cu = [0]
        for seqlen in vit_seqlens:
            global_cu.append(global_cu[-1] + seqlen)
        cu_per_rank = [vit_cp_cu_seqlens(global_cu, world, c, 1) for c in range(world)]

        local_q, local_k, local_v = (shard(t).requires_grad_(True) for t in (q, k, v))
        out = ring_npu.ring_npu_flash_attn_varlen_func(
            local_q, local_k, local_v, cu_per_rank, softmax_scale=scale, group=dist.group.WORLD
        )
        out.backward(shard(dout))

        rows = shard(torch.arange(total), pad_value=-1)
        real = rows >= 0
        tolerance = {"atol": 1e-6, "rtol": 1e-6}  # partial results merge in fp32 by design
        torch.testing.assert_close(out[real], ref.detach()[rows[real]], **tolerance)
        torch.testing.assert_close(local_q.grad[real], q_ref.grad[rows[real]], **tolerance)
        torch.testing.assert_close(local_k.grad[real], k_ref.grad[rows[real]], **tolerance)
        torch.testing.assert_close(local_v.grad[real], v_ref.grad[rows[real]], **tolerance)


class VitRingScheduleWorld2Test(_VitRingScheduleTest):
    @property
    def world_size(self):
        return 2

    def test_bidirectional_ring_matches_full_attention(self):
        self._run_all_cases()


class VitRingScheduleWorld3Test(_VitRingScheduleTest):
    @property
    def world_size(self):
        return 3

    def test_bidirectional_ring_matches_full_attention(self):
        self._run_all_cases()


class VitRingScheduleWorld4Test(_VitRingScheduleTest):
    @property
    def world_size(self):
        return 4

    def test_bidirectional_ring_matches_full_attention(self):
        self._run_all_cases()


if __name__ == "__main__":
    run_tests()
