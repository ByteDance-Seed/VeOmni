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

"""GPU (multi-process) tests for the USP zig-zag ring attention path.

Consolidates the distributed ring-attention tests, all sharing the same
single-device reference backend (FA2 on Ampere/Hopper or FA4 on Blackwell,
auto-selected via ``._ref``):

  * ``ZigzagRingAttentionTest`` — dense zig-zag ring vs full causal attention
    (world >= 2).
  * ``ZigzagRingVarlenWorld{2,4}Test`` — packed/varlen zig-zag ring vs full
    packed causal attention (world 2 and 4).
  * ``USPAttentionE2ETest`` — the real attention op (Ulysses all-to-all + ring)
    for both dense and varlen, sharded cp-outer / ulysses-inner (world 4).

Each test skips unless enough GPUs and a flash-attn backend are available.
"""

import os
import sys

import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


if not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import pytest
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

from veomni.distributed import parallel_state as PS
from veomni.distributed.sequence_parallel.data import (
    local_cu_seqlens,
    zigzag_reorder,
    zigzag_reorder_varlen,
)
from veomni.distributed.sequence_parallel.ring_attention import (
    zigzag_ring_flash_attn_func,
    zigzag_ring_flash_attn_varlen_func,
)

from ..ulysses.utils import SequenceParallelTest
from ._ref import ATTN_IMPL_WITH_SP, ref_attn_func, ref_attn_varlen_func
from ._ref import FA_OK as _FA_OK


ULYSSES = 2
CP = 2


# ── dense zig-zag ring attention ──────────────────────────────────────────────
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


# ── packed / varlen zig-zag ring attention ────────────────────────────
class _ZigzagRingVarlenTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _init(self):
        store = dist.FileStore(self.file_name, self.world_size)
        get_torch_device().set_device(self.rank)
        c10d.init_process_group(get_dist_comm_backend(), store=store, rank=self.rank, world_size=self.world_size)

    def _run_case(self):
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
        ref = ref_attn_varlen_func(qf, kf, vf, cu, max_seqlen, scale, causal=True)
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

    @pytest.mark.skipif(not _FA_OK, reason="a flash-attn backend (FA2 or FA4) is required")
    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    def test_varlen_ring_matches_full(self):
        self._run_case()


class ZigzagRingVarlenWorld4Test(_ZigzagRingVarlenTest):
    @property
    def world_size(self):
        return 4

    @pytest.mark.skipif(not _FA_OK, reason="a flash-attn backend (FA2 or FA4) is required")
    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_varlen_ring_matches_full(self):
        self._run_case()


# ── op-level USP end-to-end (Ulysses all-to-all + ring) ───────────────
class _AttnModule(torch.nn.Module):
    """Minimal module exposing the attributes flash_attention_forward reads."""

    def __init__(self):
        super().__init__()
        self.is_causal = True

        class _Cfg:
            _attn_implementation = ATTN_IMPL_WITH_SP

        self.config = _Cfg()
        self.layer_idx = 0


class USPAttentionE2ETest(MultiProcessTestCase):
    @property
    def world_size(self):
        return ULYSSES * CP

    def setUp(self):
        super().setUp()
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def _init(self):
        store = dist.FileStore(self.file_name, self.world_size)
        get_torch_device().set_device(self.rank)
        c10d.init_process_group(get_dist_comm_backend(), store=store, rank=self.rank, world_size=self.world_size)
        mesh = init_device_mesh(get_device_type(), (ULYSSES, CP), mesh_dim_names=("ulysses", "cp"))
        mesh[("ulysses", "cp")]._flatten(mesh_dim_name="sp")
        state = PS.ParallelState(
            dp_size=1,
            dp_replicate_size=1,
            dp_shard_size=1,
            tp_size=1,
            pp_size=1,
            cp_size=CP,
            ulysses_size=ULYSSES,
            device_type=get_device_type(),
            device_mesh=mesh,
        )
        PS._PARALLEL_STATE = state
        return state, mesh

    def _usp_slice(self, full, mesh):
        """cp-outer zig-zag + ulysses-inner contiguous slice (matches collator)."""
        uly_rank = mesh.get_local_rank("ulysses")
        cp_rank = mesh.get_local_rank("cp")
        reordered = zigzag_reorder(full.detach(), dim=1, cp_size=CP)
        cp_chunk = reordered.shape[1] // CP
        cp_region = reordered[:, cp_rank * cp_chunk : (cp_rank + 1) * cp_chunk]
        uly_chunk = cp_region.shape[1] // ULYSSES
        return cp_region[:, uly_rank * uly_chunk : (uly_rank + 1) * uly_chunk].clone()

    @pytest.mark.skipif(not _FA_OK, reason="a flash-attn backend (FA2 or FA4) is required")
    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_usp_matches_full_attention(self):
        from veomni.ops.kernels.attention import flash_attention_forward

        state, mesh = self._init()
        dev = get_device_type()
        b, h, d = 1, 8, 64
        seq = 2 * self.world_size * 64
        scale = d**-0.5

        # full (b, h, s, d) inputs (op transposes internally)
        q = torch.randn(b, h, seq, d, device=dev, dtype=torch.bfloat16)
        k = torch.randn(b, h, seq, d, device=dev, dtype=torch.bfloat16)
        v = torch.randn(b, h, seq, d, device=dev, dtype=torch.bfloat16)
        for t in (q, k, v):
            dist.broadcast(t, 0)

        # reference: single-device full causal attention (b, s, h, d layout)
        ref = ref_attn_func(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            scale,
            causal=True,
        )  # (b, s, h, d)

        # local shards in (b, h, s_local, d) as the model produces pre-op
        def shard_bhsd(full_bhsd):
            full_bshd = full_bhsd.transpose(1, 2).contiguous()
            local_bshd = self._usp_slice(full_bshd, mesh)
            return local_bshd.transpose(1, 2).contiguous()

        lq = shard_bhsd(q)
        lk = shard_bhsd(k)
        lv = shard_bhsd(v)

        module = _AttnModule().to(dev)
        out, _ = flash_attention_forward(module, lq, lk, lv, attention_mask=None, scaling=scale)
        # out is (b, s_local, h, d)
        ref_local = self._usp_slice(ref, mesh)
        torch.testing.assert_close(out, ref_local, atol=2e-2, rtol=2e-2)

    def _usp_slice_varlen(self, full, cu, mesh):
        """Per-document zig-zag cp-outer + ulysses-inner slice (varlen layout)."""
        uly_rank = mesh.get_local_rank("ulysses")
        cp_rank = mesh.get_local_rank("cp")
        reordered = zigzag_reorder_varlen(full.detach(), cu, dim=1, cp_size=CP)
        cp_chunk = reordered.shape[1] // CP
        cp_region = reordered[:, cp_rank * cp_chunk : (cp_rank + 1) * cp_chunk]
        uly_chunk = cp_region.shape[1] // ULYSSES
        return cp_region[:, uly_rank * uly_chunk : (uly_rank + 1) * uly_chunk].clone()

    @pytest.mark.skipif(not _FA_OK, reason="a flash-attn backend (FA2 or FA4) is required")
    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_usp_varlen_matches_full(self):
        from veomni.ops.kernels.attention import flash_attention_forward

        state, mesh = self._init()
        dev = get_device_type()
        b, h, d = 1, 8, 64
        # packed documents; each length divisible by 2 * world_size
        doc_lens = [2 * self.world_size * 12, 2 * self.world_size * 8, 2 * self.world_size * 4]
        seq = sum(doc_lens)
        scale = d**-0.5
        cu = torch.tensor(
            [0, *[sum(doc_lens[: i + 1]) for i in range(len(doc_lens))]],
            device=dev,
            dtype=torch.int32,
        )
        max_seqlen = max(doc_lens)

        q = torch.randn(b, h, seq, d, device=dev, dtype=torch.bfloat16)
        k = torch.randn(b, h, seq, d, device=dev, dtype=torch.bfloat16)
        v = torch.randn(b, h, seq, d, device=dev, dtype=torch.bfloat16)
        for t in (q, k, v):
            dist.broadcast(t, 0)

        # reference: single-device full packed varlen causal attention (b, s, h, d)
        ref = ref_attn_varlen_func(
            q.transpose(1, 2).reshape(seq, h, d).contiguous(),
            k.transpose(1, 2).reshape(seq, h, d).contiguous(),
            v.transpose(1, 2).reshape(seq, h, d).contiguous(),
            cu,
            max_seqlen,
            scale,
            causal=True,
        ).reshape(b, seq, h, d)

        def shard_bhsd(full_bhsd):
            full_bshd = full_bhsd.transpose(1, 2).contiguous()
            local_bshd = self._usp_slice_varlen(full_bshd, cu, mesh)
            return local_bshd.transpose(1, 2).contiguous()

        lq = shard_bhsd(q)
        lk = shard_bhsd(k)
        lv = shard_bhsd(v)

        module = _AttnModule().to(dev)
        # The op derives LOCAL cu_seqlens from the GLOBAL cu_seq_lens_q kwarg.
        out, _ = flash_attention_forward(
            module, lq, lk, lv, attention_mask=None, scaling=scale, cu_seq_lens_q=cu, cu_seq_lens_k=cu
        )
        ref_local = self._usp_slice_varlen(ref, cu, mesh)
        torch.testing.assert_close(out, ref_local, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    run_tests()
