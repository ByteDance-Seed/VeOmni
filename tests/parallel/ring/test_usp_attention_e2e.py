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

"""End-to-end USP (Ulysses x Ring) equivalence test through the VeOmni
``flash_attention_forward`` op.

Builds a ``[ulysses, cp]`` device mesh + ParallelState, shards Q/K/V with the
same layout the ``SequenceParallelCollator`` uses (cp-outer zig-zag,
ulysses-inner contiguous), runs the real attention op (Ulysses all-to-all +
ring), and checks the gathered output matches single-device full causal
FlashAttention. Requires 4 GPUs and flash-attn (FA2).
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
from veomni.distributed.sequence_parallel.data import zigzag_reorder, zigzag_reorder_varlen

from ._ref import ATTN_IMPL_WITH_SP, ref_attn_func, ref_attn_varlen_func
from ._ref import FA_OK as _FA_OK


ULYSSES = 2
CP = 2


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
