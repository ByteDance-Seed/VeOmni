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

"""Ascend numerical tests for NPU zig-zag Ring Attention and USP routing."""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

from veomni.distributed import parallel_state as PS
from veomni.distributed.sequence_parallel.ring_attention.layout import (
    local_cu_seqlens,
    zigzag_reorder,
    zigzag_reorder_packed,
)
from veomni.distributed.sequence_parallel.ring_attention.npu import (
    zigzag_ring_npu_flash_attn_func,
    zigzag_ring_npu_flash_attn_varlen_func,
)
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.import_utils import is_torch_npu_available


pytestmark = pytest.mark.skipif(not is_torch_npu_available(), reason="torch_npu is required")

_ATOL = 8e-2
_RTOL = 6e-2


def _dense_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    dtype = q.dtype
    q = q.float().transpose(1, 2)
    k = k.float().transpose(1, 2)
    v = v.float().transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    seq = scores.shape[-1]
    mask = torch.triu(torch.ones((seq, seq), dtype=torch.bool, device=scores.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).to(dtype).contiguous()


def _varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    outputs = []
    cu = [int(value) for value in cu_seqlens.tolist()]
    for start, end in zip(cu[:-1], cu[1:]):
        outputs.append(
            _dense_reference(
                q[start:end].unsqueeze(0),
                k[start:end].unsqueeze(0),
                v[start:end].unsqueeze(0),
                scale,
            ).squeeze(0)
        )
    return torch.cat(outputs, dim=0)


class NPUZigzagRingTest(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _init(self):
        store = dist.FileStore(self.file_name, self.world_size)
        get_torch_device().set_device(self.rank)
        dist.init_process_group(get_dist_comm_backend(), store=store, rank=self.rank, world_size=self.world_size)
        return dist.distributed_c10d._get_default_group()

    def _shard_dense(self, full: torch.Tensor) -> torch.Tensor:
        reordered = zigzag_reorder(full.detach(), dim=1, cp_size=self.world_size)
        chunk = reordered.shape[1] // self.world_size
        return reordered[:, self.rank * chunk : (self.rank + 1) * chunk].clone()

    def _shard_varlen(self, full: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        reordered = zigzag_reorder_packed(full.detach(), cu_seqlens, dim=0, cp_size=self.world_size)
        chunk = reordered.shape[0] // self.world_size
        return reordered[self.rank * chunk : (self.rank + 1) * chunk].clone()

    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    def test_dense_forward_backward_matches_full_attention(self):
        group = self._init()
        device = get_device_type()
        batch, seq, heads, dim = 1, 128, 4, 128
        scale = dim**-0.5
        q = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        grad = torch.randn_like(q)
        for tensor in (q, k, v, grad):
            dist.broadcast(tensor, 0)

        q_ref = q.clone().requires_grad_(True)
        k_ref = k.clone().requires_grad_(True)
        v_ref = v.clone().requires_grad_(True)
        out_ref = _dense_reference(q_ref, k_ref, v_ref, scale)
        out_ref.backward(grad)

        q_local = self._shard_dense(q).requires_grad_(True)
        k_local = self._shard_dense(k).requires_grad_(True)
        v_local = self._shard_dense(v).requires_grad_(True)
        grad_local = self._shard_dense(grad)
        out = zigzag_ring_npu_flash_attn_func(q_local, k_local, v_local, softmax_scale=scale, causal=True, group=group)
        out.backward(grad_local)

        torch.testing.assert_close(out, self._shard_dense(out_ref), atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(q_local.grad, self._shard_dense(q_ref.grad), atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(k_local.grad, self._shard_dense(k_ref.grad), atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(v_local.grad, self._shard_dense(v_ref.grad), atol=_ATOL, rtol=_RTOL)

    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    def test_varlen_forward_backward_matches_full_attention(self):
        group = self._init()
        device = get_device_type()
        doc_lens = [32, 64, 128]
        total, heads, dim = sum(doc_lens), 4, 128
        scale = dim**-0.5
        cu = torch.tensor([0, 32, 96, total], dtype=torch.long, device=device)
        q = torch.randn(total, heads, dim, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        grad = torch.randn_like(q)
        for tensor in (q, k, v, grad):
            dist.broadcast(tensor, 0)

        q_ref = q.clone().requires_grad_(True)
        k_ref = k.clone().requires_grad_(True)
        v_ref = v.clone().requires_grad_(True)
        out_ref = _varlen_reference(q_ref, k_ref, v_ref, cu, scale)
        out_ref.backward(grad)

        q_local = self._shard_varlen(q, cu).requires_grad_(True)
        k_local = self._shard_varlen(k, cu).requires_grad_(True)
        v_local = self._shard_varlen(v, cu).requires_grad_(True)
        grad_local = self._shard_varlen(grad, cu)
        local_cu = local_cu_seqlens(cu, self.world_size)
        out = zigzag_ring_npu_flash_attn_varlen_func(
            q_local,
            k_local,
            v_local,
            local_cu,
            max(doc_lens) // self.world_size,
            softmax_scale=scale,
            causal=True,
            group=group,
        )
        out.backward(grad_local)

        torch.testing.assert_close(out, self._shard_varlen(out_ref, cu), atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(q_local.grad, self._shard_varlen(q_ref.grad, cu), atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(k_local.grad, self._shard_varlen(k_ref.grad, cu), atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(v_local.grad, self._shard_varlen(v_ref.grad, cu), atol=_ATOL, rtol=_RTOL)


class _AttentionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_causal = True

        class _Config:
            _attn_implementation = "veomni_flash_attention_2_with_sp"

        self.config = _Config()
        self.layer_idx = 0


class NPUUSPAttentionTest(MultiProcessTestCase):
    ulysses_size = 2
    cp_size = 2

    @property
    def world_size(self):
        return self.ulysses_size * self.cp_size

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _init(self):
        store = dist.FileStore(self.file_name, self.world_size)
        get_torch_device().set_device(self.rank)
        dist.init_process_group(get_dist_comm_backend(), store=store, rank=self.rank, world_size=self.world_size)
        mesh = init_device_mesh(
            get_device_type(),
            (self.ulysses_size, self.cp_size),
            mesh_dim_names=("ulysses", "cp"),
        )
        mesh[("ulysses", "cp")]._flatten(mesh_dim_name="sp")
        PS._PARALLEL_STATE = PS.ParallelState(
            dp_size=1,
            dp_replicate_size=1,
            dp_shard_size=1,
            cp_size=self.cp_size,
            ulysses_size=self.ulysses_size,
            device_type=get_device_type(),
            device_mesh=mesh,
        )
        return mesh

    def _usp_slice(self, full: torch.Tensor, mesh, cu_seqlens: torch.Tensor = None) -> torch.Tensor:
        if cu_seqlens is None:
            reordered = zigzag_reorder(full.detach(), dim=1, cp_size=self.cp_size)
        else:
            reordered = zigzag_reorder_packed(full.detach(), cu_seqlens, dim=1, cp_size=self.cp_size)
        cp_rank = mesh.get_local_rank("cp")
        ulysses_rank = mesh.get_local_rank("ulysses")
        cp_chunk = reordered.shape[1] // self.cp_size
        cp_region = reordered[:, cp_rank * cp_chunk : (cp_rank + 1) * cp_chunk]
        ulysses_chunk = cp_region.shape[1] // self.ulysses_size
        return cp_region[:, ulysses_rank * ulysses_chunk : (ulysses_rank + 1) * ulysses_chunk].clone()

    def _run_unified_attention_case(self, packed: bool):
        from veomni.ops.kernels.attention import flash_attention_forward

        mesh = self._init()
        device = get_device_type()
        heads, dim = 4, 128
        doc_lens = [64, 32] if packed else [128]
        seq = sum(doc_lens)
        scale = dim**-0.5
        cu = torch.tensor([0, *torch.tensor(doc_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
        q = torch.randn(1, heads, seq, dim, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for tensor in (q, k, v):
            dist.broadcast(tensor, 0)

        q_bshd = q.transpose(1, 2).contiguous()
        k_bshd = k.transpose(1, 2).contiguous()
        v_bshd = v.transpose(1, 2).contiguous()
        if packed:
            reference = _varlen_reference(
                q_bshd.squeeze(0), k_bshd.squeeze(0), v_bshd.squeeze(0), cu, scale
            ).unsqueeze(0)
        else:
            reference = _dense_reference(q_bshd, k_bshd, v_bshd, scale)

        def shard_bhsd(full):
            local = self._usp_slice(full.transpose(1, 2).contiguous(), mesh, cu if packed else None)
            return local.transpose(1, 2).contiguous()

        kwargs = {"cu_seq_lens_q": cu, "cu_seq_lens_k": cu} if packed else {}
        out, _ = flash_attention_forward(
            _AttentionModule().to(device),
            shard_bhsd(q),
            shard_bhsd(k),
            shard_bhsd(v),
            attention_mask=None,
            scaling=scale,
            **kwargs,
        )
        expected = self._usp_slice(reference, mesh, cu if packed else None)
        torch.testing.assert_close(out, expected, atol=_ATOL, rtol=_RTOL)

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_unified_attention_dense_routes_to_npu_ring(self):
        self._run_unified_attention_case(packed=False)

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_unified_attention_varlen_routes_to_npu_ring(self):
        self._run_unified_attention_case(packed=True)


if __name__ == "__main__":
    run_tests()
