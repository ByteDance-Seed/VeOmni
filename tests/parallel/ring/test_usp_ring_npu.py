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

"""Ascend numerical tests for NPU zig-zag / bidirectional (ViT) Ring Attention and USP routing."""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

from veomni.distributed import parallel_state as PS
from veomni.distributed.sequence_parallel.ring_attention.layout import (
    local_cu_seqlens,
    vit_cp_cu_seqlens,
    vit_usp_slice,
    zigzag_reorder,
    zigzag_reorder_packed,
)
from veomni.distributed.sequence_parallel.ring_attention.npu import (
    _npu_fa_backward,
    _npu_fa_forward,
    ring_npu_flash_attn_varlen_func,
    update_npu_out_and_softmax_stats,
    zigzag_ring_npu_flash_attn_func,
    zigzag_ring_npu_flash_attn_varlen_func,
)
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.import_utils import is_torch_npu_available


pytestmark = pytest.mark.skipif(not is_torch_npu_available(), reason="torch_npu is required")

_ATOL = 8e-2
_RTOL = 6e-2


def _dense_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, causal: bool = True
) -> torch.Tensor:
    dtype = q.dtype
    q = q.float().transpose(1, 2)
    k = k.float().transpose(1, 2)
    v = v.float().transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
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
    causal: bool = True,
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
                causal,
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


# ── zero-length segments in npu_fusion_attention (bidirectional ViT ring) ─────
# The bidirectional ring pairs per-rank vit shards; a shard is empty on some rank
# when a vit seq has fewer merger units than cp ranks, so the TND offsets contain
# repeated endpoints (including a leading 0). These single-card probes pin the
# torch_npu behaviour the ring relies on.
@pytest.mark.parametrize("head_dim", [72, 128])
def test_npu_fusion_attention_zero_length_segments(head_dim):
    device = get_device_type()
    heads, scale = 4, head_dim**-0.5
    # seg0: empty Q, 4 K rows | seg1: 8 Q rows, empty K | seg2: 4 Q rows, 8 K rows
    cu_q, cu_k = [0, 0, 8, 12], [0, 4, 4, 12]
    q = torch.randn(12, heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)
    kwargs = dict(input_layout="TND", softmax_scale=scale, dropout_p=0.0, causal=False, softmax_layout="TND")

    out, smax, ssum, rng = _npu_fa_forward(q, k, v, actual_seq_qlen=cu_q, actual_seq_kvlen=cu_k, **kwargs)
    assert torch.isfinite(out).all()
    assert (ssum[0:8] == 0).all(), "rows without keys must report softmax_sum == 0"
    assert not torch.isposinf(smax[0:8]).any(), "rows without keys must not report softmax_max == +inf"
    scores = torch.einsum("qnd,knd->nqk", q[8:12].float(), k[4:12].float()) * scale
    expected = torch.einsum("nqk,knd->qnd", torch.softmax(scores, -1), v[4:12].float()).to(out.dtype)
    torch.testing.assert_close(out[8:12], expected, atol=_ATOL, rtol=_RTOL)

    # Merging an empty-key block must leave previously accumulated rows unchanged.
    prev_out, prev_max, prev_sum = update_npu_out_and_softmax_stats(
        None, None, None, *_npu_fa_forward(q, k, v, actual_seq_qlen=[0, 12], actual_seq_kvlen=[0, 12], **kwargs)[:3]
    )
    merged_out, _, _ = update_npu_out_and_softmax_stats(prev_out.clone(), prev_max, prev_sum, out, smax, ssum)
    assert torch.isfinite(merged_out).all()
    torch.testing.assert_close(merged_out[0:8], prev_out[0:8])

    smax_rows, ssum_rows = smax[..., 0], ssum[..., 0]
    dq, dk, dv = _npu_fa_backward(
        dout, q, k, v, out, smax_rows, ssum_rows, rng_state=rng, actual_seq_qlen=cu_q, actual_seq_kvlen=cu_k, **kwargs
    )
    for grad in (dq, dk, dv):
        assert torch.isfinite(grad).all()
    assert (dq[0:8] == 0).all(), "queries without keys must get zero dq"
    assert (dk[0:4] == 0).all() and (dv[0:4] == 0).all(), "keys without queries must get zero dk/dv"


# ── bidirectional (ViT) packed ring ───────────────────────────────────────────
class _NPUBidirectionalRingTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _run_case(self):
        store = dist.FileStore(self.file_name, self.world_size)
        get_torch_device().set_device(self.rank)
        dist.init_process_group(get_dist_comm_backend(), store=store, rank=self.rank, world_size=self.world_size)
        group = dist.distributed_c10d._get_default_group()
        device = get_device_type()
        heads, dim = 4, 72  # Qwen3-VL ViT head_dim
        # Two vit seqs have fewer merger units than ranks -> empty shards mid-cu.
        vit_seqlens = [4, 360, 8, 124]
        total = sum(vit_seqlens)
        scale = dim**-0.5
        global_cu = torch.tensor([0, *torch.tensor(vit_seqlens).cumsum(0).tolist()])

        q = torch.randn(total, heads, dim, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        grad = torch.randn_like(q)
        for tensor in (q, k, v, grad):
            dist.broadcast(tensor, 0)

        q_ref, k_ref, v_ref = (t.clone().requires_grad_(True) for t in (q, k, v))
        out_ref = _varlen_reference(q_ref, k_ref, v_ref, global_cu, scale, causal=False)
        out_ref.backward(grad)

        def shard(t, pad_value=0):
            return vit_usp_slice(t, vit_seqlens, 0, self.world_size, self.rank, 1, 0, pad_value=pad_value)

        cu_per_rank = [vit_cp_cu_seqlens(global_cu, self.world_size, c, 1) for c in range(self.world_size)]
        q_local, k_local, v_local = (shard(t).requires_grad_(True) for t in (q, k, v))
        out = ring_npu_flash_attn_varlen_func(q_local, k_local, v_local, cu_per_rank, softmax_scale=scale, group=group)
        out.backward(shard(grad))

        rows = shard(torch.arange(total, device=device), pad_value=-1)
        real = rows >= 0
        torch.testing.assert_close(out[real], out_ref.detach()[rows[real]], atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(q_local.grad[real], q_ref.grad[rows[real]], atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(k_local.grad[real], k_ref.grad[rows[real]], atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(v_local.grad[real], v_ref.grad[rows[real]], atol=_ATOL, rtol=_RTOL)


class NPUBidirectionalRingWorld2Test(_NPUBidirectionalRingTest):
    @property
    def world_size(self):
        return 2

    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    def test_bidirectional_ring_matches_full_attention(self):
        self._run_case()


class NPUBidirectionalRingWorld4Test(_NPUBidirectionalRingTest):
    @property
    def world_size(self):
        return 4

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_bidirectional_ring_matches_full_attention(self):
        self._run_case()


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
            cp_layout="zigzag",
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

    def _run_bidirectional_attention_case(self):
        from veomni.ops.kernels.attention import flash_attention_forward

        mesh = self._init()
        device = get_device_type()
        cp_rank, ulysses_rank = mesh.get_local_rank("cp"), mesh.get_local_rank("ulysses")
        heads, dim = 4, 72
        vit_seqlens = [4, 360, 8, 124]
        total = sum(vit_seqlens)
        scale = dim**-0.5
        # ViT on NPU keeps cu_seqlens on the host (Patch.5 of the Qwen3-VL ViT forward).
        global_cu = torch.tensor([0, *torch.tensor(vit_seqlens).cumsum(0).tolist()], dtype=torch.int32)
        q = torch.randn(1, heads, total, dim, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for tensor in (q, k, v):
            dist.broadcast(tensor, 0)

        def shard(full, dim_, pad_value=0):
            return vit_usp_slice(
                full, vit_seqlens, dim_, self.cp_size, cp_rank, self.ulysses_size, ulysses_rank, pad_value=pad_value
            )

        reference = _varlen_reference(
            q.transpose(1, 2).squeeze(0),
            k.transpose(1, 2).squeeze(0),
            v.transpose(1, 2).squeeze(0),
            global_cu,
            scale,
            causal=False,
        )
        module = _AttentionModule().to(device)
        module.is_causal = False
        out, _ = flash_attention_forward(
            module,
            shard(q, 2),
            shard(k, 2),
            shard(v, 2),
            attention_mask=None,
            scaling=scale,
            cu_seq_lens_q=global_cu,
            cu_seq_lens_k=global_cu,
            is_causal=False,
        )
        rows = shard(torch.arange(total, device=device), 0, pad_value=-1)
        real = rows >= 0
        torch.testing.assert_close(out.squeeze(0)[real], reference[rows[real]], atol=_ATOL, rtol=_RTOL)

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_unified_attention_bidirectional_routes_to_npu_vit_ring(self):
        self._run_bidirectional_attention_case()

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_unified_attention_dense_routes_to_npu_ring(self):
        self._run_unified_attention_case(packed=False)

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_unified_attention_varlen_routes_to_npu_ring(self):
        self._run_unified_attention_case(packed=True)


if __name__ == "__main__":
    run_tests()
