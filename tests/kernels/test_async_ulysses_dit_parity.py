"""Four-GPU DiT async Ulysses parity against the sync gather/scatter path.

Checks QK-norm grads after reverse all-to-all, including padded sequences.
"""

import sys

import pytest
import torch
import torch.distributed as c10d
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from tests.parallel.ulysses.utils import SequenceParallelTest, sync_tensor
from veomni.distributed.sequence_parallel import gather_heads_scatter_seq, gather_seq_scatter_heads
from veomni.distributed.sequence_parallel.comm import (
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
)
from veomni.distributed.sequence_parallel.data import gather_outputs, slice_input_tensor
from veomni.distributed.sequence_parallel.utils import unpadding_tensor_for_seqeunce_parallel
from veomni.kernels import VeomniKernel
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.helper import enable_high_precision_for_bf16, set_seed
from veomni.utils.import_utils import is_torch_npu_available


_NCCL_AVAILABLE = c10d.is_available() and c10d.is_backend_available(get_dist_comm_backend())
if not _NCCL_AVAILABLE:
    if __name__ == "__main__":
        sys.exit(0)
    pytest.skip("c10d NCCL not available", allow_module_level=True)


def _scale_ratio(sp_t: torch.Tensor, dp_t: torch.Tensor, eps: float = 1e-12) -> float:
    """Least-squares scale of ``sp_t`` relative to ``dp_t``."""
    spf = sp_t.detach().float().reshape(-1)
    dpf = dp_t.detach().float().reshape(-1)
    denom = torch.dot(dpf, dpf).item()
    if denom <= eps:
        return float("nan") if torch.dot(spf, spf).item() > eps else 1.0
    num = torch.dot(spf, dpf).item()
    return num / denom


def _safe_assert_close(title: str, a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> bool:
    """Print max-abs and scale, then ``assert_close``. Returns whether they matched."""
    max_diff = (a.detach().float() - b.detach().float()).abs().max().item()
    ratio = _scale_ratio(a, b)
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        if dist.get_rank() == 0:
            print(f"[PASS] {title}: equal=True, ratio={ratio:.6f}, max_abs_diff={max_diff:.6e}")
        return True
    except AssertionError:
        if dist.get_rank() == 0:
            print(f"[FAIL] {title}: equal=False, ratio={ratio:.6f}, max_abs_diff={max_diff:.6e}")
        return False


class RMSNorm(nn.Module):
    """RMSNorm on the last dim, matching the Wan SelfAttention weight layout."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionDiT(nn.Module):
    """Wan-style attention. QK RMSNorm is on the full hidden dim, not head dim."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sp_async: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.sp_async = sp_async
        self.eps = eps

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_o = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, unpadded_seq_len: int) -> torch.Tensor:
        if not self.sp_async:
            q = self.q_norm(self.q_proj(x))
            k = self.k_norm(self.k_proj(x))
            v = self.v_proj(x)
            q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, unpadded_dim_size=unpadded_seq_len)
            k = gather_seq_scatter_heads(k, seq_dim=1, head_dim=2, unpadded_dim_size=unpadded_seq_len)
            v = gather_seq_scatter_heads(v, seq_dim=1, head_dim=2, unpadded_dim_size=unpadded_seq_len)
        else:
            q, k, v = VeomniKernel("async_ulysses_qkv", "dit")(
                x,
                self.q_proj.weight,
                self.q_proj.bias,
                self.k_proj.weight,
                self.k_proj.bias,
                self.v_proj.weight,
                self.v_proj.bias,
                self.q_norm.weight,
                None,
                self.k_norm.weight,
                None,
                seq_dimension=1,
                head_dimension=2,
                unpadded_dim_size=unpadded_seq_len,
                head_dim=self.head_dim,
                norm_type="rmsnorm",
                normalized_shape=self.dim,
                eps=self.eps,
            )

        q = rearrange(q, "B N (h d) -> B h N d", d=self.head_dim).contiguous()
        k = rearrange(k, "B N (h d) -> B h N d", d=self.head_dim).contiguous()
        v = rearrange(v, "B N (h d) -> B h N d", d=self.head_dim).contiguous()

        x = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        B, h, N, d = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, N, h * d)

        if not self.sp_async:
            x = gather_heads_scatter_seq(x, head_dim=2, seq_dim=1)
            x = self.proj_o(x)
        else:
            x = VeomniKernel("async_ulysses_o", "dit")(
                x,
                self.proj_o.weight,
                self.proj_o.bias,
                seq_dimension=1,
                head_dimension=2,
                unpadded_dim_size=unpadded_seq_len,
            )
        x = self.proj_drop(x)
        return x


class AsyncUlyssesDiTSequenceParallelTest(SequenceParallelTest):
    """DiT async QKV/O vs sync gather/scatter, including QK-norm grads."""

    @staticmethod
    def _get_input_data():
        heads = 16
        hidden_dim = 64 * heads
        batch_size = 2
        seq_len = 8192
        input_ = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32).to(get_device_type())
        dist.broadcast(input_, src=0)
        return input_

    @staticmethod
    def _get_input_data_for_padding():
        heads = 16
        hidden_dim = 64 * heads
        batch_size = 2
        seq_len = 8191
        input_ = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32).to(get_device_type())
        dist.broadcast(input_, src=0)
        return input_

    @staticmethod
    def _overlapping_grad(output) -> torch.Tensor:
        return output.sum() * 2

    @staticmethod
    def _non_overlapping_grad(output) -> torch.Tensor:
        t = torch.ones_like(output)
        return torch.sum(output * t)

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    @pytest.mark.skipif(is_torch_npu_available(), reason="npu skip async ulysses dit")
    def test_self_attn_dit(self):
        """Compare DiT async and sync attention outputs and grads."""
        self._get_process_group()
        sp_group = get_ulysses_sequence_parallel_group()
        full_input = self._get_input_data()
        unpad_size = full_input.size(1)
        part_input = slice_input_tensor(full_input, dim=1, group=sp_group)
        full_input.requires_grad = True
        part_input.requires_grad = True

        attn_dp = (
            AttentionDiT(
                dim=64 * 16, num_heads=16, qkv_bias=False, qk_norm=True, attn_drop=0, proj_drop=0, sp_async=False
            )
            .to(get_device_type())
            .float()
        )
        attn_sp = (
            AttentionDiT(
                dim=64 * 16, num_heads=16, qkv_bias=False, qk_norm=True, attn_drop=0, proj_drop=0, sp_async=True
            )
            .to(get_device_type())
            .float()
        )
        attn_sp.load_state_dict(self._sync_model(attn_sp.state_dict(), self.rank))
        attn_dp.load_state_dict(self._sync_model(attn_sp.state_dict(), self.rank))

        loss_func = self._overlapping_grad

        sp_rst = attn_sp(part_input, unpad_size)
        sp_full_rst = gather_outputs(
            sp_rst, gather_dim=1, padding_dim=1, unpad_dim_size=unpad_size, scale_grad=False, group=sp_group
        )
        loss_sp = loss_func(sp_rst)
        loss_sp.backward()

        attn_sp_o_grad = attn_sp.proj_o.weight.grad.detach().clone()
        attn_sp_q_grad = attn_sp.q_proj.weight.grad.detach().clone()
        attn_sp_k_grad = attn_sp.k_proj.weight.grad.detach().clone()
        attn_sp_v_grad = attn_sp.v_proj.weight.grad.detach().clone()
        attn_sp_k_norm_grad = attn_sp.k_norm.weight.grad.detach().clone()
        attn_sp_q_norm_grad = attn_sp.q_norm.weight.grad.detach().clone()
        part_input_grad = part_input.grad.detach().clone()

        dist.all_reduce(attn_sp_o_grad)
        dist.all_reduce(attn_sp_q_grad)
        dist.all_reduce(attn_sp_k_grad)
        dist.all_reduce(attn_sp_v_grad)
        dist.all_reduce(attn_sp_k_norm_grad)
        dist.all_reduce(attn_sp_q_norm_grad)
        part_input_grad = sync_tensor(part_input_grad, 1)
        part_input_grad = unpadding_tensor_for_seqeunce_parallel(part_input_grad, 1, unpad_size)

        set_ulysses_sequence_parallel_group(None)
        dp_rst = attn_dp(full_input, unpad_size)
        loss_dp = loss_func(dp_rst)
        loss_dp.backward()

        attn_dp_o_grad = attn_dp.proj_o.weight.grad.detach().clone()
        attn_dp_q_grad = attn_dp.q_proj.weight.grad.detach().clone()
        attn_dp_k_grad = attn_dp.k_proj.weight.grad.detach().clone()
        attn_dp_v_grad = attn_dp.v_proj.weight.grad.detach().clone()
        attn_dp_k_norm_grad = attn_dp.k_norm.weight.grad.detach().clone()
        attn_dp_q_norm_grad = attn_dp.q_norm.weight.grad.detach().clone()
        full_input_grad = full_input.grad.detach().clone()

        _safe_assert_close("forward_output", dp_rst, sp_full_rst, atol=1e-6, rtol=1e-5)
        _safe_assert_close("proj_o.weight.grad", attn_dp_o_grad, attn_sp_o_grad, atol=1e-3, rtol=1e-4)
        _safe_assert_close("q_proj.weight.grad", attn_dp_q_grad, attn_sp_q_grad, atol=1e-4, rtol=1e-4)
        _safe_assert_close("k_proj.weight.grad", attn_dp_k_grad, attn_sp_k_grad, atol=1e-4, rtol=1e-4)
        _safe_assert_close("v_proj.weight.grad", attn_dp_v_grad, attn_sp_v_grad, atol=3e-3, rtol=1e-4)
        _safe_assert_close("k_norm.weight.grad", attn_dp_k_norm_grad, attn_sp_k_norm_grad, atol=2e-3, rtol=1e-4)
        _safe_assert_close("q_norm.weight.grad", attn_dp_q_norm_grad, attn_sp_q_norm_grad, atol=2e-3, rtol=1e-4)
        _safe_assert_close("input.grad", full_input_grad, part_input_grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    @pytest.mark.skipif(is_torch_npu_available(), reason="npu skip async ulysses dit")
    def test_self_attn_dit_padding(self):
        """Same comparison with a sequence length that needs SP padding."""
        self._get_process_group()
        sp_group = get_ulysses_sequence_parallel_group()
        full_input = self._get_input_data_for_padding()
        unpad_size = full_input.size(1)
        part_input = slice_input_tensor(full_input, dim=1, group=sp_group)
        full_input.requires_grad = True
        part_input.requires_grad = True

        attn_dp = (
            AttentionDiT(
                dim=64 * 16, num_heads=16, qkv_bias=False, qk_norm=True, attn_drop=0, proj_drop=0, sp_async=False
            )
            .to(get_device_type())
            .float()
        )
        attn_sp = (
            AttentionDiT(
                dim=64 * 16, num_heads=16, qkv_bias=False, qk_norm=True, attn_drop=0, proj_drop=0, sp_async=True
            )
            .to(get_device_type())
            .float()
        )
        attn_sp.load_state_dict(self._sync_model(attn_sp.state_dict(), self.rank))
        attn_dp.load_state_dict(self._sync_model(attn_sp.state_dict(), self.rank))

        loss_func = self._non_overlapping_grad

        sp_rst = attn_sp(part_input, unpad_size)
        sp_full_rst = gather_outputs(
            sp_rst, gather_dim=1, padding_dim=1, unpad_dim_size=unpad_size, scale_grad=False, group=sp_group
        )
        loss_sp = loss_func(sp_rst)
        loss_sp.backward()

        attn_sp_o_grad = attn_sp.proj_o.weight.grad.detach().clone()
        attn_sp_q_grad = attn_sp.q_proj.weight.grad.detach().clone()
        attn_sp_k_grad = attn_sp.k_proj.weight.grad.detach().clone()
        attn_sp_v_grad = attn_sp.v_proj.weight.grad.detach().clone()
        attn_sp_k_norm_grad = attn_sp.k_norm.weight.grad.detach().clone()
        attn_sp_q_norm_grad = attn_sp.q_norm.weight.grad.detach().clone()
        part_input_grad = part_input.grad.detach().clone()

        dist.all_reduce(attn_sp_o_grad)
        dist.all_reduce(attn_sp_q_grad)
        dist.all_reduce(attn_sp_k_grad)
        dist.all_reduce(attn_sp_v_grad)
        dist.all_reduce(attn_sp_k_norm_grad)
        dist.all_reduce(attn_sp_q_norm_grad)
        part_input_grad = sync_tensor(part_input_grad, 1)
        part_input_grad = unpadding_tensor_for_seqeunce_parallel(part_input_grad, 1, unpad_size)

        set_ulysses_sequence_parallel_group(None)
        dp_rst = attn_dp(full_input, unpad_size)
        loss_dp = loss_func(dp_rst)
        loss_dp.backward()

        attn_dp_o_grad = attn_dp.proj_o.weight.grad.detach().clone()
        attn_dp_q_grad = attn_dp.q_proj.weight.grad.detach().clone()
        attn_dp_k_grad = attn_dp.k_proj.weight.grad.detach().clone()
        attn_dp_v_grad = attn_dp.v_proj.weight.grad.detach().clone()
        attn_dp_k_norm_grad = attn_dp.k_norm.weight.grad.detach().clone()
        attn_dp_q_norm_grad = attn_dp.q_norm.weight.grad.detach().clone()
        full_input_grad = full_input.grad.detach().clone()

        _safe_assert_close("[padding] forward_output", dp_rst, sp_full_rst, atol=1e-6, rtol=1e-5)
        _safe_assert_close("[padding] proj_o.weight.grad", attn_dp_o_grad, attn_sp_o_grad, atol=1e-3, rtol=1e-4)
        _safe_assert_close("[padding] q_proj.weight.grad", attn_dp_q_grad, attn_sp_q_grad, atol=1e-4, rtol=1e-4)
        _safe_assert_close("[padding] k_proj.weight.grad", attn_dp_k_grad, attn_sp_k_grad, atol=1e-4, rtol=1e-4)
        _safe_assert_close("[padding] v_proj.weight.grad", attn_dp_v_grad, attn_sp_v_grad, atol=3e-3, rtol=1e-4)
        _safe_assert_close(
            "[padding] k_norm.weight.grad", attn_dp_k_norm_grad, attn_sp_k_norm_grad, atol=2e-3, rtol=1e-4
        )
        _safe_assert_close(
            "[padding] q_norm.weight.grad", attn_dp_q_norm_grad, attn_sp_q_norm_grad, atol=2e-3, rtol=1e-4
        )
        _safe_assert_close("[padding] input.grad", full_input_grad, part_input_grad, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    assert not get_torch_device()._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    set_seed(seed=0, full_determinism=True)
    enable_high_precision_for_bf16()
    run_tests()
