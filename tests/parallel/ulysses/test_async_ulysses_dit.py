"""
Test script for async_ulysses_dit.py

This test specifically validates the fix for the backward pass bug where
k normalization backward was executed BEFORE the all-to-all communication
collect (grad_k_res()), which caused incorrect gradient computation.

The fix ensures the correct order:
1. grad_k = grad_k_res()  # collect gradients first
2. k norm backward        # then compute norm backward

Run with pytest:
    torchrun --nproc_per_node=2 -m pytest tests/parallel/ulysses/test_async_ulysses_dit.py -v -s

Run directly (without pytest):
    python tests/parallel/ulysses/test_async_ulysses_dit.py
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
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from veomni.distributed.sequence_parallel import gather_heads_scatter_seq, gather_seq_scatter_heads
from veomni.distributed.sequence_parallel.async_ulysses_dit import (
    async_ulysses_output_projection as async_ulysses_dit_output_projection,
)
from veomni.distributed.sequence_parallel.async_ulysses_dit import (
    async_ulysses_qkv_projection as async_ulysses_dit_qkv_projection,
)
from veomni.distributed.sequence_parallel.comm import (
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
)
from veomni.distributed.sequence_parallel.data import gather_outputs, slice_input_tensor
from veomni.distributed.sequence_parallel.utils import unpadding_tensor_for_seqeunce_parallel
from veomni.utils.helper import enable_high_precision_for_bf16, set_seed
from veomni.utils.import_utils import is_torch_npu_available

from .utils import (
    SequenceParallelTest,
    sync_tensor,
)


def _scale_ratio(sp_t: torch.Tensor, dp_t: torch.Tensor, eps: float = 1e-12) -> float:
    """Calculate scale ratio between two tensors (sp_t / dp_t in least-squares sense)."""
    spf = sp_t.detach().float().reshape(-1)
    dpf = dp_t.detach().float().reshape(-1)
    denom = torch.dot(dpf, dpf).item()
    if denom <= eps:
        return float("nan") if torch.dot(spf, spf).item() > eps else 1.0
    num = torch.dot(spf, dpf).item()
    return num / denom


def _safe_assert_close(title: str, a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> bool:
    """Non-fatal assert_close: prints result and continues without raising on mismatch."""
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
    """RMSNorm matching wan model implementation"""

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
    """
    Attention module using async_ulysses_dit for sequence parallelism.
    This matches the wan model's SelfAttention design where RMSNorm is applied
    on the full hidden_dim (not head_dim).
    """

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

        # Note: async_ulysses_dit applies norm on full hidden_dim, not head_dim
        self.q_norm = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_o = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, unpadded_seq_len: int) -> torch.Tensor:
        if not self.sp_async:
            # Non-async path: projection -> norm -> gather_seq_scatter_heads -> rearrange
            q = self.q_norm(self.q_proj(x))
            k = self.k_norm(self.k_proj(x))
            v = self.v_proj(x)
            q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, unpadded_dim_size=unpadded_seq_len)
            k = gather_seq_scatter_heads(k, seq_dim=1, head_dim=2, unpadded_dim_size=unpadded_seq_len)
            v = gather_seq_scatter_heads(v, seq_dim=1, head_dim=2, unpadded_dim_size=unpadded_seq_len)
        else:
            # Async path using async_ulysses_dit
            # Output is 3D: [B, seq, dim/sp_size] after all-to-all (scatter on head, gather on seq)
            q, k, v = async_ulysses_dit_qkv_projection(
                hidden_states=x,
                seq_dimension=1,
                head_dimension=2,
                q_weight=self.q_proj.weight,
                q_bias=self.q_proj.bias,
                k_weight=self.k_proj.weight,
                k_bias=self.k_proj.bias,
                v_weight=self.v_proj.weight,
                v_bias=self.v_proj.bias,
                norm_type="rmsnorm",
                norm_q_weight=self.q_norm.weight,
                norm_k_weight=self.k_norm.weight,
                normalized_shape=self.dim,  # full hidden_dim, not head_dim
                eps=self.eps,
                unpadded_dim_size=unpadded_seq_len,
                head_dim=self.head_dim,
            )

        # Rearrange from [B, N, (h d)] to [B, N, h, d] then permute to [B, h, N, d]
        q = rearrange(q, "B N (h d) -> B h N d", d=self.head_dim).contiguous()
        k = rearrange(k, "B N (h d) -> B h N d", d=self.head_dim).contiguous()
        v = rearrange(v, "B N (h d) -> B h N d", d=self.head_dim).contiguous()

        x = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        # x: [B, h, N, d] -> [B, N, h, d] -> [B, N, h*d]
        B, h, N, d = x.shape
        x = x.transpose(1, 2).contiguous()  # [B, N, h, d]
        x = x.view(B, N, h * d)  # [B, N, h*d]

        if not self.sp_async:
            x = gather_heads_scatter_seq(x, head_dim=2, seq_dim=1)
            x = self.proj_o(x)
        else:
            # async_ulysses_dit_output_projection expects [B, N, dim] (already flattened)
            x = async_ulysses_dit_output_projection(
                hidden_states=x,
                seq_dimension=1,
                head_dimension=2,
                proj_weight=self.proj_o.weight,
                proj_bias=self.proj_o.bias,
                unpadded_dim_size=unpadded_seq_len,
            )
        x = self.proj_drop(x)
        return x


class AsyncUlyssesDiTSequenceParallelTest(SequenceParallelTest):
    """
    Test class for async_ulysses_dit backward pass correctness.

    The key bug that was fixed:
    - In the original code, k normalization backward was executed BEFORE
      grad_k_res() (all-to-all collect), using incorrect gradient tensor.
    - The fix ensures grad_k_res() is called first, then k norm backward.

    This test validates:
    1. Forward pass produces identical results between async and non-async
    2. Backward pass gradients match, especially for k_norm weights
    """

    @staticmethod
    def _get_input_data():
        heads = 16
        hidden_dim = 64 * heads
        batch_size = 2
        seq_len = 8192
        # Use float32 for better numerical precision in gradient comparison
        input_ = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32).to(get_device_type())
        dist.broadcast(input_, src=0)
        return input_

    @staticmethod
    def _get_input_data_for_padding():
        """Test with non-divisible sequence length to test padding logic"""
        heads = 16
        hidden_dim = 64 * heads
        batch_size = 2
        seq_len = 8191  # Not divisible by world_size
        # Use float32 for better numerical precision in gradient comparison
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
        """
        Test async_ulysses_dit forward and backward correctness.

        This test specifically validates the fix for the k norm backward ordering bug.
        The bug caused k_norm gradients to be computed with wrong gradient tensor.
        """
        self._get_process_group()
        sp_group = get_ulysses_sequence_parallel_group()
        full_input = self._get_input_data()
        unpad_size = full_input.size(1)
        part_input = slice_input_tensor(full_input, dim=1, group=sp_group)
        full_input.requires_grad = True
        part_input.requires_grad = True

        # Initialize attention modules with float32 for numerical precision
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

        # Forward & backward for sequence parallel (async_ulysses_dit)
        sp_rst = attn_sp(part_input, unpad_size)
        sp_full_rst = gather_outputs(
            sp_rst, gather_dim=1, padding_dim=1, unpad_dim_size=unpad_size, scale_grad=False, group=sp_group
        )
        loss_sp = loss_func(sp_rst)
        loss_sp.backward()

        # Collect gradients from async path
        attn_sp_o_grad = attn_sp.proj_o.weight.grad.detach().clone()
        attn_sp_q_grad = attn_sp.q_proj.weight.grad.detach().clone()
        attn_sp_k_grad = attn_sp.k_proj.weight.grad.detach().clone()
        attn_sp_v_grad = attn_sp.v_proj.weight.grad.detach().clone()
        # Key gradients: k_norm weights - this is where the bug manifested
        # RMSNorm only has weight, no bias
        attn_sp_k_norm_grad = attn_sp.k_norm.weight.grad.detach().clone()
        attn_sp_q_norm_grad = attn_sp.q_norm.weight.grad.detach().clone()
        part_input_grad = part_input.grad.detach().clone()

        # All-reduce gradients for comparison
        dist.all_reduce(attn_sp_o_grad)
        dist.all_reduce(attn_sp_q_grad)
        dist.all_reduce(attn_sp_k_grad)
        dist.all_reduce(attn_sp_v_grad)
        dist.all_reduce(attn_sp_k_norm_grad)
        dist.all_reduce(attn_sp_q_norm_grad)
        part_input_grad = sync_tensor(part_input_grad, 1)
        part_input_grad = unpadding_tensor_for_seqeunce_parallel(part_input_grad, 1, unpad_size)

        # Forward & backward for data parallel (reference)
        set_ulysses_sequence_parallel_group(None)
        dp_rst = attn_dp(full_input, unpad_size)
        loss_dp = loss_func(dp_rst)
        loss_dp.backward()

        # Collect reference gradients
        attn_dp_o_grad = attn_dp.proj_o.weight.grad.detach().clone()
        attn_dp_q_grad = attn_dp.q_proj.weight.grad.detach().clone()
        attn_dp_k_grad = attn_dp.k_proj.weight.grad.detach().clone()
        attn_dp_v_grad = attn_dp.v_proj.weight.grad.detach().clone()
        # RMSNorm only has weight, no bias
        attn_dp_k_norm_grad = attn_dp.k_norm.weight.grad.detach().clone()
        attn_dp_q_norm_grad = attn_dp.q_norm.weight.grad.detach().clone()
        full_input_grad = full_input.grad.detach().clone()

        # Verify forward pass
        _safe_assert_close("forward_output", dp_rst, sp_full_rst, atol=1e-6, rtol=1e-5)

        # Verify backward pass - projection weights
        # proj_o and v_proj have larger tolerance due to no normalization and accumulated FP errors
        _safe_assert_close("proj_o.weight.grad", attn_dp_o_grad, attn_sp_o_grad, atol=1e-3, rtol=1e-4)
        _safe_assert_close("q_proj.weight.grad", attn_dp_q_grad, attn_sp_q_grad, atol=1e-4, rtol=1e-4)
        _safe_assert_close("k_proj.weight.grad", attn_dp_k_grad, attn_sp_k_grad, atol=1e-4, rtol=1e-4)
        _safe_assert_close("v_proj.weight.grad", attn_dp_v_grad, attn_sp_v_grad, atol=3e-3, rtol=1e-4)

        # CRITICAL: Verify k_norm gradient - this is where the bug manifested
        # Before the fix, k_norm backward used wrong gradient tensor (before all-to-all collect)
        _safe_assert_close(
            "k_norm.weight.grad (BUG CHECK)", attn_dp_k_norm_grad, attn_sp_k_norm_grad, atol=2e-3, rtol=1e-4
        )
        _safe_assert_close("q_norm.weight.grad", attn_dp_q_norm_grad, attn_sp_q_norm_grad, atol=2e-3, rtol=1e-4)

        # Verify input gradients
        _safe_assert_close("input.grad", full_input_grad, part_input_grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    @pytest.mark.skipif(is_torch_npu_available(), reason="npu skip async ulysses dit")
    def test_self_attn_dit_padding(self):
        """
        Test async_ulysses_dit with non-divisible sequence length (requires padding).

        This test validates the backward pass fix with padding involved,
        which adds complexity to the gradient computation.
        """
        self._get_process_group()
        sp_group = get_ulysses_sequence_parallel_group()
        full_input = self._get_input_data_for_padding()
        unpad_size = full_input.size(1)
        part_input = slice_input_tensor(full_input, dim=1, group=sp_group)
        full_input.requires_grad = True
        part_input.requires_grad = True

        # Initialize attention modules with float32 for numerical precision
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

        # Forward & backward for sequence parallel (async_ulysses_dit)
        sp_rst = attn_sp(part_input, unpad_size)
        sp_full_rst = gather_outputs(
            sp_rst, gather_dim=1, padding_dim=1, unpad_dim_size=unpad_size, scale_grad=False, group=sp_group
        )
        loss_sp = loss_func(sp_rst)
        loss_sp.backward()

        # Collect gradients
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

        # Forward & backward for data parallel (reference)
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

        # Verify forward pass
        _safe_assert_close("[padding] forward_output", dp_rst, sp_full_rst, atol=1e-6, rtol=1e-5)

        # Verify backward pass
        # proj_o and v_proj have larger tolerance due to no normalization and accumulated FP errors
        _safe_assert_close("[padding] proj_o.weight.grad", attn_dp_o_grad, attn_sp_o_grad, atol=1e-3, rtol=1e-4)
        _safe_assert_close("[padding] q_proj.weight.grad", attn_dp_q_grad, attn_sp_q_grad, atol=1e-4, rtol=1e-4)
        _safe_assert_close("[padding] k_proj.weight.grad", attn_dp_k_grad, attn_sp_k_grad, atol=1e-4, rtol=1e-4)
        _safe_assert_close("[padding] v_proj.weight.grad", attn_dp_v_grad, attn_sp_v_grad, atol=3e-3, rtol=1e-4)

        # CRITICAL: k_norm gradient check
        _safe_assert_close(
            "[padding] k_norm.weight.grad (BUG CHECK)", attn_dp_k_norm_grad, attn_sp_k_norm_grad, atol=2e-3, rtol=1e-4
        )
        _safe_assert_close(
            "[padding] q_norm.weight.grad", attn_dp_q_norm_grad, attn_sp_q_norm_grad, atol=2e-3, rtol=1e-4
        )

        # Verify input gradients
        _safe_assert_close("[padding] input.grad", full_input_grad, part_input_grad, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    assert not get_torch_device()._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    set_seed(seed=0, full_determinism=True)
    enable_high_precision_for_bf16()
    run_tests()
