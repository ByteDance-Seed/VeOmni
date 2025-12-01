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

from veomni.distributed.sequence_parallel.comm import (
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
)
from veomni.distributed.sequence_parallel.data import gather_outputs, slice_input_tensor
from veomni.distributed.sequence_parallel.utils import unpadding_tensor_for_seqeunce_parallel
from veomni.utils.helper import enable_high_precision_for_bf16, set_seed

from .attention import Attention
from .utils import (
    SequenceParallelTest,
    sync_tensor,
)


class AsyncAttentionSequenceParallelTest(SequenceParallelTest):
    @staticmethod
    def _get_input_data():
        heads = 16
        hidden_dim = 64 * heads
        batch_size = 2
        seq_len = 8192
        input_ = torch.randn(batch_size, seq_len, hidden_dim).to(get_device_type())
        dist.broadcast(input_, src=0)

        return input_

    @staticmethod
    def _get_input_data_for_padding():
        heads = 16
        hidden_dim = 64 * heads
        batch_size = 2
        seq_len = 8191
        input_ = torch.randn(batch_size, seq_len, hidden_dim).to(get_device_type())
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
    def test_self_attn(self):
        self._get_process_group()
        sp_group = get_ulysses_sequence_parallel_group()
        full_input = self._get_input_data()
        unpad_size = full_input.size(1)
        part_input = slice_input_tensor(full_input, dim=1, group=sp_group)
        full_input.requires_grad = True
        part_input.requires_grad = True

        # initialize attn module
        attn_dp = Attention(
            dim=64 * 16, num_heads=16, qkv_bias=False, qk_norm=True, attn_drop=0, proj_drop=0, sp_async=False
        ).to(get_device_type())
        attn_sp = Attention(
            dim=64 * 16, num_heads=16, qkv_bias=False, qk_norm=True, attn_drop=0, proj_drop=0, sp_async=False
        ).to(get_device_type())
        attn_sp.load_state_dict(self._sync_model(attn_sp.state_dict(), self.rank))
        attn_dp.load_state_dict(self._sync_model(attn_sp.state_dict(), self.rank))

        loss_func = self._overlapping_grad

        # forward & backward for sp
        sp_rst = attn_sp(part_input, unpad_size)
        sp_full_rst = gather_outputs(
            sp_rst, gather_dim=1, padding_dim=1, unpad_dim_size=unpad_size, scale_grad=False, group=sp_group
        )
        loss_sp = loss_func(sp_rst)
        loss_sp.backward()
        attn_sp_o_grad = attn_sp.proj_o.weight.grad.detach().clone()
        attn_sp_q_grad = attn_sp.q_proj.weight.grad.detach().clone()
        part_input_grad = part_input.grad.detach().clone()
        dist.all_reduce(attn_sp_o_grad)
        dist.all_reduce(attn_sp_q_grad)
        part_input_grad = sync_tensor(part_input_grad, 1)
        part_input_grad = unpadding_tensor_for_seqeunce_parallel(part_input_grad, 1, unpad_size)

        # forward & backward for dp
        set_ulysses_sequence_parallel_group(None)
        dp_rst = attn_dp(full_input, unpad_size)
        loss_dp = loss_func(dp_rst)
        loss_dp.backward()
        attn_dp_o_grad = attn_dp.proj_o.weight.grad.detach().clone()
        attn_dp_q_grad = attn_dp.q_proj.weight.grad.detach().clone()
        full_input_grad = full_input.grad.detach().clone()

        torch.testing.assert_close(dp_rst, sp_full_rst, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(attn_dp_o_grad, attn_sp_o_grad, atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(attn_dp_q_grad, attn_sp_q_grad, atol=2e-3, rtol=1e-4)
        torch.testing.assert_close(full_input_grad, part_input_grad, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_self_attn_padding(self):
        self._get_process_group()
        sp_group = get_ulysses_sequence_parallel_group()
        full_input = self._get_input_data_for_padding()
        unpad_size = full_input.size(1)
        part_input = slice_input_tensor(full_input, dim=1, group=sp_group)
        full_input.requires_grad = True
        part_input.requires_grad = True

        # initialize attn module
        attn_dp = Attention(
            dim=64 * 16, num_heads=16, qkv_bias=False, qk_norm=True, attn_drop=0, proj_drop=0, sp_async=False
        ).to(get_device_type())
        attn_sp = Attention(
            dim=64 * 16, num_heads=16, qkv_bias=False, qk_norm=True, attn_drop=0, proj_drop=0, sp_async=False
        ).to(get_device_type())
        attn_sp.load_state_dict(self._sync_model(attn_sp.state_dict(), self.rank))
        attn_dp.load_state_dict(self._sync_model(attn_sp.state_dict(), self.rank))

        loss_func = self._non_overlapping_grad

        # forward & backward for sp
        sp_rst = attn_sp(part_input, unpad_size)
        sp_full_rst = gather_outputs(
            sp_rst, gather_dim=1, padding_dim=1, unpad_dim_size=unpad_size, scale_grad=False, group=sp_group
        )
        loss_sp = loss_func(sp_rst)
        loss_sp.backward()
        attn_sp_o_grad = attn_sp.proj_o.weight.grad.detach().clone()
        attn_sp_q_grad = attn_sp.q_proj.weight.grad.detach().clone()
        part_input_grad = part_input.grad.detach().clone()
        dist.all_reduce(attn_sp_o_grad)
        dist.all_reduce(attn_sp_q_grad)
        part_input_grad = sync_tensor(part_input_grad, 1)
        part_input_grad = unpadding_tensor_for_seqeunce_parallel(part_input_grad, 1, unpad_size)

        # forward & backward for dp
        set_ulysses_sequence_parallel_group(None)
        dp_rst = attn_dp(full_input, unpad_size)
        loss_dp = loss_func(dp_rst)
        loss_dp.backward()
        attn_dp_o_grad = attn_dp.proj_o.weight.grad.detach().clone()
        attn_dp_q_grad = attn_dp.q_proj.weight.grad.detach().clone()
        full_input_grad = full_input.grad.detach().clone()

        torch.testing.assert_close(dp_rst, sp_full_rst, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(attn_dp_o_grad, attn_sp_o_grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(attn_dp_q_grad, attn_sp_q_grad, atol=2e-3, rtol=1e-4)
        torch.testing.assert_close(full_input_grad, part_input_grad, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    assert not get_torch_device()._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    set_seed(seed=0, full_determinism=True)
    enable_high_precision_for_bf16()
    run_tests()
