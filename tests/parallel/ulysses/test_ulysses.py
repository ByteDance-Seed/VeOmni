import sys
from types import SimpleNamespace

import torch
import torch.distributed as c10d
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PreTrainedConfig

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
from veomni.models.transformers.masking_utils import create_causal_mask
from veomni.ops.kernels.attention import flash as flash_backend
from veomni.ops.kernels.attention import flex as flex_backend
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

        attn_dp = Attention(
            dim=64 * 16, num_heads=16, qkv_bias=False, qk_norm=True, attn_drop=0, proj_drop=0, sp_async=False
        ).to(get_device_type())
        attn_sp = Attention(
            dim=64 * 16, num_heads=16, qkv_bias=False, qk_norm=True, attn_drop=0, proj_drop=0, sp_async=False
        ).to(get_device_type())
        attn_sp.load_state_dict(self._sync_model(attn_sp.state_dict(), self.rank))
        attn_dp.load_state_dict(self._sync_model(attn_sp.state_dict(), self.rank))

        sp_rst = attn_sp(part_input, unpad_size)
        sp_full_rst = gather_outputs(
            sp_rst, gather_dim=1, padding_dim=1, unpad_dim_size=unpad_size, scale_grad=False, group=sp_group
        )
        loss_sp = self._non_overlapping_grad(sp_rst)
        loss_sp.backward()
        attn_sp_o_grad = attn_sp.proj_o.weight.grad.detach().clone()
        attn_sp_q_grad = attn_sp.q_proj.weight.grad.detach().clone()
        part_input_grad = part_input.grad.detach().clone()
        dist.all_reduce(attn_sp_o_grad)
        dist.all_reduce(attn_sp_q_grad)
        part_input_grad = sync_tensor(part_input_grad, 1)
        part_input_grad = unpadding_tensor_for_seqeunce_parallel(part_input_grad, 1, unpad_size)

        set_ulysses_sequence_parallel_group(None)
        dp_rst = attn_dp(full_input, unpad_size)
        loss_dp = self._non_overlapping_grad(dp_rst)
        loss_dp.backward()
        attn_dp_o_grad = attn_dp.proj_o.weight.grad.detach().clone()
        attn_dp_q_grad = attn_dp.q_proj.weight.grad.detach().clone()
        full_input_grad = full_input.grad.detach().clone()

        torch.testing.assert_close(dp_rst, sp_full_rst, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(attn_dp_o_grad, attn_sp_o_grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(attn_dp_q_grad, attn_sp_q_grad, atol=2e-3, rtol=1e-4)
        torch.testing.assert_close(full_input_grad, part_input_grad, atol=1e-5, rtol=1e-5)


class _FakeFlashAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="veomni_flash_attention_2_with_sp")
        self.is_causal = False
        self.proj = nn.Linear(1, 1, bias=False)


class _FakeFlexAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="flex_attention")


def _sdpa_flash_oracle(query, key, value, attention_mask, **kwargs):
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=kwargs["is_causal"],
            enable_gqa=True,
        )
    return output.transpose(1, 2).contiguous()


class AttentionBackendSequenceParallelTest(SequenceParallelTest):
    def _build_qkv(self, *, sequence_length, head_dim, dtype, seed):
        group = self._get_process_group()
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        device = torch.device(get_device_type(), rank)
        generator = torch.Generator(device=device).manual_seed(seed)
        tensors = [
            torch.randn(1, heads, sequence_length, head_dim, device=device, dtype=dtype, generator=generator)
            for heads in (4, 1, 1)
        ]
        output_gradient = torch.randn(1, sequence_length, 4, head_dim, device=device, dtype=dtype, generator=generator)
        for tensor in (*tensors, output_gradient):
            dist.broadcast(tensor, src=0, group=group)
        local_slice = slice(rank * sequence_length // world_size, (rank + 1) * sequence_length // world_size)
        baseline = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
        local = [tensor[:, :, local_slice, :].detach().clone().requires_grad_(True) for tensor in tensors]
        return group, world_size, device, local_slice, baseline, local, output_gradient

    @staticmethod
    def _assert_qkv_gradients(local, baseline):
        for local_tensor, baseline_tensor in zip(local, baseline, strict=True):
            gathered_gradient = sync_tensor(local_tensor.grad, dim=2)
            torch.testing.assert_close(gathered_gradient, baseline_tensor.grad, rtol=8e-2, atol=8e-2)

    @pytest.mark.skipif(
        get_device_type() != "cuda" or get_torch_device().device_count() < 2,
        reason="attention backend Ulysses parity requires at least 2 CUDA devices",
    )
    def test_flash_wrapper_matches_non_sp_oracle(self):
        group, world_size, device, local_slice, baseline, local, output_gradient = self._build_qkv(
            sequence_length=8,
            head_dim=8,
            dtype=torch.float32,
            seed=9171,
        )
        module = _FakeFlashAttentionModule().to(device)
        original_backend = flash_backend._flash_attention_forward
        original_get_parallel_state = flash_backend.get_parallel_state
        try:
            flash_backend._flash_attention_forward = _sdpa_flash_oracle
            flash_backend.get_parallel_state = lambda: SimpleNamespace(ulysses_enabled=False)
            baseline_output, _ = flash_backend.flash_attention_forward(module, *baseline, attention_mask=None)

            flash_backend.get_parallel_state = lambda: SimpleNamespace(
                ulysses_enabled=True,
                ulysses_group=group,
                ulysses_size=world_size,
            )
            local_output, _ = flash_backend.flash_attention_forward(module, *local, attention_mask=None)

            torch.testing.assert_close(sync_tensor(local_output, dim=1), baseline_output, rtol=1e-5, atol=1e-6)
            baseline_output.backward(output_gradient)
            local_output.backward(output_gradient[:, local_slice].contiguous())
            self._assert_qkv_gradients(local, baseline)
        finally:
            flash_backend._flash_attention_forward = original_backend
            flash_backend.get_parallel_state = original_get_parallel_state

    @pytest.mark.skipif(
        get_device_type() != "cuda" or get_torch_device().device_count() < 2,
        reason="FlexAttention Ulysses parity requires at least 2 CUDA devices",
    )
    def test_flex_wrapper_matches_non_sp_oracle(self):
        group, world_size, device, local_slice, baseline, local, output_gradient = self._build_qkv(
            sequence_length=16,
            head_dim=16,
            dtype=torch.bfloat16,
            seed=9190,
        )
        generator = torch.Generator(device=device).manual_seed(9191)
        head_auxiliary = torch.randn(4, device=device, dtype=torch.bfloat16, generator=generator)
        dist.broadcast(head_auxiliary, src=0, group=group)

        def mask_mod(batch_idx, head_idx, query_idx, key_idx):
            query_span = query_idx // 4
            key_span = key_idx // 4
            return (query_span == key_span) | ((query_idx >= key_idx) & (key_span % 2 == 0))

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=16,
            KV_LEN=16,
            device=device,
            BLOCK_SIZE=128,
        )
        module = _FakeFlexAttentionModule().to(device)
        original_get_parallel_state = flex_backend.get_parallel_state
        try:
            flex_backend.get_parallel_state = lambda: SimpleNamespace(ulysses_enabled=False)
            baseline_output, baseline_lse = flex_backend.flex_attention_forward(
                module,
                *baseline,
                block_mask,
                kernel_options={"BACKEND": "TRITON"},
                s_aux=head_auxiliary,
            )

            flex_backend.get_parallel_state = lambda: SimpleNamespace(
                ulysses_enabled=True,
                ulysses_group=group,
                ulysses_size=world_size,
            )
            local_output, local_lse = flex_backend.flex_attention_forward(
                module,
                *local,
                block_mask,
                kernel_options={"BACKEND": "TRITON"},
                s_aux=head_auxiliary,
            )

            torch.testing.assert_close(sync_tensor(local_output, dim=1), baseline_output, rtol=3e-2, atol=3e-2)
            torch.testing.assert_close(sync_tensor(local_lse, dim=2), baseline_lse, rtol=3e-2, atol=3e-2)
            baseline_output.backward(output_gradient)
            local_output.backward(output_gradient[:, local_slice].contiguous())
            self._assert_qkv_gradients(local, baseline)
        finally:
            flex_backend.get_parallel_state = original_get_parallel_state

    @pytest.mark.skipif(
        get_device_type() != "cuda" or get_torch_device().device_count() < 2,
        reason="packed FlexAttention Ulysses parity requires at least 2 CUDA devices",
    )
    def test_packed_causal_flex_wrapper_matches_non_sp_oracle(self):
        sequence_length = 16
        group, world_size, device, local_slice, baseline, local, output_gradient = self._build_qkv(
            sequence_length=sequence_length,
            head_dim=16,
            dtype=torch.bfloat16,
            seed=9192,
        )
        config = PreTrainedConfig()
        config._attn_implementation = "veomni_flex_attention_with_sp"
        attention_mask = torch.ones(1, sequence_length, device=device, dtype=torch.long)
        cu_seq_lens_q = torch.tensor([0, 3, 8, 12, 16], device=device, dtype=torch.int32)
        module = _FakeFlexAttentionModule().to(device)
        original_get_parallel_state = flex_backend.get_parallel_state
        try:
            flex_backend.get_parallel_state = lambda: SimpleNamespace(ulysses_enabled=False)
            baseline_block_mask = create_causal_mask(
                config=config,
                inputs_embeds=torch.empty(1, sequence_length, 1, device=device),
                attention_mask=attention_mask,
                past_key_values=None,
                cu_seq_lens_q=cu_seq_lens_q,
            )
            baseline_output, _ = flex_backend.flex_attention_forward(
                module,
                *baseline,
                baseline_block_mask,
                kernel_options={"BACKEND": "TRITON"},
            )

            flex_backend.get_parallel_state = lambda: SimpleNamespace(
                ulysses_enabled=True,
                ulysses_group=group,
                ulysses_size=world_size,
            )
            local_block_mask = create_causal_mask(
                config=config,
                inputs_embeds=torch.empty(1, sequence_length // world_size, 1, device=device),
                attention_mask=attention_mask,
                past_key_values=None,
                cu_seq_lens_q=cu_seq_lens_q,
            )
            local_output, _ = flex_backend.flex_attention_forward(
                module,
                *local,
                local_block_mask,
                kernel_options={"BACKEND": "TRITON"},
            )

            def index(value):
                return torch.tensor(value, device=device)

            assert baseline_block_mask.shape == local_block_mask.shape == (1, 1, sequence_length, sequence_length)
            assert not local_block_mask.mask_mod(index(0), index(0), index(8), index(7))
            assert local_block_mask.mask_mod(index(0), index(0), index(7), index(3))
            torch.testing.assert_close(sync_tensor(local_output, dim=1), baseline_output, rtol=3e-2, atol=3e-2)

            baseline_output.backward(output_gradient)
            local_output.backward(output_gradient[:, local_slice].contiguous())
            self._assert_qkv_gradients(local, baseline)
        finally:
            flex_backend.get_parallel_state = original_get_parallel_state


if __name__ == "__main__":
    assert not get_torch_device()._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    set_seed(seed=0, full_determinism=True)
    enable_high_precision_for_bf16()
    run_tests()
