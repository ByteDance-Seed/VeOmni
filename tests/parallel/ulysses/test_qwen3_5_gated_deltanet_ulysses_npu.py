# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Ascend NPU coverage for Qwen3.5 GatedDeltaNet under Ulysses SP.

The GPU test uses a physical batch and omits ``cu_seq_lens_q``. The NPU GDN
kernels exercise the production packed-varlen contract instead: physical batch
size is one and logical sequence boundaries are carried by ``cu_seq_lens_q``.
"""

from __future__ import annotations

import importlib
import os
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.tools.training_utils import make_npu_ops_config
from veomni.models.auto import _bind_veomni_ops
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.import_utils import is_torch_npu_available


pytestmark = [
    pytest.mark.qwen3_5_ulysses,
    pytest.mark.skipif(not is_torch_npu_available(), reason="Ascend NPU test"),
]

_PATCHED_MODULE = "veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_npu"
_DTYPE = torch.bfloat16


@dataclass
class _TinyQwen3_5Config:
    hidden_size: int = 128
    linear_num_value_heads: int = 4
    linear_num_key_heads: int = 2
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = _DTYPE


def _bind_npu_gdn_ops():
    """Bind the NPU module exactly like model construction, with AscendC GDN."""
    module = importlib.import_module(_PATCHED_MODULE)
    ops_config = make_npu_ops_config(
        "qwen3_5",
        rms_norm_gated_implementation="npu",
        causal_conv1d_implementation="npu",
        chunk_gated_delta_rule_implementation="npu_ascendc",
    )
    _bind_veomni_ops(module, ops_config)
    return module


def _assert_npu_forward_deterministic(
    layer: torch.nn.Module,
    inputs: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    *,
    repeats: int = 5,
) -> None:
    """Require bitwise-stable AscendC outputs for identical packed input."""
    with torch.no_grad():
        reference = layer(inputs, attention_mask=None, cu_seq_lens_q=cu_seqlens).detach()
        for _ in range(repeats - 1):
            actual = layer(inputs, attention_mask=None, cu_seq_lens_q=cu_seqlens)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def _broadcast_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        dist.broadcast(param.data, src=0)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)


def _all_reduce_parameter_grads(module: torch.nn.Module) -> None:
    """Sum grads in a rank-stable collective order and report asymmetric gaps."""
    world_size = dist.get_world_size()
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        grad_count = torch.tensor(int(param.grad is not None), device=param.device, dtype=torch.int32)
        dist.all_reduce(grad_count, op=dist.ReduceOp.SUM)
        present_ranks = int(grad_count.item())
        if present_ranks == 0:
            continue

        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        assert present_ranks == world_size, (
            f"Gradient for {name} is present on {present_ranks}/{world_size} ranks; "
            "Ulysses parameter usage must be rank-symmetric"
        )


def _make_cu_seqlens(sequence_lengths: tuple[int, ...]) -> torch.LongTensor:
    return torch.tensor((0, *torch.tensor(sequence_lengths).cumsum(0).tolist()), dtype=torch.long)


def _slice_mixed_qkv(
    mixed_qkv: torch.Tensor,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    sp_size: int,
    sp_rank: int,
) -> torch.Tensor:
    key_dim = num_k_heads * head_k_dim
    local_key_dim = num_k_heads // sp_size * head_k_dim
    local_value_dim = num_v_heads // sp_size * head_v_dim
    key_offset = sp_rank * local_key_dim
    value_offset = sp_rank * local_value_dim

    query = mixed_qkv[:, :, key_offset : key_offset + local_key_dim]
    key = mixed_qkv[:, :, key_dim + key_offset : key_dim + key_offset + local_key_dim]
    value = mixed_qkv[:, :, 2 * key_dim + value_offset : 2 * key_dim + value_offset + local_value_dim]
    return torch.cat((query, key, value), dim=-1)


def test_qwen3_5_gated_deltanet_npu_conv1d_slicing_matches_full() -> None:
    """NPU depthwise conv on each local head shard matches the full result."""
    if not get_torch_device().is_available():
        pytest.skip("Ascend NPU is not available")

    modeling = _bind_npu_gdn_ops()
    config = _TinyQwen3_5Config()
    device_type = get_device_type()
    layer = modeling.Qwen3_5GatedDeltaNet(config, layer_idx=0).to(device=device_type, dtype=_DTYPE)

    torch.manual_seed(42)
    seq_len = 65
    sp_size = 2
    mixed_qkv_full = torch.randn(1, seq_len, layer.conv_dim, device=device_type, dtype=_DTYPE)
    cu_seqlens = torch.tensor([0, seq_len], device=device_type, dtype=torch.long)
    weight_full = layer.conv1d.weight.squeeze(1)

    out_full = layer.causal_conv1d_fn(
        x=mixed_qkv_full,
        weight=weight_full,
        bias=None,
        activation=layer.activation,
        seq_idx=None,
        backend="triton",
        cu_seqlens=cu_seqlens,
    )[0]

    local_key_dim = config.linear_num_key_heads // sp_size * config.linear_key_head_dim
    local_value_dim = config.linear_num_value_heads // sp_size * config.linear_value_head_dim
    for sp_rank in range(sp_size):
        mixed_qkv_local = _slice_mixed_qkv(
            mixed_qkv_full,
            num_k_heads=config.linear_num_key_heads,
            num_v_heads=config.linear_num_value_heads,
            head_k_dim=config.linear_key_head_dim,
            head_v_dim=config.linear_value_head_dim,
            sp_size=sp_size,
            sp_rank=sp_rank,
        )
        weight_local = layer._get_local_conv1d_weight(sp_rank, local_key_dim, local_value_dim)
        out_local = layer.causal_conv1d_fn(
            x=mixed_qkv_local,
            weight=weight_local,
            bias=None,
            activation=layer.activation,
            seq_idx=None,
            backend="triton",
            cu_seqlens=cu_seqlens,
        )[0]
        out_expected = _slice_mixed_qkv(
            out_full,
            num_k_heads=config.linear_num_key_heads,
            num_v_heads=config.linear_num_value_heads,
            head_k_dim=config.linear_key_head_dim,
            head_v_dim=config.linear_value_head_dim,
            sp_size=sp_size,
            sp_rank=sp_rank,
        )
        torch.testing.assert_close(out_local, out_expected, rtol=0, atol=1e-3)


@pytest.mark.parametrize("sequence_lengths", [(128,), (65, 67)])
def test_qwen3_5_gated_deltanet_npu_forward_deterministic_no_sp(
    sequence_lengths: tuple[int, ...],
) -> None:
    """AscendC packed-varlen GDN is deterministic without sequence parallelism."""
    if not get_torch_device().is_available():
        pytest.skip("Ascend NPU is not available")

    modeling = _bind_npu_gdn_ops()
    config = _TinyQwen3_5Config()
    device_type = get_device_type()

    torch.manual_seed(42)
    layer = modeling.Qwen3_5GatedDeltaNet(config, layer_idx=0).to(device=device_type, dtype=_DTYPE)
    layer.train()
    inputs = torch.randn(
        1,
        sum(sequence_lengths),
        config.hidden_size,
        device=device_type,
        dtype=_DTYPE,
    )
    cu_seqlens = _make_cu_seqlens(sequence_lengths)

    no_sp_state = SimpleNamespace(ulysses_enabled=False)
    with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=no_sp_state):
        _assert_npu_forward_deterministic(layer, inputs, cu_seqlens)


def _run_gated_deltanet_npu_sp_fw_bw(
    rank: int,
    world_size: int,
    init_file: str,
    sequence_lengths: tuple[int, ...],
) -> None:
    device_type = get_device_type()
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state

    try:
        init_parallel_state(dp_size=1, ulysses_size=world_size, device_type=device_type)
        modeling = _bind_npu_gdn_ops()
        config = _TinyQwen3_5Config()

        torch.manual_seed(42)
        layer = modeling.Qwen3_5GatedDeltaNet(config, layer_idx=0).to(device=device_type, dtype=_DTYPE)
        _broadcast_module(layer)
        layer.train()

        total_seq_len = sum(sequence_lengths)
        assert total_seq_len % world_size == 0
        if rank == 0:
            full_input = torch.randn(
                1,
                total_seq_len,
                config.hidden_size,
                device=device_type,
                dtype=_DTYPE,
            )
        else:
            full_input = torch.empty(
                1,
                total_seq_len,
                config.hidden_size,
                device=device_type,
                dtype=_DTYPE,
            )
        dist.broadcast(full_input, src=0)
        cu_seqlens = _make_cu_seqlens(sequence_lengths)

        local_seq_len = total_seq_len // world_size
        local_slice = slice(rank * local_seq_len, (rank + 1) * local_seq_len)
        local_input = full_input[:, local_slice].contiguous().detach().requires_grad_(True)

        baseline_out = None
        baseline_input_grad = None
        baseline_param_grads = None
        if rank == 0:
            baseline_input = full_input.detach().clone().requires_grad_(True)
            no_sp_state = SimpleNamespace(ulysses_enabled=False)
            with patch(f"{_PATCHED_MODULE}.get_parallel_state", return_value=no_sp_state):
                baseline_out = layer(
                    baseline_input,
                    attention_mask=None,
                    cu_seq_lens_q=cu_seqlens,
                )
                baseline_out.mean().backward()
            baseline_input_grad = baseline_input.grad.detach().clone()
            baseline_param_grads = {
                name: param.grad.detach().clone() if param.grad is not None else None
                for name, param in layer.named_parameters()
            }
            baseline_out = baseline_out.detach()
            layer.zero_grad(set_to_none=True)

        dist.barrier()

        sp_out_local = layer(
            local_input,
            attention_mask=None,
            cu_seq_lens_q=cu_seqlens,
        )
        total_numel = float(total_seq_len * sp_out_local.shape[-1])
        (sp_out_local.sum() / total_numel).backward()

        _all_reduce_parameter_grads(layer)

        output_shards = [torch.empty_like(sp_out_local) for _ in range(world_size)]
        dist.all_gather(output_shards, sp_out_local.detach())
        sp_out_full = torch.cat(output_shards, dim=1)

        input_grad_shards = [torch.empty_like(local_input.grad) for _ in range(world_size)]
        dist.all_gather(input_grad_shards, local_input.grad.detach())
        sp_input_grad_full = torch.cat(input_grad_shards, dim=1)

        if rank == 0:
            torch.testing.assert_close(sp_out_full, baseline_out, rtol=5e-3, atol=5e-3)
            torch.testing.assert_close(sp_input_grad_full, baseline_input_grad, rtol=1e-2, atol=5e-4)
            for name, param in layer.named_parameters():
                baseline_grad = baseline_param_grads[name]
                if baseline_grad is None and param.grad is None:
                    continue
                assert baseline_grad is not None and param.grad is not None, f"Missing gradient for {name}"
                torch.testing.assert_close(
                    param.grad,
                    baseline_grad,
                    rtol=1e-2,
                    atol=5e-4,
                    msg=lambda msg, n=name: f"{msg}\nParameter gradient mismatch for {n}",
                )
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


def _run_gated_deltanet_npu_sp_determinism(
    rank: int,
    world_size: int,
    init_file: str,
    sequence_lengths: tuple[int, ...],
) -> None:
    device_type = get_device_type()
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_dist_comm_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state

    try:
        init_parallel_state(dp_size=1, ulysses_size=world_size, device_type=device_type)
        modeling = _bind_npu_gdn_ops()
        config = _TinyQwen3_5Config()

        torch.manual_seed(42)
        layer = modeling.Qwen3_5GatedDeltaNet(config, layer_idx=0).to(device=device_type, dtype=_DTYPE)
        _broadcast_module(layer)
        layer.train()

        total_seq_len = sum(sequence_lengths)
        assert total_seq_len % world_size == 0
        if rank == 0:
            full_input = torch.randn(
                1,
                total_seq_len,
                config.hidden_size,
                device=device_type,
                dtype=_DTYPE,
            )
        else:
            full_input = torch.empty(
                1,
                total_seq_len,
                config.hidden_size,
                device=device_type,
                dtype=_DTYPE,
            )
        dist.broadcast(full_input, src=0)

        local_seq_len = total_seq_len // world_size
        local_input = full_input[:, rank * local_seq_len : (rank + 1) * local_seq_len].contiguous()
        _assert_npu_forward_deterministic(layer, local_input, _make_cu_seqlens(sequence_lengths))
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.parametrize("sequence_lengths", [(128,), (65, 67)])
def test_qwen3_5_gated_deltanet_npu_sp_fw_bw_equivalence(sequence_lengths: tuple[int, ...]) -> None:
    """AscendC packed-varlen GDN under SP matches the non-SP NPU baseline."""
    world_size = 2
    if get_torch_device().device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} NPUs")

    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_gated_deltanet_npu_sp_fw_bw,
            args=(world_size, init_file, sequence_lengths),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.parametrize("sequence_lengths", [(128,), (65, 67)])
def test_qwen3_5_gated_deltanet_npu_forward_deterministic_sp(
    sequence_lengths: tuple[int, ...],
) -> None:
    """AscendC packed-varlen GDN is deterministic under two-way Ulysses SP."""
    world_size = 2
    if get_torch_device().device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} NPUs")

    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "init")
        mp.spawn(
            _run_gated_deltanet_npu_sp_determinism,
            args=(world_size, init_file, sequence_lengths),
            nprocs=world_size,
            join=True,
        )
