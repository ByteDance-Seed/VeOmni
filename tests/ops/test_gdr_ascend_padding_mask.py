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

"""Numerical regression for padded GDR dot tiles on Ascend 910B4."""

import os

import pytest
import torch

from veomni.utils.device import IS_NPU_AVAILABLE


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="GDR padding-mask regression requires an NPU")

_BOUNDARIES = (0, 31, 73, 127)
_CHUNK_SIZE = 64
_HEADS = 4
_KEY_DIM = 128
_SUPPORTED_DEVICE_NAMES = frozenset({"Ascend910B4", "Ascend910B4-1"})


def _cpu_elementwise_reference(k, g_cumsum, beta):
    """Build the strict-lower GDR matrix without using matmul or dot."""
    k_cpu = k.detach().float().cpu()
    g_cpu = g_cumsum.detach().float().cpu()
    beta_cpu = beta.detach().float().cpu()
    reference = torch.zeros(
        (1, _BOUNDARIES[-1], _HEADS, _CHUNK_SIZE),
        dtype=torch.float32,
    )

    for start, end in zip(_BOUNDARIES, _BOUNDARIES[1:]):
        length = end - start
        for row in range(1, length):
            row_k = k_cpu[0, start + row]
            prior_k = k_cpu[0, start : start + row]
            dot_values = (prior_k * row_k.unsqueeze(0)).sum(dim=-1)
            g_difference = g_cpu[0, start + row].unsqueeze(0) - g_cpu[0, start : start + row]
            g_scale = g_difference.clamp(min=-50.0, max=50.0)
            values = dot_values * g_scale.exp() * beta_cpu[0, start + row].unsqueeze(0)
            reference[0, start + row, :, :row] = values.transpose(0, 1)

    return reference


def _assert_finite_strict_lower(output, *, poison):
    assert torch.isfinite(output).all(), f"{poison} destination produced non-finite values"
    for start, end in zip(_BOUNDARIES, _BOUNDARIES[1:]):
        for row in range(end - start):
            diagonal_upper_and_padding = output[0, start + row, :, row:]
            assert torch.count_nonzero(diagonal_upper_and_padding).item() == 0, (
                f"{poison} destination left a non-zero value outside the strict lower triangle "
                f"for segment_start={start}, row={row}"
            )


def test_varlen_padded_dot_mask_is_finite_and_strictly_lower_triangular():
    import torch_npu  # noqa: F401 - register the torch.npu device namespace

    device_index = 0
    torch.npu.set_device(device_index)
    device_name = torch.npu.get_device_name(device_index)
    if device_name not in _SUPPORTED_DEVICE_NAMES:
        pytest.skip(f"requires Ascend 910B4, got {device_name!r}")
    assert os.environ.get("TRITON_ASCEND_ARCH") == "Ascend910B4", (
        "set TRITON_ASCEND_ARCH=Ascend910B4 before importing Triton on Ascend 910B4"
    )

    from veomni.ops.kernels.gated_delta_rule._ascend.triton.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd_kernel,
    )
    from veomni.ops.kernels.gated_delta_rule._ascend.triton.cumsum import chunk_local_cumsum
    from veomni.ops.kernels.gated_delta_rule._ascend.triton.utils import prepare_chunk_indices

    torch.manual_seed(42)
    device = torch.device(f"npu:{device_index}")
    tokens = _BOUNDARIES[-1]
    k = torch.randn(1, tokens, _HEADS, _KEY_DIM, device=device, dtype=torch.bfloat16)
    k_float = k.float()
    k = (k_float * torch.rsqrt((k_float * k_float).sum(dim=-1, keepdim=True) + 1e-6)).to(k.dtype)
    g = -torch.nn.functional.softplus(torch.randn(1, tokens, _HEADS, device=device, dtype=torch.float32))
    beta = torch.rand(1, tokens, _HEADS, device=device, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor(_BOUNDARIES, dtype=torch.long, device=device)
    g_cumsum = chunk_local_cumsum(
        g,
        chunk_size=_CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        head_first=False,
    )
    chunk_indices = prepare_chunk_indices(cu_seqlens, _CHUNK_SIZE)
    transposed_g = g_cumsum.transpose(1, 2).contiguous()
    transposed_beta = beta.transpose(1, 2).contiguous()

    outputs = []
    for poison, fill_value in (("finite sentinel", 12345.0), ("NaN", float("nan"))):
        output = torch.full(
            (1, tokens, _HEADS, _CHUNK_SIZE),
            fill_value,
            device=device,
            dtype=torch.float32,
        )
        chunk_scaled_dot_kkt_fwd_kernel[(24,)](
            k=k,
            g=transposed_g,
            beta=transposed_beta,
            A=output,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=tokens,
            H=_HEADS,
            K=_KEY_DIM,
            BT=_CHUNK_SIZE,
            BK=128,
            NT=len(chunk_indices),
            B=1,
            TOTAL_TASKS=len(chunk_indices),
        )
        torch.npu.synchronize()
        output_cpu = output.detach().float().cpu()
        _assert_finite_strict_lower(output_cpu, poison=poison)
        outputs.append(output_cpu)

    reference = _cpu_elementwise_reference(k, g_cumsum, beta)
    torch.testing.assert_close(outputs[0], reference, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(outputs[1], reference, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0.0, rtol=0.0)
