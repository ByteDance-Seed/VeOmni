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

"""SituGLU (``situ_and_mul``/kimi_k3) eager reference and registry contracts.

``situ_eager`` is the single definition of the SituGLU math shared by the
op registry's ``eager`` row and the Kimi K3 MoE expert loops; these tests
pin the formula, the packed-entry dtype contract, and the device gating
of the fused ``triton`` row.
"""

import pytest
import torch

from veomni.ops import OP_REGISTRY, resolve_op


def _reference(gate, up, beta=1.0, linear_beta=None):
    gate = gate.float()
    up = up.float()
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return situ_a * up


def test_situ_eager_matches_formula():
    from veomni.ops.kernels.situ_and_mul.kimi_k3.eager import situ_eager

    torch.manual_seed(7)
    gate = torch.randn(5, 8, 16)
    up = torch.randn(5, 8, 16)

    torch.testing.assert_close(situ_eager(gate, up), _reference(gate, up), rtol=0, atol=0)
    torch.testing.assert_close(situ_eager(gate, up, beta=0.5), _reference(gate, up, beta=0.5), rtol=0, atol=0)
    torch.testing.assert_close(
        situ_eager(gate, up, beta=0.5, linear_beta=2.0),
        _reference(gate, up, beta=0.5, linear_beta=2.0),
        rtol=0,
        atol=0,
    )
    # fp32 compute core: bf16 inputs are upcast inside, not truncated.
    torch.testing.assert_close(
        situ_eager(gate.bfloat16(), up.bfloat16()),
        _reference(gate, up),
        rtol=1e-2,
        atol=1e-2,
    )


def test_situ_and_mul_packed_entry_splits_and_casts():
    from veomni.ops.kernels.situ_and_mul.kimi_k3.eager import situ_and_mul

    torch.manual_seed(11)
    x = torch.randn(4, 24)
    d = x.shape[-1] // 2
    expected = _reference(x[..., :d], x[..., d:])
    out = situ_and_mul(x)
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, expected.to(x.dtype), rtol=0, atol=0)

    # bf16 in -> bf16 out (fp32 compute inside, from the bf16-rounded inputs).
    x_bf = x.bfloat16()
    out_bf = situ_and_mul(x_bf, beta=2.0, linear_beta=1.5)
    assert out_bf.dtype == torch.bfloat16
    x_bf_f = x_bf.float()
    d_bf = x_bf_f.shape[-1] // 2
    torch.testing.assert_close(
        out_bf.float(),
        _reference(x_bf_f[..., :d_bf], x_bf_f[..., d_bf:], beta=2.0, linear_beta=1.5).to(torch.bfloat16).float(),
        rtol=0,
        atol=0,
    )


def test_eager_registry_row_dispatches_to_packed_entry():
    entry = resolve_op("situ_and_mul", "kimi_k3", "eager")
    torch.manual_seed(13)
    x = torch.randn(3, 10)
    d = x.shape[-1] // 2
    expected = _reference(x[..., :d], x[..., d:]).to(x.dtype)
    torch.testing.assert_close(entry.wrapper(x), expected, rtol=0, atol=0)
    # kwargs flow through the row to the core.
    torch.testing.assert_close(
        entry.wrapper(x, beta=0.25, linear_beta=4.0),
        _reference(x[..., :d], x[..., d:], beta=0.25, linear_beta=4.0).to(x.dtype),
        rtol=0,
        atol=0,
    )


def test_triton_row_is_device_gated_to_npu():
    """The fused Triton-Ascend row must not resolve off Ascend NPUs."""
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        pytest.skip("contract is the non-npu raise; npu hosts resolve the row")
    with pytest.raises(RuntimeError, match="not registered for device"):
        resolve_op("situ_and_mul", "kimi_k3", "triton")


def test_default_resolution_prefers_the_platform_row():
    entry = resolve_op("situ_and_mul", "kimi_k3", None)
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        # The fused Triton row is the NPU default only while triton is
        # importable (unit-CI images install triton-ascend explicitly);
        # otherwise the registry resolves to the eager reference by design
        # (OpRegistry.default_impl), so track the row's availability instead
        # of assuming it.
        if "triton" in OP_REGISTRY.list_available("situ_and_mul", "kimi_k3"):
            assert entry.impl == "triton"
        else:
            assert entry.impl == "eager"
    else:
        assert entry.impl == "eager"
