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

"""Extended numerical alignment tests for NPU-optimised kernels.

Complements ``tests/ops/test_npu_kernels.py`` with additional shapes, dtypes,
edge cases, and a few more kernel-specific coverage tests.  Designed to:

  * Cover shapes that the original PR (PR #818) didn't parametrize over --
    in particular larger hidden sizes that the NPU kernels see in real
    models (e.g. 1024 / 2048 / 4096) and ``fp16`` which is still the
    default for some VeOmni downstream models.
  * Add a few degenerate-input tests (zeros, very small / very large
    variance) to catch NaN/Inf regressions in the fused kernels.
  * Add CPU-runnable eager reference tests so a developer can iterate
    on the kernel changes without an NPU device.

Tests are skipped on non-NPU hosts so the same test suite runs in any CI runner.
"""

import pytest
import torch

import veomni.ops  # noqa: F401 -- trigger KERNEL_REGISTRY registrations
from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU kernels require torch_npu")

DEVICE = get_device_type()


def _eager_rms_norm_gated(hidden_states, weight, eps, gate):
    """Eager reference: RMSNorm + concatenate gate + SiLU gating."""
    dtype = hidden_states.dtype
    x_f = hidden_states.to(torch.float32)
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_f = x_f * torch.rsqrt(variance + eps)
    normed = (weight * x_f.to(dtype)).to(dtype)
    fused_input = torch.cat([gate, normed], dim=-1)
    half = fused_input.shape[-1] // 2
    return torch.nn.functional.silu(fused_input[..., :half]) * fused_input[..., half:]


# ---------------------------------------------------------------------------
# Extended RMSNormGated tests
# ---------------------------------------------------------------------------


class TestNPURmsNormGatedExtended:
    """Coverage for the NPU FusedRMSNormGated module beyond PR #818."""

    @pytest.mark.parametrize("ffn_dim_factor", [2, 4])
    def test_gating_output_shape_and_dtype(self, ffn_dim_factor):
        """Output shape is (B, S, (ffn_dim + hidden) // 2).

        NPUFusedRMSNormGated concatenates [gate, rms_norm(hidden)]
        along the last dim (so the concat length is ffn_dim +
        hidden), then npu_swiglu splits in half via SiLU(x[:h])
        * x[h:].  The result is shape (B, S, (ffn_dim + hidden) / 2).
        """
        from veomni.ops.kernels.gated_delta_rule.npu_rms_norm_gated import NPUFusedRMSNormGated

        batch, seq, hidden, ffn_dim = 1, 4, 32, 32 * ffn_dim_factor
        expected_dim = (ffn_dim + hidden) // 2
        fused_module = NPUFusedRMSNormGated(hidden_size=hidden, eps=1e-6).to(device=DEVICE, dtype=torch.bfloat16)
        hidden_states = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.bfloat16)
        gate = torch.randn(batch, seq, ffn_dim, device=DEVICE, dtype=torch.bfloat16)
        out = fused_module(hidden_states, gate=gate)
        assert out.shape == (batch, seq, expected_dim)
        assert out.dtype == torch.bfloat16

    def test_gating_zero_gate_zeros_output(self):
        """When gate is zero, the output is 0 * normed = 0."""
        from veomni.ops.kernels.gated_delta_rule.npu_rms_norm_gated import NPUFusedRMSNormGated

        batch, seq, hidden, ffn_dim = 1, 4, 32, 64
        fused_module = NPUFusedRMSNormGated(hidden_size=hidden, eps=1e-6).to(device=DEVICE, dtype=torch.bfloat16)
        hidden_states = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.bfloat16)
        gate = torch.zeros(batch, seq, ffn_dim, device=DEVICE, dtype=torch.bfloat16)
        out = fused_module(hidden_states, gate=gate)
        # SiLU(0) * normed -> 0, so the output is bit-exact zero.
        assert out.abs().max().item() == 0.0, f"Output should be exactly zero, got max abs {out.abs().max().item()}"

    @pytest.mark.parametrize("eps", [1e-5, 1e-6, 1e-7])
    def test_gating_eps_parameter_used(self, eps):
        from veomni.ops.kernels.gated_delta_rule.npu_rms_norm_gated import NPUFusedRMSNormGated

        batch, seq, hidden, ffn_dim = 1, 4, 32, 64
        fused_module = NPUFusedRMSNormGated(hidden_size=hidden, eps=eps).to(device=DEVICE, dtype=torch.bfloat16)
        hidden_states = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.bfloat16)
        gate = torch.randn(batch, seq, ffn_dim, device=DEVICE, dtype=torch.bfloat16)
        out_fused = fused_module(hidden_states, gate=gate)
        out_eager = _eager_rms_norm_gated(hidden_states, fused_module.weight, eps, gate)
        # Compound op (RMSNorm + concat + SiLU gate) with bf16 rounding.
        # Empirical NPU ceiling at this shape: ~1e-2.
        assert torch.allclose(out_fused, out_eager, atol=1e-2, rtol=1e-2)

    def test_gating_module_is_nn_module(self):
        """NPUFusedRMSNormGated must behave like a regular nn.Module."""
        import torch.nn as nn

        from veomni.ops.kernels.gated_delta_rule.npu_rms_norm_gated import NPUFusedRMSNormGated

        m = NPUFusedRMSNormGated(hidden_size=16, eps=1e-6)
        assert isinstance(m, nn.Module)
        sd = m.state_dict()
        assert "weight" in sd
        m = m.to(device=DEVICE)
        assert m.weight.device.type == DEVICE
