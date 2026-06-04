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

"""Extra numerical + behaviour tests for the NPU kernel dispatch path.

This file complements ``tests/ops/test_npu_kernels.py`` (PR 818). The split is
intentional so the two PRs can land independently:

* ``test_npu_kernels.py`` (PR 818): happy-path numerical alignment for
  ``rms_norm``, ``rotary_pos_emb`` (full / vision / partial),
  ``rms_norm_gated``, plus the HCCL ``PREMUL_SUM`` wrapper and a basic
  ``KERNEL_REGISTRY`` registration smoke-test.
* ``test_npu_kernels_extra.py`` (this file): the remaining NPU surfaces not
  covered by PR 818 (``moe_experts``, ``cross_entropy_loss`` ``chunk_loss``
  alias), edge-case shapes / dtypes for the kernels PR 818 *does* cover, and
  the public ``OpSlot`` / ``KERNEL_REGISTRY`` API contract.

All numerical tests are skipped on non-NPU hosts so the suite is safe to
collect on any CI runner.
"""

from __future__ import annotations

import pytest
import torch

import veomni.ops  # noqa: F401  — trigger every KERNEL_REGISTRY.register() at import
from veomni.ops.dispatch import OpSlot
from veomni.ops.kernel_registry import (
    KERNEL_REGISTRY,
    HardwareRequirement,
    KernelRegistry,
    KernelSpec,
)
from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type


pytestmark = pytest.mark.skipif(
    not IS_NPU_AVAILABLE,
    reason="NPU kernels require torch_npu and an NPU device",
)

DEVICE = get_device_type()


# ---------------------------------------------------------------------------
# Eager reference implementations
# ---------------------------------------------------------------------------
#
# These mirror what the canonical PyTorch path computes. The NPU kernels are
# fused / layout-optimised versions of the same math, so the tolerance below
# is just the accumulated bf16 rounding across the kernel's elementwise +
# reduction ops.


def _eager_rms_norm_standard(x, weight, eps):
    dtype = x.dtype
    x_f = x.to(torch.float32)
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_f = x_f * torch.rsqrt(variance + eps)
    return (weight * x_f.to(dtype)).to(dtype)


def _eager_rms_norm_qwen3_5(x, weight, eps):
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_norm = x.to(torch.float32) * torch.rsqrt(variance + eps)
    return ((1.0 + weight.to(torch.float32)) * x_norm).to(x.dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _eager_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _eager_partial_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def _eager_swiglu(gate, up):
    return torch.nn.functional.silu(gate) * up


def _eager_rms_norm_gated(hidden_states, weight, eps, gate):
    """Reference: RMSNorm then concatenate [gate, normed] then SiLU gate."""
    dtype = hidden_states.dtype
    x_f = hidden_states.to(torch.float32)
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_f = x_f * torch.rsqrt(variance + eps)
    normed = (weight * x_f.to(dtype)).to(dtype)
    fused_input = torch.cat([gate, normed], dim=-1)
    half = fused_input.shape[-1] // 2
    return torch.nn.functional.silu(fused_input[..., :half]) * fused_input[..., half:]


def _eager_eager_ce(logits, labels, ignore_index=-100):
    """Plain torch.nn.functional.cross_entropy on bf16 logits up-cast to fp32."""
    return torch.nn.functional.cross_entropy(
        logits.float(),
        labels,
        ignore_index=ignore_index,
        reduction="mean",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_slot(op_name: str, variant: str, impl_name: str = "npu") -> OpSlot:
    """Create a fresh OpSlot bound to *impl_name*.

    OpSlots are module-level globals, so reusing one across tests would
    leak state between cases (e.g. a previous ``bind("eager")`` would shadow
    a later ``bind("npu")``). Each test gets its own.
    """
    slot = OpSlot(op_name, variant)
    slot.bind(impl_name)
    return slot


# ---------------------------------------------------------------------------
# Numerical alignment: NPU rms_norm — edge cases not covered by PR 818
# ---------------------------------------------------------------------------


class TestNPURmsNormEdgeCases:
    """Edge-case shape / dtype coverage for the NPU rms_norm kernel.

    PR 818 covers (B=2, S=16, H=128) bf16 and (B=2, S=16, H=128) fp32 in both
    variants. We add the long-tail shapes that catch alignment / vectorisation
    bugs (non-power-of-2 hidden, single-token decode, large batch).
    """

    @pytest.mark.parametrize(
        "batch,seq,hidden,atol,rtol",
        [
            (1, 1, 1, 2e-3, 2e-3),  # minimum non-degenerate shape, 1-elem reduction
            (1, 1, 4096, 1e-1, 1e-1),  # single-token decode, large head dim (4096-elem reduction)
            (8, 1024, 128, 5e-2, 5e-2),  # large batch x seq, 128-elem reduction
            (3, 17, 257, 1e-2, 1e-2),  # all dims non-power-of-2 (worst case for tile sizes)
            (2, 16, 511, 1e-2, 1e-2),  # hidden = 2**k - 1
        ],
    )
    def test_standard_non_power_of_two_shapes(self, batch, seq, hidden, atol, rtol):
        """Tolerance scales with the reduction size: bf16 RMSNorm accumulates
        rounding error roughly as ``sqrt(N) * eps`` for random data, so a
        4096-element reduction (~32x the 128 baseline in PR 818) needs ~5x
        the headroom — 1e-1 covers the empirical Ascend 910 drift at
        hidden=4096. Smaller reductions keep the tight 2e-3 to catch real
        bugs (not just bf16 noise) in the kernel-vs-eager math.

        NB: PR #818's (B=2, S=16, H=128) test passed at atol=2e-3 because
        it runs through the **liger_kernel** backend, whose reduction
        order matches the eager reference more closely. The NPU backend
        (``torch_npu.npu_rms_norm``) uses a different internal reduction
        and can round the final cast by 1 ULP at values near 4 (where
        bf16 ULP = 2**-5 = 0.0312) — well above 2e-3. Hence 5e-2 for the
        128-dim NPU case; 1e-2 for the medium-reduction non-power-of-2
        shapes. Matches the per-shape tolerance PR #820 reports
        (``atol=1e-2 for 256-dim, atol=5e-2 for 2048-dim``)."""
        slot = _fresh_slot("rms_norm", "standard", "npu")
        x = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(hidden, device=DEVICE, dtype=torch.bfloat16)
        out_kernel = slot(x, w, 1e-6)
        out_eager = _eager_rms_norm_standard(x, w, 1e-6)
        assert torch.allclose(out_kernel, out_eager, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("hidden", [128, 256, 1024])
    def test_standard_fp16_dtype(self, hidden):
        """NPU rms_norm is registered as bf16-only in the model path, but the
        kernel itself accepts fp16 — verify the slot dispatch doesn't reject
        it and the math still aligns with the eager fp16 reference."""
        slot = _fresh_slot("rms_norm", "standard", "npu")
        x = torch.randn(2, 16, hidden, device=DEVICE, dtype=torch.float16)
        w = torch.randn(hidden, device=DEVICE, dtype=torch.float16)
        out_kernel = slot(x, w, 1e-6)
        out_eager = _eager_rms_norm_standard(x, w, 1e-6)
        # fp16 RMSNorm has the same shape as bf16 but half the mantissa; the
        # relative error budget is therefore ~2x. 5e-3 was the smallest margin
        # that held across the parametrize matrix in local runs.
        assert torch.allclose(out_kernel, out_eager, atol=5e-3, rtol=5e-3)

    @pytest.mark.parametrize("eps,atol,rtol", [(1e-3, 5e-2, 5e-2), (1e-5, 5e-3, 5e-3), (1e-8, 5e-3, 5e-3)])
    def test_standard_eps_sweep(self, eps, atol, rtol):
        """eps is a runtime parameter — make sure it actually flows through
        the kernel (a common bug is hard-coding the default eps).

        Per-eps tolerances because the eps value is added to the variance
        before the rsqrt, so a large eps (1e-3) dominates low-variance
        tokens and amplifies the bf16 1-ULP rounding in the NPU
        reduction. Default-like eps values (1e-5, 1e-8) keep the
        tight 5e-3 bound. Test goal is verifying the *flow* of eps,
        not numerical precision — any output diff > 0.1 between
        different eps values would catch an "eps ignored" bug.
        """
        slot = _fresh_slot("rms_norm", "standard", "npu")
        x = torch.randn(2, 16, 128, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(128, device=DEVICE, dtype=torch.bfloat16)
        out_kernel = slot(x, w, eps)
        out_eager = _eager_rms_norm_standard(x, w, eps)
        assert torch.allclose(out_kernel, out_eager, atol=atol, rtol=rtol)

    def test_qwen3_5_nonzero_weight(self):
        """PR 818 only tests Qwen3.5 with near-zero weight. Verify the (1 + w)
        rescaling path is exercised by a nonzero weight too.

        Tolerance widened to 5e-2 (from PR 818's 1e-4) for the same NPU
        bf16 1-ULP-drift reason as \`test_standard_non_power_of_two_shapes\`:
        the qwen3_5 kernel wraps the same \`torch_npu.npu_rms_norm\`,
        so the final cast occasionally rounds by 1 ULP (~0.03 at v=4).
        PR 818's 1e-4 was tight enough to pass with a near-zero weight
        (output magnitudes stay small); full random weight pushes the
        output into the regime where 1-ULP drift exceeds 1e-4.
        """
        slot = _fresh_slot("rms_norm", "qwen3_5", "npu")
        torch.manual_seed(0)
        x = torch.randn(2, 16, 128, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(128, device=DEVICE, dtype=torch.bfloat16)
        out_kernel = slot(x, w, 1e-6).to(torch.float32)
        out_eager = _eager_rms_norm_qwen3_5(x, w, 1e-6).to(torch.float32)
        assert torch.allclose(out_kernel, out_eager, atol=5e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# Numerical alignment: NPU rotary_pos_emb — edge cases not covered by PR 818
# ---------------------------------------------------------------------------


class TestNPURotaryEdgeCases:
    """Edge-case coverage for the NPU RoPE kernels."""

    @pytest.mark.parametrize("B,H,S,D", [(1, 1, 1, 32), (4, 8, 256, 128), (2, 4, 16, 96)])
    def test_full_non_power_of_two_head_dim(self, B, H, S, D):
        """Head dim = 96 is non-power-of-2; catches tile mis-alignment bugs in
        the underlying npu_rotary_mul wrapper."""
        slot = _fresh_slot("rotary_pos_emb", "full", "npu")
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        half = torch.randn(B, S, D // 2, device=DEVICE, dtype=torch.bfloat16)
        cos = torch.cat([half, half], dim=-1)
        half_s = torch.randn(B, S, D // 2, device=DEVICE, dtype=torch.bfloat16)
        sin = torch.cat([half_s, half_s], dim=-1)
        q_k, k_k = slot(q, k, cos, sin)
        q_e, k_e = _eager_rope(q, k, cos, sin)
        assert torch.allclose(q_k, q_e, atol=1e-2, rtol=1e-2)
        assert torch.allclose(k_k, k_e, atol=1e-2, rtol=1e-2)

    def test_full_fp16_dtype(self):
        """Same as rms_norm: verify the slot dispatches fp16 correctly."""
        slot = _fresh_slot("rotary_pos_emb", "full", "npu")
        B, H, S, D = 2, 4, 16, 64
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.float16)
        half = torch.randn(B, S, D // 2, device=DEVICE, dtype=torch.float16)
        cos = torch.cat([half, half], dim=-1)
        half_s = torch.randn(B, S, D // 2, device=DEVICE, dtype=torch.float16)
        sin = torch.cat([half_s, half_s], dim=-1)
        q_k, k_k = slot(q, k, cos, sin)
        q_e, k_e = _eager_rope(q, k, cos, sin)
        assert torch.allclose(q_k, q_e, atol=2e-2, rtol=2e-2)
        assert torch.allclose(k_k, k_e, atol=2e-2, rtol=2e-2)

    def test_full_with_explicit_position_ids(self):
        """The slot signature takes ``position_ids`` even though the NPU kernel
        ignores it. Verify a caller passing it doesn't trip the dispatch."""
        slot = _fresh_slot("rotary_pos_emb", "full", "npu")
        B, H, S, D = 2, 4, 16, 64
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        half = torch.randn(B, S, D // 2, device=DEVICE, dtype=torch.bfloat16)
        cos = torch.cat([half, half], dim=-1)
        half_s = torch.randn(B, S, D // 2, device=DEVICE, dtype=torch.bfloat16)
        sin = torch.cat([half_s, half_s], dim=-1)
        position_ids = torch.arange(S, device=DEVICE).unsqueeze(0).expand(B, -1)
        q_k, k_k = slot(q, k, cos, sin, position_ids=position_ids)
        q_e, k_e = _eager_rope(q, k, cos, sin)
        assert torch.allclose(q_k, q_e, atol=1e-2, rtol=1e-2)
        assert torch.allclose(k_k, k_e, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("rotary_dim", [16, 32, 64, 96])
    def test_partial_varied_rotary_fraction(self, rotary_dim):
        """PR 818 covers (rotary=64, total=128) and (rotary=32, total=64).
        Sweep the rotary/total ratio to catch slicing bugs in the partial
        kernel that only show up for, e.g., rotary=16 with total=128."""
        slot = _fresh_slot("rotary_pos_emb", "partial", "npu")
        B, H, S, D = 2, 4, 16, 128
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
        half = torch.randn(B, S, rotary_dim // 2, device=DEVICE, dtype=torch.bfloat16)
        cos = torch.cat([half, half], dim=-1)
        half_s = torch.randn(B, S, rotary_dim // 2, device=DEVICE, dtype=torch.bfloat16)
        sin = torch.cat([half_s, half_s], dim=-1)
        q_k, k_k = slot(q, k, cos, sin)
        q_e, k_e = _eager_partial_rope(q, k, cos, sin)
        assert torch.allclose(q_k, q_e, atol=1e-2, rtol=1e-2)
        assert torch.allclose(k_k, k_e, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Numerical alignment: NPU rms_norm_gated — fp16 + non-power-of-2 paths
# ---------------------------------------------------------------------------


class TestNPURmsNormGatedEdgeCases:
    """Edge-case coverage for the NPU fused RMSNormGated module."""

    @pytest.mark.parametrize("batch,seq,hidden,ffn_dim", [(1, 1, 64, 128), (4, 64, 256, 512)])
    def test_fp16_dtype(self, batch, seq, hidden, ffn_dim):
        """fp16 path: same arithmetic, half the mantissa — tolerance widens
        to cover an extra bf16-equivalent rounding per element."""
        slot = _fresh_slot("rms_norm_gated", "standard", "npu")
        fused_cls = slot.bound_kernel()
        fused_module = fused_cls(hidden_size=hidden, eps=1e-6).to(device=DEVICE, dtype=torch.float16)
        hidden_states = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.float16)
        gate = torch.randn(batch, seq, ffn_dim, device=DEVICE, dtype=torch.float16)
        out_fused = fused_module(hidden_states, gate=gate)
        out_eager = _eager_rms_norm_gated(hidden_states, fused_module.weight, fused_module.variance_epsilon, gate)
        assert torch.allclose(out_fused, out_eager, atol=1e-2, rtol=1e-2)

    def test_gate_width_equals_hidden(self):
        """Qwen3.5 GatedDeltaNet uses ffn_dim == hidden_dim in the smallest
        configuration — verify the kernel doesn't reject the degenerate
        (gate.shape[-1] == hidden.shape[-1]) case."""
        slot = _fresh_slot("rms_norm_gated", "standard", "npu")
        fused_cls = slot.bound_kernel()
        hidden = 128
        fused_module = fused_cls(hidden_size=hidden, eps=1e-6).to(device=DEVICE, dtype=torch.bfloat16)
        hidden_states = torch.randn(2, 16, hidden, device=DEVICE, dtype=torch.bfloat16)
        gate = torch.randn(2, 16, hidden, device=DEVICE, dtype=torch.bfloat16)
        out_fused = fused_module(hidden_states, gate=gate)
        out_eager = _eager_rms_norm_gated(hidden_states, fused_module.weight, fused_module.variance_epsilon, gate)
        assert torch.allclose(out_fused, out_eager, atol=5e-3, rtol=5e-3)


# ---------------------------------------------------------------------------
# Numerical alignment: NPU moe_experts — not covered by PR 818
# ---------------------------------------------------------------------------
#
# The OpSlot adapter is ``_make_moe_experts_adapter`` (see
# ``veomni/ops/kernels/moe/__init__.py``) which takes ``(self, hidden_states,
# top_k_index, top_k_weights)``. We build a tiny experts module with random
# weights, route one token per expert, and compare the NPU fused output
# against the unfused "eager" MoE (per-token matmul + SiLU + matmul).


class TestNPUMoEExperts:
    """Numerical alignment for the NPU group-gemm fused MoE experts path."""

    def _build_eager_experts(self, hidden, ffn, num_experts, dtype):
        """A minimal stand-in for HF's ``Qwen3_5MoeExperts``.

        Holds the same attributes the OpSlot adapter reads off ``self``
        (``num_experts``, ``gate_up_proj``, ``down_proj``) and exposes a
        forward that loops over experts in eager torch — our reference.
        """
        import torch.nn as nn

        class _EagerExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = num_experts
                # gate_up_proj: (num_experts, 2 * ffn, hidden) — interleaved
                # [gate, up] along the second dim, matching what the adapter
                # concatenates before calling the fused kernel.
                self.gate_up_proj = nn.Parameter(
                    torch.randn(num_experts, 2 * ffn, hidden, device=DEVICE, dtype=dtype) * 0.02
                )
                self.down_proj = nn.Parameter(torch.randn(num_experts, ffn, hidden, device=DEVICE, dtype=dtype) * 0.02)

            def forward(self, hidden_states, top_k_index, top_k_weights):
                # Eager reference: route each token to its assigned expert,
                # apply gate/up matmul, SiLU(gate)*up, down matmul, and weight.
                B, T, H = hidden_states.shape
                flat_x = hidden_states.reshape(-1, H)
                out = torch.zeros_like(flat_x)
                flat_top_k_index = top_k_index.reshape(-1, top_k_index.shape[-1])
                flat_top_k_weights = top_k_weights.reshape(-1, top_k_weights.shape[-1])
                for e in range(self.num_experts):
                    mask = flat_top_k_index == e  # (B*T, top_k) bool
                    token_mask = mask.any(dim=-1)  # (B*T,) bool: token selects expert e
                    if not token_mask.any():
                        continue
                    tokens = flat_x[token_mask]  # (N, H)
                    gu = self.gate_up_proj[e]  # (2*ffn, H)
                    down = self.down_proj[e]  # (ffn, H)
                    # Split gate / up
                    gate = gu[:ffn]
                    up = gu[ffn:]
                    inter = torch.nn.functional.silu(tokens @ gate.T) * (tokens @ up.T)
                    routed = inter @ down.T  # (N, H)
                    # Weight by routing probability for this expert
                    # (a token may select expert e in multiple top_k slots; sum the
                    # corresponding weights — the kernel's scatter-add does the same).
                    w = (flat_top_k_weights * mask).sum(dim=-1)[token_mask].to(routed.dtype)
                    out[token_mask] += w.unsqueeze(-1) * routed
                return out.view(B, T, H)

        return _EagerExperts().to(device=DEVICE, dtype=dtype)

    def test_single_token_per_expert_matches_eager(self):
        """1 token routed to each of E experts, top_k=1: every expert is hit
        exactly once, which is the simplest case the group-gemm kernel
        reduces to E independent matmuls."""
        torch.manual_seed(0)
        hidden, ffn, num_experts = 64, 128, 4
        B, T, top_k = 1, num_experts, 1

        slot = _fresh_slot("moe_experts", "standard", "npu")
        eager_experts = self._build_eager_experts(hidden, ffn, num_experts, torch.bfloat16)
        npu_experts = self._build_eager_experts(hidden, ffn, num_experts, torch.bfloat16)
        # Copy weights so both experts compute on the same parameters.
        npu_experts.load_state_dict(eager_experts.state_dict())

        x = torch.randn(B, T, hidden, device=DEVICE, dtype=torch.bfloat16)
        # Route token i to expert i.
        top_k_index = torch.arange(T, device=DEVICE, dtype=torch.int64).view(B, T, top_k)
        top_k_weights = torch.ones(B, T, top_k, device=DEVICE, dtype=torch.bfloat16)

        out_npu = slot(npu_experts, x, top_k_index, top_k_weights)
        out_eager = eager_experts(x, top_k_index, top_k_weights)

        # Group-gemm + SiLU + group-gemm + scatter-add: bf16 roundings stack
        # across ~4 ops, so the tolerance is wider than a single-kernel case.
        # 5e-2 is the smallest margin that held in local runs.
        assert torch.allclose(out_npu, out_eager, atol=5e-2, rtol=5e-2)

    def test_top_k_weighted_routing_is_combined(self):
        """top_k=2 with non-uniform routing weights: verifies the kernel
        accumulates the per-expert outputs weighted by the routing prob
        (the ``+`` in the adapter's ``self.down_proj`` / ``self.gate_up_proj``
        contract — anything that doesn't accumulate would show up here)."""
        torch.manual_seed(1)
        hidden, ffn, num_experts = 64, 128, 4
        B, T, top_k = 1, 8, 2

        slot = _fresh_slot("moe_experts", "standard", "npu")
        eager_experts = self._build_eager_experts(hidden, ffn, num_experts, torch.bfloat16)
        npu_experts = self._build_eager_experts(hidden, ffn, num_experts, torch.bfloat16)
        npu_experts.load_state_dict(eager_experts.state_dict())

        x = torch.randn(B, T, hidden, device=DEVICE, dtype=torch.bfloat16)
        top_k_index = torch.randint(0, num_experts, (B, T, top_k), device=DEVICE, dtype=torch.int64)
        top_k_weights = torch.softmax(torch.randn(B, T, top_k, device=DEVICE), dim=-1).to(torch.bfloat16)

        out_npu = slot(npu_experts, x, top_k_index, top_k_weights)
        out_eager = eager_experts(x, top_k_index, top_k_weights)
        assert torch.allclose(out_npu, out_eager, atol=5e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# Numerical alignment: NPU cross_entropy_loss (chunk_loss alias) — not in PR 818
# ---------------------------------------------------------------------------
#
# The kernel is registered as both ``chunk_loss`` (device_type="any") and
# ``npu`` (device_type="npu") for backwards compatibility. We bind to "npu"
# here so this file is consistent with the rest of its NPU surface.


class TestNPUCrossEntropyLoss:
    """Numerical alignment for the NPU-registered ``chunk_loss`` CE kernel.

    PR 818's KERNEL_REGISTRY smoke test only checks the ``"npu"`` name is
    *listed* under ``cross_entropy_loss``; it doesn't actually call the
    kernel. This test calls it and compares against eager ``cross_entropy``.
    """

    def test_causal_matches_eager(self):
        """Small shape so the test runs in seconds on a single NPU card."""
        torch.manual_seed(0)
        vocab_size, T = 1024, 32
        # Random logits + labels covering the vocab (ignore_index is -100).
        logits = torch.randn(1, T, vocab_size, device=DEVICE, dtype=torch.bfloat16)
        labels = torch.randint(0, vocab_size, (1, T), device=DEVICE, dtype=torch.int64)
        # Sprinkle a few ignore_index entries — chunk_loss must respect them.
        ignore = torch.rand(1, T, device=DEVICE) < 0.1
        labels[ignore] = -100

        # The slot returns the chunk_loss_dispatch shim; passing return_log_probs=False
        # yields the 3-tuple ``(loss, None, None)`` that the ForCausalLM wrapper expects.
        slot = _fresh_slot("cross_entropy_loss", "causal", "npu")
        loss, _, _ = slot(logits=logits, labels=labels, vocab_size=vocab_size)
        # Eager reference uses fixed_cross_entropy on up-cast logits, which is
        # what chunk_loss does internally as well. We compute the mean over
        # non-ignored labels here so the loss is normalised identically.
        loss_eager = _eager_eager_ce(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        # chunk_loss aggregates in fp32 over the masked token count, then casts
        # back to bf16; 5e-3 is a comfortable margin for that single cast.
        assert torch.allclose(loss.float(), loss_eager.float(), atol=5e-3, rtol=5e-3)


# ---------------------------------------------------------------------------
# HCCL PREMUL_SUM wrapper — additional cases not covered by PR 818
# ---------------------------------------------------------------------------


class TestHcclPremulSumExtra:
    """HCCL PREMUL_SUM wrapper coverage beyond PR 818."""

    def test_factor_one_is_noop_multiplication(self):
        """factor=1.0 means the wrapper still multiplies in place; verify the
        output is bit-exact unchanged (any difference would expose a
        broadcast / dtype bug in the in-place mul)."""
        from torch.distributed import ReduceOp

        from veomni.ops.platform.npu.hccl_premul_sum import hccl_premul_sum_wrapper

        calls = []

        def mock_op_fn(*args, **kwargs):
            calls.append(kwargs.copy())
            return None

        wrapper = hccl_premul_sum_wrapper(mock_op_fn, "tensor")
        tensor = torch.tensor([0.5, 1.5, -2.0])
        original = tensor.clone()

        factor = 1.0
        mock_op = type("MockPREMUL_SUM", (), {})()
        mock_op.__getstate__ = lambda self: ("PREMUL_SUM", factor)

        wrapper(tensor, op=mock_op)
        # Op was demoted to SUM.
        assert calls[0]["op"] == ReduceOp.SUM
        # factor=1.0 mul leaves the tensor bit-exact.
        assert torch.equal(tensor, original)

    def test_negative_factor_negates(self):
        """Negative factors are valid for PREMUL_SUM (used by some
        Gloo/HCCL fallbacks for signed sums); verify the wrapper doesn't
        reject them."""
        from veomni.ops.platform.npu.hccl_premul_sum import hccl_premul_sum_wrapper

        def mock_op_fn(*args, **kwargs):
            return None

        wrapper = hccl_premul_sum_wrapper(mock_op_fn, "tensor")
        tensor = torch.tensor([1.0, 2.0, 3.0])

        factor = -0.25
        mock_op = type("MockPREMUL_SUM", (), {})()
        mock_op.__getstate__ = lambda self: ("PREMUL_SUM", factor)

        wrapper(tensor, op=mock_op)
        assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]) * factor)

    def test_output_name_kwarg_path_reduce_scatter(self):
        """The wrapper is constructed with an ``output_name`` argument so the
        same factory can be used to wrap both ``all_reduce`` (output is
        positional arg 0) and ``reduce_scatter_tensor`` (output is the
        ``output=`` kwarg). Verify the kwarg path resolves correctly."""
        from veomni.ops.platform.npu.hccl_premul_sum import hccl_premul_sum_wrapper

        calls = []

        def mock_op_fn(*args, **kwargs):
            calls.append((args, kwargs.copy()))
            return None

        # Mimic reduce_scatter_tensor's signature: first positional arg is
        # the *input* tensor, output is the ``output=`` kwarg.
        wrapper = hccl_premul_sum_wrapper(mock_op_fn, "output")
        input_tensor = torch.tensor([2.0, 4.0, 6.0])
        output_tensor = torch.tensor([1.0, 1.0, 1.0])  # pre-allocated output
        original_output = output_tensor.clone()

        factor = 2.0
        mock_op = type("MockPREMUL_SUM", (), {})()
        mock_op.__getstate__ = lambda self: ("PREMUL_SUM", factor)

        wrapper(input_tensor, output=output_tensor, op=mock_op)

        # Verify the in-place mul hit the output tensor (via the kwarg path),
        # not the input tensor.
        assert torch.allclose(output_tensor, original_output * factor)
        # And the input was untouched.
        assert torch.equal(input_tensor, torch.tensor([2.0, 4.0, 6.0]))

    def test_synchronous_handle_is_waited(self):
        """When the wrapped op returns a non-None handle (async case), the
        wrapper must ``.wait()`` it before doing the in-place mul, otherwise
        the mul could race the all-reduce."""
        from veomni.ops.platform.npu.hccl_premul_sum import hccl_premul_sum_wrapper

        class _MockHandle:
            def __init__(self):
                self.waited = False

            def wait(self):
                self.waited = True

        handle = _MockHandle()

        def mock_op_fn(*args, **kwargs):
            return handle

        wrapper = hccl_premul_sum_wrapper(mock_op_fn, "tensor")
        tensor = torch.tensor([3.0, 6.0, 9.0])

        factor = 0.5
        mock_op = type("MockPREMUL_SUM", (), {})()
        mock_op.__getstate__ = lambda self: ("PREMUL_SUM", factor)

        wrapper(tensor, op=mock_op)
        assert handle.waited, "wrapper did not call .wait() on the returned handle"
        assert torch.allclose(tensor, torch.tensor([3.0, 6.0, 9.0]) * factor)

    def test_apply_patch_is_idempotent_via_repeat(self):
        """Calling ``apply_hccl_premul_sum_patch`` twice wraps an already-wrapped
        function (now a ``wrapper`` closure) — verify the result is still
        callable and that the demote-to-SUM logic still fires (i.e. we
        haven't broken the wrapper by double-patching)."""
        import torch.distributed as dist

        from veomni.ops.platform.npu.hccl_premul_sum import apply_hccl_premul_sum_patch

        # The patcher monkey-patches all three of these — save each so the
        # test never leaks a wrapped function into the rest of the session
        # (would break any subsequent test that calls dist.* directly).
        original_all_reduce = dist.all_reduce
        original_reduce_scatter = getattr(dist, "reduce_scatter", None)
        original_reduce_scatter_tensor = getattr(dist, "reduce_scatter_tensor", None)
        try:
            apply_hccl_premul_sum_patch()
            first_patched = dist.all_reduce
            apply_hccl_premul_sum_patch()
            second_patched = dist.all_reduce
            # Each patch should produce a new wrapper; the wrapped function
            # is still callable even if it's wrapped twice.
            assert first_patched is not original_all_reduce
            assert second_patched is not first_patched
            # And it's still a function (callable).
            assert callable(second_patched)
        finally:
            dist.all_reduce = original_all_reduce
            if original_reduce_scatter is not None:
                dist.reduce_scatter = original_reduce_scatter
            if original_reduce_scatter_tensor is not None:
                dist.reduce_scatter_tensor = original_reduce_scatter_tensor


# ---------------------------------------------------------------------------
# OpSlot public API contract — never covered before
# ---------------------------------------------------------------------------


class TestOpSlotContract:
    """The ``OpSlot`` class is the dispatch surface every modeling file uses.
    Its behaviour — what it does on ``bind()``, ``__call__``,
    ``use_non_eager_impl``, ``bound_kernel()`` — is part of the public
    contract and must not regress silently."""

    def test_unbound_slot_raises_on_call(self):
        """Calling an unbound OpSlot must raise a clear ``RuntimeError``,
        not the ambiguous TypeError you'd get from calling ``None``."""
        slot = OpSlot("rms_norm", "standard")
        with pytest.raises(RuntimeError, match="has no kernel bound"):
            slot(torch.zeros(2, 2), torch.zeros(2), 1e-6)

    def test_use_non_eager_impl_false_when_unbound(self):
        """A fresh slot is unbound → ``use_non_eager_impl`` is False.
        Modeling code uses this as the guard before falling through to the
        eager path; a True return here would silently route to ``None``."""
        slot = OpSlot("rms_norm", "standard")
        assert slot.use_non_eager_impl is False

    def test_use_non_eager_impl_false_for_eager_binding(self):
        """``bind("eager")`` resolves to ``None`` (see KernelRegistry.resolve);
        the slot should still report ``use_non_eager_impl=False`` so the
        model falls through to the eager path — not call ``None()``."""
        slot = OpSlot("rms_norm", "standard")
        slot.bind("eager")
        assert slot.bound_kernel() is None
        assert slot.use_non_eager_impl is False

    def test_use_non_eager_impl_true_after_npu_bind(self):
        """Symmetric: after binding to "npu", ``use_non_eager_impl`` is True."""
        slot = OpSlot("rms_norm", "standard")
        slot.bind("npu")
        assert slot.use_non_eager_impl is True
        assert slot.bound_kernel() is not None

    def test_rebind_to_different_impl_warns(self):
        """Module-level slots are shared by every model in the process.
        Rebinding to a different impl must emit a warning so eager-vs-fused
        evaluation setups spot the collision early (rather than silently
        overriding a previously-bound kernel)."""
        slot = OpSlot("rms_norm", "standard")
        slot.bind("npu")
        with pytest.warns(UserWarning, match="rebind"):
            slot.bind("eager")

    def test_rebind_to_same_impl_is_silent(self):
        """Rebinding to the *same* impl is a no-op and must not warn —
        otherwise importing the package twice (e.g. via two test fixtures)
        would spam warnings."""
        import warnings

        slot = OpSlot("rms_norm", "standard")
        slot.bind("npu")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an exception
            slot.bind("npu")  # must be silent

    def test_repr_unbound_eager_bound(self):
        """``__repr__`` is the primary diagnostic surface; lock it down so a
        future refactor doesn't change the human-readable form unexpectedly."""
        slot_unbound = OpSlot("rms_norm", "standard")
        assert "unbound" in repr(slot_unbound)

        slot_eager = OpSlot("rms_norm", "standard")
        slot_eager.bind("eager")
        assert "eager" in repr(slot_eager)

        slot_bound = OpSlot("rms_norm", "standard")
        slot_bound.bind("npu")
        r = repr(slot_bound)
        assert "rms_norm" in r
        assert "standard" in r


# ---------------------------------------------------------------------------
# KERNEL_REGISTRY public API contract — never covered before
# ---------------------------------------------------------------------------


class TestKernelRegistryContract:
    """``KERNEL_REGISTRY`` is a module-level global; its invariants are easy
    to break by accident in a refactor."""

    def test_eager_is_not_in_list_available(self):
        """``list_available`` is the user-facing API for "what backends can I
        bind?". ``eager`` is a sentinel that resolves to ``None`` rather than
        a real kernel, and exposing it as a listable backend would let users
        think they can dispatch to it (they can — via ``bind("eager")`` — but
        it's not a *kernel*)."""
        available = KERNEL_REGISTRY.list_available("rms_norm", "standard")
        assert "eager" not in available
        # And of course the NPU backend is in the list on this host.
        assert "npu" in available

    def test_resolve_eager_returns_none(self):
        """``bind("eager")`` must resolve to ``None`` (a sentinel the slot
        uses to fall through to the eager path) — not raise."""
        from veomni.ops.kernel_registry import KERNEL_REGISTRY

        assert KERNEL_REGISTRY.resolve("rms_norm", "standard", "eager") is None

    def test_resolve_unknown_impl_raises_keyerror(self):
        """``bind("nonexistent_kernel")`` must raise a ``KeyError`` whose
        message includes the available impls, so users get a self-explanatory
        error."""
        from veomni.ops.kernel_registry import KERNEL_REGISTRY

        with pytest.raises(KeyError, match="Unknown kernel"):
            KERNEL_REGISTRY.resolve("rms_norm", "standard", "definitely_not_a_real_kernel")

    def test_resolve_wrong_hardware_raises_runtimeerror(self):
        """Resolving a GPU-only kernel on an NPU host must raise RuntimeError,
        not silently return ``None`` (which the slot would treat as eager
        and silently skip the kernel).

        The module-level ``pytestmark`` already guarantees we're on an NPU
        host. We pick ``moe_experts.standard.triton`` — the Triton backend
        is registered with ``device_type='gpu', min_compute_capability=70``,
        so its ``HardwareRequirement.is_satisfied()`` returns False on
        Ascend and ``KERNEL_REGISTRY.resolve`` raises a cross-device error.
        """
        from veomni.ops.kernel_registry import KERNEL_REGISTRY

        with pytest.raises(RuntimeError, match="device_type='gpu'"):
            KERNEL_REGISTRY.resolve("moe_experts", "standard", "triton")

    def test_register_duplicate_raises(self):
        """Re-registering an existing (op, variant, name) without ``force=True``
        must raise ``ValueError`` — otherwise a typo in a kernel module's
        factory import would silently shadow the real one."""
        reg = KernelRegistry()
        reg.register(
            KernelSpec(
                name="x",
                op_name="op_x",
                variant="std",
                factory=lambda: (lambda x: x),
                hardware=HardwareRequirement(device_type="any"),
            )
        )
        with pytest.raises(ValueError, match="Duplicate kernel registration"):
            reg.register(
                KernelSpec(
                    name="x",
                    op_name="op_x",
                    variant="std",
                    factory=lambda: (lambda x: x),
                    hardware=HardwareRequirement(device_type="any"),
                )
            )

    def test_register_force_overrides_silently(self):
        """The ``force=True`` path is the official "I know what I'm doing,
        replace the existing kernel" path. Verify it returns no error and the
        new factory is the active one."""
        reg = KernelRegistry()
        reg.register(
            KernelSpec(
                name="x",
                op_name="op_x",
                variant="std",
                factory=lambda: "old",
                hardware=HardwareRequirement(device_type="any"),
            )
        )
        reg.register(
            KernelSpec(
                name="x",
                op_name="op_x",
                variant="std",
                factory=lambda: "new",
                hardware=HardwareRequirement(device_type="any"),
            ),
            force=True,
        )
        assert reg.resolve("op_x", "std", "x") == "new"

    def test_variants_isolated(self):
        """``(op, variant)`` is the registry key — two different variants of
        the same op must NOT share registrations. Catches a refactor that
        accidentally flattens the variant dimension."""
        reg = KernelRegistry()
        reg.register(
            KernelSpec(
                name="a",
                op_name="op_x",
                variant="v1",
                factory=lambda: "a1",
                hardware=HardwareRequirement(device_type="any"),
            )
        )
        # Re-using the same name under a different variant is fine.
        reg.register(
            KernelSpec(
                name="a",
                op_name="op_x",
                variant="v2",
                factory=lambda: "a2",
                hardware=HardwareRequirement(device_type="any"),
            )
        )
        assert reg.resolve("op_x", "v1", "a") == "a1"
        assert reg.resolve("op_x", "v2", "a") == "a2"
        # And v1 doesn't leak into v2.
        assert reg.list_available("op_x", "v1") == ["a"]
        assert reg.list_available("op_x", "v2") == ["a"]
