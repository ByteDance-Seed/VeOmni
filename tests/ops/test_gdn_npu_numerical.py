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
"""Numerical parity of the Qwen3.5 GatedDeltaNet NPU kernels against eager.

Covers the NPU backends this PR adds:

- ``causal_conv1d`` ``npu`` (vendored MindSpeed-MM Triton) vs a pure-torch
  causal depthwise conv1d.
- ``chunk_gated_delta_rule`` ``npu`` (vendored Triton) and ``npu_ascendc``
  (AscendC fused ``torch.ops.npu.*`` via ``fla_npu``) vs the transformers eager
  reference ``torch_chunk_gated_delta_rule`` (the impl the model falls back to
  when the op is left ``eager``).

Each delta-rule backend is checked forward **and** backward, for a 64-aligned
sequence and for a ragged **varlen** (``cu_seqlens``, non-64-multiple) packing.
Because eager ``torch_chunk_gated_delta_rule`` has no ``cu_seqlens`` path, the
varlen reference is built per segment (batch=1) and concatenated. The varlen
backward case is the regression guard for the ``l2norm_bwd`` floor->ceil fix:
with ``use_qk_l2norm_in_kernel=True`` the kernel runs ``l2norm_bwd``, whose floor
task count used to drop the partial last block on ragged ``T`` and leave
uninitialized ``dx`` (blowing up ``dq``/``dk``).

Skipped on non-NPU hosts; the ``npu_ascendc`` cases additionally skip when
``fla_npu`` is absent. Relative-error (``get_err_ratio``) thresholds may need a
small bump on the first self-hosted run and are commented accordingly.
"""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

import veomni.ops  # noqa: F401 — trigger KERNEL_REGISTRY registrations
from veomni.ops.dispatch import OpSlot
from veomni.ops.kernels.gated_delta_rule._ascend.triton_core.utils import get_err_ratio
from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU kernels require torch_npu")

DEVICE = get_device_type()

_HAS_FLA_NPU = importlib.util.find_spec("fla_npu") is not None
_needs_fla_npu = pytest.mark.skipif(not _HAS_FLA_NPU, reason="npu_ascendc backend requires fla_npu")

# Relative-RMS-error thresholds (get_err_ratio). bf16 fused GDN drifts a few
# percent from the fp32 eager math; backward accumulates a bit more. Bump here
# if the first self-hosted run lands just over.
_FWD_RATIO = 2e-2
_BWD_RATIO = 4e-2


# ---------------------------------------------------------------------------
# Eager references
# ---------------------------------------------------------------------------
def _eager_causal_conv1d(x, weight, bias, activation):
    """Pure-torch causal depthwise conv1d. x: [B, T, D], weight: [D, W]."""
    b, t, d = x.shape
    w = weight.shape[-1]
    xt = F.pad(x.transpose(1, 2), (w - 1, 0))  # [B, D, T+W-1] — causal left pad
    out = F.conv1d(xt, weight.unsqueeze(1), bias, groups=d).transpose(1, 2)  # [B, T, D]
    if activation == "silu":
        out = F.silu(out)
    return out.to(x.dtype)


def _eager_chunk_gdr(q, k, v, g, beta, use_qk_l2norm_in_kernel):
    """transformers reference. q/k/v: [B, T, H, D], g/beta: [B, T, H] -> o [B, T, H, Dv]."""
    from veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_npu import (
        torch_chunk_gated_delta_rule,
    )

    o, _ = torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
    return o


def _eager_chunk_gdr_varlen(q, k, v, g, beta, cu_seqlens, use_qk_l2norm_in_kernel):
    """Varlen reference: eager has no cu_seqlens path, so run each packed segment
    (batch=1) independently and concatenate along seq. Inputs packed [1, T, H, D]."""
    outs = []
    cu = cu_seqlens.tolist()
    for start, end in zip(cu[:-1], cu[1:]):
        outs.append(
            _eager_chunk_gdr(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                g[:, start:end],
                beta[:, start:end],
                use_qk_l2norm_in_kernel,
            )
        )
    return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rand_gdr_inputs(batch, seq, num_heads, head_dim, dtype):
    """q/k/v [B, T, H, D]; g log-decay (<=0) [B, T, H]; beta in (0,1) [B, T, H]."""
    gen = lambda *s: torch.randn(*s, device=DEVICE, dtype=dtype)  # noqa: E731
    q = gen(batch, seq, num_heads, head_dim)
    k = gen(batch, seq, num_heads, head_dim)
    v = gen(batch, seq, num_heads, head_dim)
    g = F.logsigmoid(gen(batch, seq, num_heads))  # log-decay, <= 0
    beta = gen(batch, seq, num_heads).sigmoid()  # gate in (0, 1)
    return [q, k, v, g, beta]


def _leaf(t):
    return t.detach().clone().requires_grad_(True)


def _chunk_kernel(backend):
    slot = OpSlot("chunk_gated_delta_rule", "standard")
    slot.bind(backend)
    return slot.bound_kernel()


def _call_chunk(fn, inputs, cu_seqlens):
    q, k, v, g, beta = inputs
    if cu_seqlens is not None:
        out = fn(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens)
    else:
        out = fn(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    return out[0] if isinstance(out, tuple) else out


def _assert_chunk_parity(backend, batch, seq, num_heads, head_dim, cu_seqlens):
    base = _rand_gdr_inputs(batch, seq, num_heads, head_dim, torch.bfloat16)
    k_in = [_leaf(t) for t in base]  # for the kernel
    r_in = [_leaf(t) for t in base]  # for the eager reference

    o_k = _call_chunk(_chunk_kernel(backend), k_in, cu_seqlens)
    grad_o = torch.randn_like(o_k)  # shared upstream grad
    o_k.backward(grad_o)

    q, k, v, g, beta = r_in
    o_r = (
        _eager_chunk_gdr_varlen(q, k, v, g, beta, cu_seqlens, True)
        if cu_seqlens is not None
        else _eager_chunk_gdr(q, k, v, g, beta, True)
    )
    o_r.backward(grad_o)

    assert get_err_ratio(o_r.float(), o_k.float()) < _FWD_RATIO, "forward mismatch"
    for name, tk, tr in zip(("dq", "dk", "dv", "dg", "dbeta"), k_in, r_in):
        assert get_err_ratio(tr.grad.float(), tk.grad.float()) < _BWD_RATIO, f"{name} mismatch"


# ---------------------------------------------------------------------------
# causal_conv1d (npu) vs eager
# ---------------------------------------------------------------------------
class TestNPUCausalConv1d:
    @pytest.mark.parametrize("batch,seq,dim,width", [(2, 64, 128, 4), (1, 48, 64, 3)])
    def test_npu_matches_eager_fwd_bwd(self, batch, seq, dim, width):
        slot = OpSlot("causal_conv1d", "standard")
        slot.bind("npu")
        conv = slot.bound_kernel()

        x = torch.randn(batch, seq, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(dim, width, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        bias = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        xr, wr, br = _leaf(x), _leaf(weight), _leaf(bias)

        out = conv(x, weight, bias=bias, activation="silu")
        y = out[0] if isinstance(out, tuple) else out
        grad_o = torch.randn_like(y)
        y.backward(grad_o)

        y_ref = _eager_causal_conv1d(xr, wr, br, "silu")
        y_ref.backward(grad_o)

        # Depthwise conv + SiLU: a few bf16 ULPs. 2e-2 covers it.
        assert torch.allclose(y.float(), y_ref.float(), atol=2e-2, rtol=2e-2)
        assert torch.allclose(x.grad.float(), xr.grad.float(), atol=2e-2, rtol=2e-2)
        assert torch.allclose(weight.grad.float(), wr.grad.float(), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# chunk_gated_delta_rule (npu, triton-ascend) vs eager
# ---------------------------------------------------------------------------
class TestNPUChunkGatedDeltaRule:
    def test_npu_fwd_bwd_aligned(self):
        # seq = 64 multiple: the well-behaved path.
        _assert_chunk_parity("npu", batch=1, seq=128, num_heads=4, head_dim=64, cu_seqlens=None)

    def test_npu_fwd_bwd_varlen(self):
        # Ragged packing, total T (88) not a multiple of 64 -> exercises
        # l2norm_bwd's partial last block (the floor->ceil fix).
        cu = torch.tensor([0, 40, 88], device=DEVICE, dtype=torch.int32)
        _assert_chunk_parity("npu", batch=1, seq=88, num_heads=4, head_dim=64, cu_seqlens=cu)


# ---------------------------------------------------------------------------
# chunk_gated_delta_rule (npu_ascendc, fla_npu) vs eager
# ---------------------------------------------------------------------------
@_needs_fla_npu
class TestNPUAscendcChunkGatedDeltaRule:
    def test_npu_ascendc_fwd_bwd_aligned(self):
        _assert_chunk_parity("npu_ascendc", batch=1, seq=128, num_heads=4, head_dim=64, cu_seqlens=None)

    def test_npu_ascendc_fwd_bwd_varlen(self):
        cu = torch.tensor([0, 40, 88], device=DEVICE, dtype=torch.int32)
        _assert_chunk_parity("npu_ascendc", batch=1, seq=88, num_heads=4, head_dim=64, cu_seqlens=cu)
