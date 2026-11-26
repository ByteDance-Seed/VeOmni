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

"""AscendC chunk_kda head-split guard tests (NPU-only).

Third verified fused-kernel failure mode: stateless varlen calls above
``_KDA_MAX_FUSED_HEADS`` qk heads corrupt the first heads' outputs on some
segment structures (full-head sp1 / non-SP training at the 96-head KDA
config). The guard splits those calls into head groups within the bound;
these tests pin the split's contract: the captured corrupting shape comes
back finite, sub-bound calls keep the unsplit path, the split is
bit-identical to a single fused call (per-head independence), the guarded
output matches the eager reference within bf16 tolerance, and the fused
backward stays finite.
"""

import pytest
import torch

from veomni.utils.import_utils import is_package_available


try:
    import torch_npu  # noqa: F401

    _NPU_HARDWARE_AVAILABLE = torch.npu.is_available()
except ImportError:
    _NPU_HARDWARE_AVAILABLE = False

# fla_npu and triton_ascend_kernels are out-of-tree NPU packages that are
# deliberately not part of the lockfile (same policy as the gated_delta_rule
# fla_npu backend), so stock NPU CI images do not ship them. The registry
# exposes the ascendc impl with requires=("fla_npu", "triton_ascend_kernels")
# and resolves to eager when they are missing; these contract tests follow
# the same rule and skip instead of failing collection.
_MISSING_PACKAGES = tuple(pkg for pkg in ("fla_npu", "triton_ascend_kernels") if not is_package_available(pkg))
_NPU_AVAILABLE = _NPU_HARDWARE_AVAILABLE and not _MISSING_PACKAGES

if _NPU_AVAILABLE:
    # Unexpected failures importing the implementation under test fail
    # collection here instead of silently skipping every test.
    from veomni.ops.kernels.kda.kimi_k3 import ascendc as kda_wrapper
    from veomni.ops.kernels.kda.kimi_k3.ascendc import chunk_kda_ascendc
    from veomni.ops.kernels.kda.kimi_k3.eager import chunk_kda_naive as kda_naive
else:
    kda_naive = None
    kda_wrapper = None
    chunk_kda_ascendc = None

_SKIP_REASON = "AscendC chunk_kda requires an Ascend NPU with fla_npu and triton_ascend_kernels installed"
if _NPU_HARDWARE_AVAILABLE and _MISSING_PACKAGES:
    _SKIP_REASON += f" (missing: {', '.join(_MISSING_PACKAGES)})"

pytestmark = [
    pytest.mark.skipif(not _NPU_AVAILABLE, reason=_SKIP_REASON),
]

DEVICE = "npu:0"
DTYPE = torch.bfloat16
K = 128
D = 128

# Captured sp1 training structure: segments (452, 340, 215, 1). The 1-token
# tail keeps the short-segment dispatch engaged exactly like production, and
# the (452, 340, 215) fused group is where the full-head corruption was
# characterized (H=96 corrupts, H<=92 clean on this structure).
CORRUPT_CU = (0, 452, 792, 1007, 1008)
# The same structure without the tail runs the pure fused path (no
# short-segment dispatch; 18 chunks, within the budget).
FUSED_CU = (0, 452, 792, 1007)
# An sp2-style captured structure with a 1-token tail.
CLEAN_CU = (0, 387, 913, 914)

FLAGS = dict(
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
    transpose_state_layout=True,
    disable_recompute=True,
)


@pytest.fixture(autouse=True)
def _reset_oneshot_flags():
    if kda_wrapper is not None:
        kda_wrapper._KDA_HEAD_SPLIT_WARNED = False
        kda_wrapper._KDA_SHORT_SEG_WARNED = False
        kda_wrapper._KDA_CHUNK_BUDGET_WARNED = False
        kda_wrapper._KDA_CORRUPT_WARNED = False
    yield


def _make_inputs(T, H, cu, seed=1234):
    g_ = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(1, T, H, K, generator=g_) * 0.5).to(DTYPE).to(DEVICE)
    k = (torch.randn(1, T, H, K, generator=g_) * 0.5).to(DTYPE).to(DEVICE)
    v = (torch.randn(1, T, H, D, generator=g_) * 0.5).to(DTYPE).to(DEVICE)
    g = (-torch.rand(1, T, H, K, generator=g_) * 4.0).to(DTYPE).to(DEVICE)
    beta = torch.rand(1, T, H, generator=g_).to(DEVICE)
    A_log = torch.log(torch.empty(H, dtype=torch.float32).uniform_(1, 16, generator=g_)).to(DEVICE)
    dt_bias = (torch.randn(H * K, generator=g_) * 0.1).to(DEVICE)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32)
    return q, k, v, g, beta, A_log, dt_bias, cu_seqlens


def _call(t):
    o, _ = chunk_kda_ascendc(
        q=t[0],
        k=t[1],
        v=t[2],
        g=t[3],
        beta=t[4],
        A_log=t[5],
        dt_bias=t[6],
        initial_state=None,
        output_final_state=False,
        cu_seqlens=t[7],
        cu_seqlens_cpu=t[7],
        **FLAGS,
    )
    return o


def test_head_split_heals_full_head_corruption():
    """Above the head bound the captured sp1 shape must come back finite.

    Without the routing this exact call returns NaN (characterized with
    VEOMNI_KDA_ASCENDC_GUARDED=0: 24458 non-finite elements), so the test
    fails open if the guard regresses.
    """
    t = _make_inputs(1016, 96, CORRUPT_CU)
    o = _call(t)
    assert torch.isfinite(o.float()).all()
    assert kda_wrapper._KDA_HEAD_SPLIT_WARNED


def test_sub_bound_head_count_keeps_unsplit_path():
    """Calls within the bound must not engage the head-split dispatch."""
    t = _make_inputs(1016, 48, CLEAN_CU)
    o = _call(t)
    assert torch.isfinite(o.float()).all()
    assert not kda_wrapper._KDA_HEAD_SPLIT_WARNED


def test_head_split_is_bit_identical_below_bound():
    """Head independence: a single fused call equals concatenated halves.

    This is the mathematical foundation of the split -- the chunked KDA
    recurrence never mixes qk heads, so splitting is a pure re-grouping of
    kernel launches. Verified on the pure fused path (no short-segment
    dispatch involved).
    """
    t = _make_inputs(1007, 48, FUSED_CU)
    o_single = _call(t)
    groups = []
    for a in range(0, 48, 24):
        b = a + 24
        sub = (
            t[0][:, :, a:b],
            t[1][:, :, a:b],
            t[2][:, :, a:b],
            t[3][:, :, a:b],
            t[4][:, :, a:b],
            t[5][a:b],
            t[6][a * K : b * K],
            t[7],
        )
        groups.append(_call(sub))
    o_split = torch.cat(groups, dim=2)
    assert torch.equal(o_single, o_split)


def test_guarded_full_head_matches_eager_reference():
    """The healed output stays numerically faithful (bf16 tolerance)."""
    t = _make_inputs(1016, 96, CORRUPT_CU)
    o = _call(t)
    o_ref, _ = kda_naive(
        q=t[0],
        k=t[1],
        v=t[2],
        g=t[3],
        beta=t[4],
        A_log=t[5],
        dt_bias=t[6],
        scale=K**-0.5,
        cu_seqlens=t[7],
        **FLAGS,
    )
    total = CORRUPT_CU[-1]
    torch.testing.assert_close(o[:, :total].float(), o_ref.float(), rtol=2e-2, atol=2e-3)


def test_head_split_backward_finite():
    """The fused backward runs per head group and stays finite.

    The chunk-budget characterization showed the fused backward corrupts
    exactly when the forward does, so the split must cover backward too.
    """
    t = list(_make_inputs(1016, 96, CORRUPT_CU))
    for i in range(4):
        t[i] = t[i].detach().clone().requires_grad_()
    o = _call(tuple(t))
    o.float().sum().backward()
    for i in range(4):
        assert torch.isfinite(t[i].grad.float()).all()
