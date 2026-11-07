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

"""AscendC chunk_kda state-layout contract tests (NPU-only).

K3's KDA callers request the transposed state layout on both SP and non-SP
paths (``transpose_state_layout=True`` -> ``state_v_first=True``). The
AscendC entry must pass the request through to the aclnn kernel, which
natively supports both state layouts: stateless training must run
unrejected, and stateful V-first requests must numerically match the eager
fallback reference.
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
    from veomni.models.transformers.kimi_k3.source.kda_fallback import chunk_kda as kda_reference
    from veomni.ops.kernels.kda.ascendc.chunk_kda_ascendc_kimi import chunk_kda_ascendc
else:
    kda_reference = None
    chunk_kda_ascendc = None

_SKIP_REASON = "AscendC chunk_kda requires an Ascend NPU with fla_npu and triton_ascend_kernels installed"
if _NPU_HARDWARE_AVAILABLE and _MISSING_PACKAGES:
    _SKIP_REASON += f" (missing: {', '.join(_MISSING_PACKAGES)})"


pytestmark = [
    pytest.mark.skipif(
        not _NPU_AVAILABLE,
        reason=_SKIP_REASON,
    ),
    # The KDA callers legitimately use the deprecated transpose_state_layout kwarg.
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]


DEVICE = "npu:0"
DTYPE = torch.bfloat16


def _make_inputs(B, T, H, K, V, seed):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=DTYPE)
    g = -torch.rand(B, T, H, K, device=DEVICE, dtype=torch.float32)
    beta = torch.rand(B, T, H, device=DEVICE, dtype=torch.float32)
    return q, k, v, g, beta


def test_stateless_transposed_layout_runs_and_matches_reference():
    B, T, H, K, V = 2, 128, 2, 64, 64
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, seed=1234)

    o_npu, final_npu = chunk_kda_ascendc(
        q, k, v, g, beta, use_qk_l2norm_in_kernel=True, transpose_state_layout=True, chunk_size=64
    )
    assert final_npu is None

    o_ref, _ = kda_reference(
        q, k, v, g, beta, use_qk_l2norm_in_kernel=True, transpose_state_layout=True, chunk_size=64
    )
    torch.testing.assert_close(o_npu.float(), o_ref.float(), rtol=3e-2, atol=3e-2)


def test_stateful_transposed_layout_matches_reference():
    B, T, H, K, V = 2, 128, 2, 64, 64
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, seed=7)
    torch.manual_seed(8)
    h0_vfirst = torch.randn(B, H, V, K, device=DEVICE, dtype=torch.float32)

    o_npu, final_npu = chunk_kda_ascendc(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0_vfirst,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        transpose_state_layout=True,
        chunk_size=64,
    )
    assert tuple(final_npu.shape) == (B, H, V, K)

    o_ref, final_ref = kda_reference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0_vfirst,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        transpose_state_layout=True,
        chunk_size=64,
    )
    torch.testing.assert_close(o_npu.float(), o_ref.float(), rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(final_npu.float(), final_ref.float(), rtol=3e-2, atol=3e-2)


def test_varlen_stateless_transposed_layout_matches_reference():
    H, K, V, T = 2, 64, 64, 192
    q, k, v, g, beta = _make_inputs(1, T, H, K, V, seed=11)
    cu = torch.tensor([0, 100, 192], dtype=torch.long, device=DEVICE)

    o_npu, _ = chunk_kda_ascendc(
        q,
        k,
        v,
        g,
        beta,
        use_qk_l2norm_in_kernel=True,
        transpose_state_layout=True,
        cu_seqlens=cu,
        chunk_size=64,
    )
    o_ref, _ = kda_reference(
        q,
        k,
        v,
        g,
        beta,
        use_qk_l2norm_in_kernel=True,
        transpose_state_layout=True,
        cu_seqlens=cu,
        chunk_size=64,
    )
    torch.testing.assert_close(o_npu.float(), o_ref.float(), rtol=3e-2, atol=3e-2)


def test_stateless_backward_matches_reference():
    """The state-layout flag must not perturb stateless training.

    Stateless calls normalize state_v_first away inside
    ChunkKDAFunction, so the transposed-layout request runs the exact
    same AscendC forward + fused backward route as
    state_v_first=False (bit-identical outputs and gradients), and
    still matches the eager reference.
    """
    B, T, H, K, V = 2, 128, 2, 128, 128

    q, k, v, g, beta = _make_inputs(B, T, H, K, V, seed=21)

    def _run(fn, **kw):
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
        o, _ = fn(*leaves, use_qk_l2norm_in_kernel=True, chunk_size=64, **kw)
        o.float().sum().backward()
        return o.detach(), [t.grad.detach() for t in leaves]

    o_flag, grads_flag = _run(chunk_kda_ascendc, transpose_state_layout=True)
    o_base, grads_base = _run(chunk_kda_ascendc)
    assert torch.equal(o_flag, o_base)
    for a, b in zip(grads_flag, grads_base):
        assert torch.equal(a, b)

    o_ref, grads_ref = _run(kda_reference, transpose_state_layout=True)
    torch.testing.assert_close(o_flag.float(), o_ref.float(), rtol=5e-2, atol=5e-2)
    for a, b in zip(grads_flag, grads_ref):
        torch.testing.assert_close(a.float(), b.float(), rtol=5e-2, atol=5e-2)


def test_intermediate_states_honor_transposed_layout():
    """Intermediate states are a caller-visible state crossing the boundary.

    With return_intermediate_states=True and no initial/final state the
    AscendC forward must still build h in the requested transposed layout
    ([B, NT, H, V, K] under transpose_state_layout=True, matching the eager
    reference) instead of silently keeping the canonical (K, V) order used
    by the stateless fused-backward route.
    """
    B, T, H, K, V = 2, 128, 2, 64, 64
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, seed=31)

    # The wrapper only exposes intermediate states in inference mode.
    with torch.inference_mode():
        o_npu, _, h_npu = chunk_kda_ascendc(
            q,
            k,
            v,
            g,
            beta,
            use_qk_l2norm_in_kernel=True,
            return_intermediate_states=True,
            transpose_state_layout=True,
            chunk_size=64,
        )
        o_ref, _, h_ref = kda_reference(
            q,
            k,
            v,
            g,
            beta,
            use_qk_l2norm_in_kernel=True,
            return_intermediate_states=True,
            transpose_state_layout=True,
            chunk_size=64,
        )
    assert h_npu.shape == h_ref.shape
    torch.testing.assert_close(h_npu.float(), h_ref.float(), rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(o_npu.float(), o_ref.float(), rtol=3e-2, atol=3e-2)
