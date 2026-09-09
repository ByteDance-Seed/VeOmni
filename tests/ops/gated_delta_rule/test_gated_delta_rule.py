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

"""Gated delta-rule eager vs HF, and fused impls vs eager."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated, torch_chunk_gated_delta_rule

from tests.ops.tol import (
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
    GDN_CHUNK_ATOL,
    GDN_CHUNK_GRAD_ATOL,
    GDN_CHUNK_GRAD_RTOL,
    GDN_CHUNK_RTOL,
    GDN_FUSED_ATOL,
    GDN_FUSED_GRAD_ATOL,
    GDN_FUSED_GRAD_RTOL,
    GDN_FUSED_RTOL,
    GDN_NPU_ATOL,
    GDN_NPU_RTOL,
)
from veomni.ops import OP_REGISTRY, resolve_op
from veomni.ops.registry import OpEntry
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_gpu_compute_capability


def _clone(*tensors: Tensor) -> tuple[Tensor, ...]:
    return tuple(t.detach().requires_grad_(True) for t in tensors)


@pytest.mark.parametrize(
    "kernel",
    ("rms_norm_gated", "causal_conv1d", "chunk_gated_delta_rule"),
)
def test_fla_is_registered_for_cuda_and_mlu(kernel):
    assert (kernel, "standard", "fla", "cuda") in OP_REGISTRY._entries
    assert (kernel, "standard", "fla", "mlu") in OP_REGISTRY._entries
    assert OP_REGISTRY.list_registered(kernel, "standard").count("fla") == 1


@pytest.mark.parametrize(
    "kernel,impls",
    (
        ("rms_norm_gated", ["eager", "fla", "npu"]),
        ("causal_conv1d", ["eager", "fla", "npu"]),
        ("chunk_gated_delta_rule", ["eager", "fla", "flash_qla", "npu", "npu_ascendc"]),
    ),
)
def test_gated_delta_rule_registration_matrix(kernel, impls):
    assert OP_REGISTRY.list_registered(kernel, "standard") == impls


def test_npu_ascendc_is_scoped_to_chunk_gated_delta_rule():
    assert "npu_ascendc" in OP_REGISTRY.list_registered("chunk_gated_delta_rule", "standard")
    assert "npu_ascendc" not in OP_REGISTRY.list_registered("rms_norm_gated", "standard")
    assert "npu_ascendc" not in OP_REGISTRY.list_registered("causal_conv1d", "standard")


def test_rms_norm_gated_eager_matches_hf():
    torch.manual_seed(0)
    hidden = 64
    eps = 1e-6
    x = torch.randn(2, 16, hidden, dtype=torch.float32)
    gate = torch.randn(2, 16, hidden, dtype=torch.float32)
    weight = torch.randn(hidden, dtype=torch.float32)

    module = Qwen3_5RMSNormGated(hidden, eps=eps)
    with torch.no_grad():
        module.weight.copy_(weight)

    x_h, g_h = _clone(x, gate)
    out_h = module(x_h, g_h)

    x_e, g_e, w_e = _clone(x, gate, weight)
    out_e = resolve_op("rms_norm_gated", "standard", "eager").wrapper(x_e, g_e, w_e, eps=eps)
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(x_e.grad, x_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(g_e.grad, g_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(w_e.grad, module.weight.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FLA rms_norm_gated needs a GPU")
def test_rms_norm_gated_fla_matches_eager():
    pytest.importorskip("fla")
    eager = resolve_op("rms_norm_gated", "standard", "eager").wrapper
    other = resolve_op("rms_norm_gated", "standard", "fla").wrapper
    torch.manual_seed(0)
    hidden = 64
    x = torch.randn(2, 16, hidden, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(2, 16, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    x_e, g_e, w_e = _clone(x, gate, weight)
    x_o, g_o, w_o = _clone(x, gate, weight)
    out_e = eager(x_e, g_e, w_e, eps=1e-6)
    out_o = other(x_o, g_o, w_o, eps=1e-6)
    assert torch.allclose(out_e, out_o, atol=GDN_FUSED_ATOL, rtol=GDN_FUSED_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(x_e.grad, x_o.grad, atol=GDN_FUSED_GRAD_ATOL, rtol=GDN_FUSED_GRAD_RTOL)
    assert torch.allclose(g_e.grad, g_o.grad, atol=GDN_FUSED_GRAD_ATOL, rtol=GDN_FUSED_GRAD_RTOL)
    assert torch.allclose(w_e.grad, w_o.grad, atol=GDN_FUSED_GRAD_ATOL, rtol=GDN_FUSED_GRAD_RTOL)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="rms_norm_gated npu needs torch_npu")
@pytest.mark.parametrize("shape", [(2, 16, 128), (1, 8, 64)])
def test_rms_norm_gated_npu_matches_eager(shape):
    eager = resolve_op("rms_norm_gated", "standard", "eager").wrapper
    other = resolve_op("rms_norm_gated", "standard", "npu").wrapper
    torch.manual_seed(0)
    x = torch.randn(*shape, device="npu", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    weight = torch.randn(shape[-1], device="npu", dtype=torch.bfloat16)

    out_e = eager(x, gate, weight, eps=1e-6)
    out_o = other(x, gate, weight, eps=1e-6)
    assert out_o.shape == x.shape
    assert out_o.dtype == x.dtype
    assert torch.allclose(out_o.float(), out_e.float(), atol=GDN_NPU_ATOL, rtol=GDN_NPU_RTOL)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="rms_norm_gated npu needs torch_npu")
def test_rms_norm_gated_npu_zero_gate_is_zero():
    other = resolve_op("rms_norm_gated", "standard", "npu").wrapper
    x = torch.randn(1, 4, 32, device="npu", dtype=torch.bfloat16)
    gate = torch.zeros_like(x)
    weight = torch.randn(32, device="npu", dtype=torch.bfloat16)

    output = other(x, gate, weight, eps=1e-6)
    assert torch.count_nonzero(output) == 0


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="rms_norm_gated npu needs torch_npu")
@pytest.mark.parametrize("eps", [1e-5, 1e-6, 1e-7])
def test_rms_norm_gated_npu_uses_eps(eps):
    eager = resolve_op("rms_norm_gated", "standard", "eager").wrapper
    other = resolve_op("rms_norm_gated", "standard", "npu").wrapper
    torch.manual_seed(1)
    x = torch.randn(1, 4, 32, device="npu", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    weight = torch.randn(32, device="npu", dtype=torch.bfloat16)

    out_e = eager(x, gate, weight, eps=eps)
    out_o = other(x, gate, weight, eps=eps)
    assert torch.allclose(out_o.float(), out_e.float(), atol=GDN_NPU_ATOL, rtol=GDN_NPU_RTOL)


def _hf_qwen3_5_prefill_causal_conv1d(x: Tensor, weight: Tensor, bias: Tensor, *, kernel_size: int) -> Tensor:
    """Qwen3.5 GatedDeltaNet prefill conv when ``causal_conv1d_fn`` is None.

    Adapted from ``Qwen3_5GatedDeltaNet.forward``:
    ``F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])``.
    ``self.conv1d`` is ``nn.Conv1d(..., padding=kernel_size-1, groups=dim)``.

    Source:
    https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
    """
    mixed = x.transpose(1, 2)
    conv = F.conv1d(
        mixed,
        weight.unsqueeze(1),
        bias,
        padding=kernel_size - 1,
        groups=mixed.shape[1],
    )
    return F.silu(conv[:, :, : mixed.shape[-1]]).transpose(1, 2).contiguous()


def test_causal_conv1d_eager_matches_hf():
    torch.manual_seed(1)
    batch, seq, dim, kernel = 2, 16, 32, 4
    x = torch.randn(batch, seq, dim, dtype=torch.float32)
    weight = torch.randn(dim, kernel, dtype=torch.float32)
    bias = torch.randn(dim, dtype=torch.float32)

    x_e, w_e, b_e = _clone(x, weight, bias)
    out_e = resolve_op("causal_conv1d", "standard", "eager").wrapper(x_e, w_e, b_e, activation="silu")

    x_r, w_r, b_r = _clone(x, weight, bias)
    out_r = _hf_qwen3_5_prefill_causal_conv1d(x_r, w_r, b_r, kernel_size=kernel)
    assert torch.allclose(out_e, out_r, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_r.backward(go)
    assert torch.allclose(x_e.grad, x_r.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(w_e.grad, w_r.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(b_e.grad, b_r.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FLA causal_conv1d needs a GPU")
def test_causal_conv1d_fla_matches_eager():
    pytest.importorskip("fla")
    eager = resolve_op("causal_conv1d", "standard", "eager").wrapper
    other = resolve_op("causal_conv1d", "standard", "fla").wrapper
    torch.manual_seed(1)
    batch, seq, dim, kernel = 2, 16, 32, 4
    x = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, kernel, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16)

    x_e, w_e, b_e = _clone(x, weight, bias)
    x_o, w_o, b_o = _clone(x, weight, bias)
    out_e = eager(x_e, w_e, b_e, activation="silu")
    out_o = other(x_o, w_o, b_o, activation="silu")
    assert torch.allclose(out_e, out_o, atol=GDN_FUSED_ATOL, rtol=GDN_FUSED_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(x_e.grad, x_o.grad, atol=GDN_FUSED_GRAD_ATOL, rtol=GDN_FUSED_GRAD_RTOL)
    assert torch.allclose(w_e.grad, w_o.grad, atol=GDN_FUSED_GRAD_ATOL, rtol=GDN_FUSED_GRAD_RTOL)
    assert torch.allclose(b_e.grad, b_o.grad, atol=GDN_FUSED_GRAD_ATOL, rtol=GDN_FUSED_GRAD_RTOL)


def test_chunk_gated_delta_rule_eager_matches_hf():
    torch.manual_seed(2)
    batch, seq, heads, dim = 1, 32, 2, 16
    q = torch.randn(batch, seq, heads, dim, dtype=torch.float32)
    k = torch.randn(batch, seq, heads, dim, dtype=torch.float32)
    v = torch.randn(batch, seq, heads, dim, dtype=torch.float32)
    g = -torch.rand(batch, seq, heads, dtype=torch.float32) * 0.5
    beta = torch.rand(batch, seq, heads, dtype=torch.float32)

    q_h, k_h, v_h, g_h, b_h = _clone(q, k, v, g, beta)
    out_h, _ = torch_chunk_gated_delta_rule(
        q_h,
        k_h,
        v_h,
        g_h,
        b_h,
        chunk_size=16,
        use_qk_l2norm_in_kernel=True,
    )

    q_e, k_e, v_e, g_e, b_e = _clone(q, k, v, g, beta)
    out_e, _ = resolve_op("chunk_gated_delta_rule", "standard", "eager").wrapper(
        q_e,
        k_e,
        v_e,
        g_e,
        b_e,
        use_qk_l2norm_in_kernel=True,
        chunk_size=16,
    )
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(q_e.grad, q_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(k_e.grad, k_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(v_e.grad, v_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(g_e.grad, g_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(b_e.grad, b_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_chunk_gated_delta_rule_eager_uses_explicit_scale():
    torch.manual_seed(3)
    batch, seq, heads, dim = 1, 16, 2, 8
    tensors = (
        torch.randn(batch, seq, heads, dim),
        torch.randn(batch, seq, heads, dim),
        torch.randn(batch, seq, heads, dim),
        -torch.rand(batch, seq, heads) * 0.5,
        torch.rand(batch, seq, heads),
    )
    explicit_scale = 0.25
    default_scale = dim**-0.5
    eager = resolve_op("chunk_gated_delta_rule", "standard", "eager").wrapper

    q_e, k_e, v_e, g_e, b_e = _clone(*tensors)
    out_e, _ = eager(q_e, k_e, v_e, g_e, b_e, chunk_size=8, scale=explicit_scale)

    q_r, k_r, v_r, g_r, b_r = _clone(*tensors)
    out_r, _ = eager(q_r * (explicit_scale / default_scale), k_r, v_r, g_r, b_r, chunk_size=8)
    assert torch.allclose(out_e, out_r, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    grad_output = torch.randn_like(out_e)
    out_e.backward(grad_output)
    out_r.backward(grad_output)
    for actual, expected in zip((q_e, k_e, v_e, g_e, b_e), (q_r, k_r, v_r, g_r, b_r), strict=True):
        assert torch.allclose(actual.grad, expected.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


@pytest.mark.parametrize(
    "impl,vendor_module",
    (
        ("fla", "fla.ops.gated_delta_rule"),
        ("flash_qla", "flash_qla.ops.gated_delta_rule"),
    ),
)
def test_chunk_gated_delta_rule_adapter_forwards_scale(
    impl: str,
    vendor_module: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fake_vendor = ModuleType(vendor_module)

    def fake_chunk_gated_delta_rule(*args, **kwargs):
        captured.update(kwargs)
        return args[0], None

    fake_vendor.chunk_gated_delta_rule = fake_chunk_gated_delta_rule
    monkeypatch.setitem(sys.modules, vendor_module, fake_vendor)

    shape = (1, 4, 2, 8)
    query = torch.randn(shape)
    key = torch.randn(shape)
    value = torch.randn(shape)
    g = torch.randn(shape[:3])
    beta = torch.randn(shape[:3])
    explicit_scale = 0.375
    module = import_module(f"veomni.ops.kernels.gated_delta_rule.chunk_gated_delta_rule.standard.{impl}")
    module.wrapper(query, key, value, g, beta, scale=explicit_scale)

    assert captured["scale"] == explicit_scale


@pytest.mark.parametrize("impl", ("npu", "npu_ascendc"))
def test_chunk_gated_delta_rule_npu_l2norm_preserves_grad_chain(
    impl: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the NPU raw-pair autograd glue without requiring NPU hardware."""
    module = import_module(f"veomni.ops.kernels.gated_delta_rule.chunk_gated_delta_rule.standard.{impl}")
    triton_utils = import_module("veomni.ops.kernels.gated_delta_rule.vendor.triton.utils")
    monkeypatch.setattr(triton_utils, "input_guard", lambda fn: fn)

    head_first = impl == "npu_ascendc"
    explicit_scale = 0.375
    seen_scales: list[tuple[str, float]] = []

    def fake_chunk_fwd(query, key, value, g, beta, scale, *args):
        del beta, args
        seen_scales.append(("forward", scale))
        output = query + 2 * key
        if head_first:
            output = output.transpose(1, 2).contiguous()
        return g, output, query.new_empty(0), None

    def fake_chunk_bwd(query, key, value, g, beta, a, scale, initial_state, grad_output, *args):
        del a, initial_state, args
        seen_scales.append(("backward", scale))
        if head_first:
            grad_output = grad_output.transpose(1, 2).contiguous()
        grads = (
            grad_output,
            2 * grad_output,
            torch.zeros_like(value),
            torch.zeros_like(beta),
            torch.zeros_like(g),
        )
        return grads if head_first else (*grads, None)

    monkeypatch.setattr(module, "_chunk_fwd", fake_chunk_fwd)
    monkeypatch.setattr(module, "_chunk_bwd", fake_chunk_bwd)

    torch.manual_seed(6)
    shape = (1, 4, 2, 8)
    query = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(shape[:3], dtype=torch.float32, requires_grad=True)
    beta = torch.randn(shape[:3], dtype=torch.bfloat16, requires_grad=True)
    entry = OpEntry("test_chunk_gdr", "standard", impl, module.forward, module.backward)
    output, _final_state = entry.wrapper(
        query,
        key,
        value,
        g,
        beta,
        use_qk_l2norm_in_kernel=True,
        scale=explicit_scale,
    )
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    query_ref = query.detach().requires_grad_(True)
    key_ref = key.detach().requires_grad_(True)
    query_norm = query_ref * torch.rsqrt((query_ref * query_ref).sum(dim=-1, keepdim=True) + 1e-6)
    key_norm = key_ref * torch.rsqrt((key_ref * key_ref).sum(dim=-1, keepdim=True) + 1e-6)
    (query_norm + 2 * key_norm).backward(grad_output)

    assert torch.allclose(query.grad, query_ref.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(key.grad, key_ref.grad, atol=2e-2, rtol=2e-2)
    assert seen_scales == [("forward", explicit_scale), ("backward", explicit_scale)]


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="FLA chunk_gated_delta_rule needs a GPU")
def test_chunk_gated_delta_rule_fla_matches_eager():
    pytest.importorskip("fla")
    eager = resolve_op("chunk_gated_delta_rule", "standard", "eager").wrapper
    other = resolve_op("chunk_gated_delta_rule", "standard", "fla").wrapper
    torch.manual_seed(2)
    batch, seq, heads, dim = 1, 32, 2, 16
    q = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(batch, seq, heads, device="cuda", dtype=torch.float32) * 0.5
    beta = torch.rand(batch, seq, heads, device="cuda", dtype=torch.bfloat16)

    q_e, k_e, v_e, g_e, b_e = _clone(q, k, v, g, beta)
    q_o, k_o, v_o, g_o, b_o = _clone(q, k, v, g, beta)
    # FLA ignores ``chunk_size``. Compare at the vendor / eager default (64).
    out_e, _ = eager(
        q_e,
        k_e,
        v_e,
        g_e,
        b_e,
        use_qk_l2norm_in_kernel=True,
    )
    out_o, _ = other(
        q_o,
        k_o,
        v_o,
        g_o,
        b_o,
        use_qk_l2norm_in_kernel=True,
    )
    assert torch.allclose(out_e, out_o, atol=GDN_CHUNK_ATOL, rtol=GDN_CHUNK_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(q_e.grad, q_o.grad, atol=GDN_CHUNK_GRAD_ATOL, rtol=GDN_CHUNK_GRAD_RTOL)
    assert torch.allclose(k_e.grad, k_o.grad, atol=GDN_CHUNK_GRAD_ATOL, rtol=GDN_CHUNK_GRAD_RTOL)
    assert torch.allclose(v_e.grad, v_o.grad, atol=GDN_CHUNK_GRAD_ATOL, rtol=GDN_CHUNK_GRAD_RTOL)


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not 90 <= get_gpu_compute_capability() <= 100,
    reason="flash_qla requires an NVIDIA GPU from SM90 through SM100",
)
def test_chunk_gated_delta_rule_flash_qla_matches_fla():
    pytest.importorskip("flash_qla")
    fla = resolve_op("chunk_gated_delta_rule", "standard", "fla").wrapper
    other = resolve_op("chunk_gated_delta_rule", "standard", "flash_qla").wrapper
    torch.manual_seed(0)
    batch, seq, heads, dim = 1, 64, 4, 128
    q = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(batch, seq, heads, device="cuda", dtype=torch.float32).abs() * 0.5
    beta = torch.rand(batch, seq, heads, device="cuda", dtype=torch.bfloat16)

    out_fla, _ = fla(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    out_qla, _ = other(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert torch.allclose(out_fla, out_qla, atol=GDN_CHUNK_ATOL, rtol=GDN_CHUNK_RTOL)


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not 90 <= get_gpu_compute_capability() <= 100,
    reason="flash_qla requires an NVIDIA GPU from SM90 through SM100",
)
def test_chunk_gated_delta_rule_flash_qla_matches_eager():
    pytest.importorskip("flash_qla")
    eager = resolve_op("chunk_gated_delta_rule", "standard", "eager").wrapper
    other = resolve_op("chunk_gated_delta_rule", "standard", "flash_qla").wrapper
    torch.manual_seed(2)
    batch, seq, heads, dim = 1, 32, 2, 16
    q = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(batch, seq, heads, device="cuda", dtype=torch.float32) * 0.5
    beta = torch.rand(batch, seq, heads, device="cuda", dtype=torch.bfloat16)

    q_e, k_e, v_e, g_e, b_e = _clone(q, k, v, g, beta)
    q_o, k_o, v_o, g_o, b_o = _clone(q, k, v, g, beta)
    # FlashQLA ignores ``chunk_size``. Compare at the vendor / eager default (64).
    out_e, _ = eager(q_e, k_e, v_e, g_e, b_e, use_qk_l2norm_in_kernel=True)
    out_o, _ = other(q_o, k_o, v_o, g_o, b_o, use_qk_l2norm_in_kernel=True)
    assert torch.allclose(out_e, out_o, atol=GDN_CHUNK_ATOL, rtol=GDN_CHUNK_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(q_e.grad, q_o.grad, atol=GDN_CHUNK_GRAD_ATOL, rtol=GDN_CHUNK_GRAD_RTOL)
    assert torch.allclose(k_e.grad, k_o.grad, atol=GDN_CHUNK_GRAD_ATOL, rtol=GDN_CHUNK_GRAD_RTOL)
    assert torch.allclose(v_e.grad, v_o.grad, atol=GDN_CHUNK_GRAD_ATOL, rtol=GDN_CHUNK_GRAD_RTOL)


@pytest.mark.parametrize(
    "lengths,chunk_size,num_heads",
    [
        ([128, 256], 64, 4),
        ([64, 128, 32], 64, 8),
    ],
)
def test_precompute_varlen_metadata_matches_ensure(
    lengths: list[int],
    chunk_size: int,
    num_heads: int,
) -> None:
    from veomni.ops.kernels.gated_delta_rule.chunk_gated_delta_rule.standard import npu_ascendc as m

    cu_seqlens = torch.cumsum(torch.tensor((0,) + tuple(lengths), dtype=torch.long), dim=0)
    cu_seqlens_list, chunk_indices, chunk_indices_list = m.precompute_varlen_metadata(
        cu_seqlens=cu_seqlens,
        num_heads=num_heads,
        chunk_size=chunk_size,
    )
    g = torch.zeros(1, sum(lengths), num_heads)
    _, ref_list, ref_tensor, ref_list_dict = m._ensure_varlen_metadata(g, cu_seqlens, chunk_size)

    assert cu_seqlens_list == ref_list
    assert set(chunk_indices) == set(ref_tensor)
    assert set(chunk_indices_list) == set(ref_list_dict)
    for key, pre_t in chunk_indices.items():
        ref_t = ref_tensor[key]
        if pre_t is None:
            assert ref_t is None
        else:
            assert ref_t is not None
            assert pre_t.tolist() == ref_t.tolist()
    for key, pre_l in chunk_indices_list.items():
        assert pre_l == ref_list_dict[key]


def test_ensure_varlen_metadata_reuses_precomputed_tables() -> None:
    from veomni.ops.kernels.gated_delta_rule.chunk_gated_delta_rule.standard import npu_ascendc as m

    cu_seqlens = torch.cumsum(torch.tensor((0, 64, 192), dtype=torch.long), dim=0)
    num_heads = 4
    g = torch.zeros(1, 192, num_heads)
    cu_seqlens_list, chunk_indices, chunk_indices_list = m.precompute_varlen_metadata(
        cu_seqlens=cu_seqlens,
        num_heads=num_heads,
    )
    _, got_list, got_tensor, got_list_dict = m._ensure_varlen_metadata(
        g,
        cu_seqlens,
        64,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
    )
    assert got_list == cu_seqlens_list
    for key, tensor in chunk_indices.items():
        if tensor is None:
            assert got_tensor[key] is None
        else:
            assert got_tensor[key] is tensor
    for key, values in chunk_indices_list.items():
        assert got_list_dict[key] == values


def test_npu_ascendc_missing_fla_npu_raises_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import sys

    from veomni.ops.kernels.gated_delta_rule.chunk_gated_delta_rule.standard import npu_ascendc as m

    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fla_npu" or name.startswith("fla_npu."):
            raise ModuleNotFoundError("No module named 'fla_npu'", name="fla_npu")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    sys.modules.pop("fla_npu", None)

    with pytest.raises(RuntimeError, match="npu_ascendc"):
        m._ensure_fla_npu_registered()
