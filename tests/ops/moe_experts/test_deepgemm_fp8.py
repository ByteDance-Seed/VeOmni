# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Hopper/Blackwell FP8 numerics against independent quantized and BF16 references."""

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.ops.moe_experts.reference import standard_ep_reference
from veomni.ops import resolve_op


_DEVICE_CONTEXT_LAUNCH = LaunchSpec(nproc=2)


@pytest.fixture(autouse=True)
def _supported_gpu():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA Hopper or Blackwell is required.")
    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0)):
        pytest.skip("The DeepGEMM FP8 training backend targets SM90/SM100.")
    pytest.importorskip("deep_gemm")


def _relative_error(actual, expected, limit, label):
    assert torch.isfinite(actual).all(), label
    error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    assert error < limit, f"{label}: relative L2 {error.item():.6f} >= {limit}"


def _scale_for_device(scales):
    if torch.cuda.get_device_capability(scales.device) == (10, 0):
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
    return scales


def _quantized_rows(x):
    blocks = x.float().reshape(x.shape[0], -1, 128)
    scales = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    scales = _scale_for_device(scales)
    return ((blocks / scales).to(torch.float8_e4m3fn).float() * scales).reshape(x.shape)


def _quantized_columns(x):
    padded = F.pad(x.float(), (0, 0, 0, (-x.shape[0]) % 128))
    blocks = padded.reshape(-1, 128, x.shape[1])
    scales = blocks.abs().amax(1, keepdim=True).clamp_min(1e-4) / 448
    scales = _scale_for_device(scales)
    return ((blocks / scales).to(torch.float8_e4m3fn).float() * scales).reshape(padded.shape)[: x.shape[0]]


def _quantized_weight(w):
    n, k = w.shape
    blocks = w.float().reshape(n // 128, 128, k // 128, 128)
    scales = blocks.abs().amax((1, 3), keepdim=True).clamp_min(1e-4) / 448
    scales = _scale_for_device(scales)
    return ((blocks / scales).to(torch.float8_e4m3fn).float() * scales).reshape(n, k)


class _ReferenceLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return (_quantized_rows(x) @ _quantized_weight(w).T).bfloat16()

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        dx = (_quantized_rows(g) @ _quantized_weight(w)).bfloat16()
        dw = (_quantized_columns(g).T @ _quantized_columns(x)).bfloat16()
        return dx, dw


def _quantized_moe_reference(x, routing, selected, w1, w2, limit):
    output = torch.zeros_like(x)
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(selected == expert)
        if rows.numel() == 0:
            continue
        gate, up = _ReferenceLinear.apply(x[rows], w1[expert]).chunk(2, dim=-1)
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        values = _ReferenceLinear.apply(F.silu(gate) * up, w2[expert])
        output.index_add_(0, rows, (values * routing[rows, slots, None]).to(x.dtype))
    return output


def test_dynamic_routed_sizes_reuse_preparation_kernels():
    """Changing token counts reuses row-quantization and SwiGLU binaries."""
    from veomni.ops.kernels.moe_experts.shared.deepgemm_fp8 import (
        GroupedLayout,
        RecomputedSwiGLU,
        _row_quant_kernel,
        _swiglu_kernel,
        quantize_rows,
    )

    device = torch.cuda.current_device()
    cache_sizes = None
    for tokens in (129, 257, 385):
        x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        layout = GroupedLayout.from_counts(torch.tensor([tokens]), x.device)
        q, scales = quantize_rows(x, layout)
        restored = q[layout.unpad].float() * scales[layout.unpad].repeat_interleave(128, dim=1)
        torch.testing.assert_close(restored, _quantized_rows(x.detach()), rtol=0, atol=0)
        output = RecomputedSwiGLU.apply(x, None)
        grad = torch.randn_like(output)
        dx = torch.autograd.grad(output, x, grad)[0]
        gate, up = x.chunk(2, dim=-1)
        expected = F.silu(gate) * up
        expected_dx = torch.autograd.grad(expected, x, grad)[0]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(dx, expected_dx, rtol=0.01, atol=0.015625)
        # Inspect actual compiled binaries, not annotations on the Python signature.
        current = tuple(len(kernel.device_caches[device][0]) for kernel in (_row_quant_kernel, _swiglu_kernel))
        if cache_sizes is None:
            cache_sizes = current
        else:
            assert current == cache_sizes, "Dynamic routed sizes created additional kernel specializations."


@pytest.mark.parametrize("counts", [[0, 17, 129, 33], [128, 256, 0, 1]])
def test_grouped_linear_quantization_axes(counts):
    from deep_gemm.testing import calc_diff

    from veomni.ops.kernels.moe_experts.shared.deepgemm_fp8 import GroupedLayout, GroupedLinear

    torch.manual_seed(42)
    # Adjacent experts have very different ranges; sharing a token block across
    # their boundary would corrupt both the FP8 reference and the BF16 comparison.
    x = torch.cat([torch.randn(n, 256, device="cuda") * (0.1 * (i + 1)) for i, n in enumerate(counts)])
    x = x.bfloat16().requires_grad_()
    w = (torch.randn(4, 384, 256, device="cuda") * 0.04).bfloat16().requires_grad_()
    g = (torch.randn(sum(counts), 384, device="cuda") * 0.1).bfloat16()
    y = GroupedLinear.apply(x, w, GroupedLayout.from_counts(torch.tensor(counts), x.device))
    dx, dw = torch.autograd.grad(y, (x, w), g)
    offset = 0
    for i, n in enumerate(counts):
        if not n:
            assert torch.count_nonzero(dw[i]) == 0
            continue
        xe, ge = x[offset : offset + n].detach(), g[offset : offset + n]
        we = w[i].detach()
        qx, qg, qw = _quantized_rows(xe), _quantized_rows(ge), _quantized_weight(we)
        references = (
            (y[offset : offset + n], qx @ qw.T, xe.float() @ we.float().T, "forward"),
            (dx[offset : offset + n], qg @ qw, ge.float() @ we.float(), "dgrad"),
            (dw[i], _quantized_columns(ge).T @ _quantized_columns(xe), ge.float().T @ xe.float(), "wgrad"),
        )
        for actual, quantized, bf16, label in references:
            _relative_error(actual, quantized.bfloat16(), 0.005, f"quantized {label}")
            # Upstream's FP8 GEMM gate is a normalized SQUARED error, not
            # relative L2. Test each GEMM against the same 0.001 criterion.
            diff = calc_diff(actual, bf16.bfloat16())
            assert diff < 0.001, f"BF16 {label}: upstream calc_diff={diff}"
        offset += n


@pytest.mark.parametrize("input_grad,weight_grad", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("counts", [[511, 513], [1, 1, 1, 1]])
def test_grouped_linear_compact_saved_state_and_retained_graph(input_grad, weight_grad, counts):
    """Saved state stays compact while frozen operands and repeated backward work."""
    from veomni.ops.kernels.moe_experts.shared.deepgemm_fp8 import GroupedLayout, GroupedLinear

    torch.manual_seed(57)
    x = torch.randn(sum(counts), 256, device="cuda", dtype=torch.bfloat16).requires_grad_(input_grad)
    w = (torch.randn(len(counts), 384, 256, device="cuda") * 0.04).bfloat16().requires_grad_(weight_grad)
    saved_bytes = 0

    def pack(tensor):
        nonlocal saved_bytes
        saved_bytes += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        y = GroupedLinear.apply(x, w, GroupedLayout.from_counts(torch.tensor(counts), x.device))
    # Include ragged-group alignment and scale metadata in the budget. Saving
    # the BF16 input as well as the FP8 weight transpose would exceed it.
    if sum(counts) > 128:
        assert saved_bytes < 1.2 * (x.numel() + w.numel())
    else:
        # Padding tiny groups must not inflate saved activations beyond BF16.
        assert saved_bytes < 1.01 * w.numel() + x.numel() * x.element_size()
    g = torch.randn_like(y)
    leaves = tuple(value for value in (x, w) if value.requires_grad)
    first = torch.autograd.grad(y, leaves, g, retain_graph=True)
    second = torch.autograd.grad(y, leaves, g)
    for actual, expected in zip(second, first):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("limit", [None, 0.7, 1.0, 10.0])
@pytest.mark.parametrize("strided", [False, True])
def test_recomputed_swiglu_matches_eager_rounding(limit, strided):
    from veomni.ops.kernels.moe_experts.shared.deepgemm_fp8 import RecomputedSwiGLU

    torch.manual_seed(2026)
    projected = (torch.randn(131, 256, device="cuda") * 10).bfloat16()
    if strided:
        projected = projected.T.contiguous().T
    projected.requires_grad_()
    # Exact clamp boundaries must pass the gradient through, as torch.clamp does.
    with torch.no_grad():
        boundary = 10 if limit is None else limit
        projected[0, :3] = projected[0, 128:131] = torch.tensor([-boundary, 0, boundary], device="cuda")
    source = projected.detach().clone()
    gate, up = projected.chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    expected = F.silu(gate) * up
    actual = RecomputedSwiGLU.apply(projected, limit)
    grad = torch.randn_like(expected)
    if strided:
        grad = grad.T.contiguous().T
    expected_grad = torch.autograd.grad(expected, projected, grad)[0]
    actual_grad = torch.autograd.grad(actual, projected, grad)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    torch.testing.assert_close(projected, source, atol=0, rtol=0)


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_chunked_wgrad_matches_full_expert_group(monkeypatch, chunk_size):
    import veomni.ops.kernels.moe_experts.shared.deepgemm_fp8 as fp8

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Only SM90 uses the bounded FP32 wgrad workspace.")
    torch.manual_seed(2026)
    # Cover an empty first chunk, an empty final chunk, a partial expert group,
    # and a reused accumulator whose previous expert had a nonzero gradient.
    counts = [0, 0, 129, 17, 0, 0, 3, 0, 0]
    n, k = 384, 256
    x = torch.randn(sum(counts), k, device="cuda", dtype=torch.bfloat16).requires_grad_()
    w = (torch.randn(len(counts), n, k, device="cuda") * 0.04).bfloat16().requires_grad_()
    g = torch.randn(sum(counts), n, device="cuda", dtype=torch.bfloat16)
    layout = fp8.GroupedLayout.from_counts(torch.tensor(counts), x.device)
    y = fp8.GroupedLinear.apply(x, w, layout)
    full = torch.autograd.grad(y, (x, w), g, retain_graph=True)
    monkeypatch.setattr(fp8, "_WGRAD_WORKSPACE_BYTES", chunk_size * n * k * 4)
    chunked = torch.autograd.grad(y, (x, w), g)
    for actual, expected in zip(chunked, full):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.count_nonzero(chunked[1][torch.tensor(counts, device="cuda") == 0]) == 0


@pytest.mark.parametrize("experts", [7, 129, 257])
def test_sm100_direct_wgrad_matches_fp32_accumulation(experts, monkeypatch):
    import deep_gemm

    from veomni.ops.kernels.moe_experts.shared import deepgemm_fp8 as fp8

    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Direct K-grouped BF16 output requires SM100.")
    torch.manual_seed(2026)
    counts = [0] * experts
    # Includes empty groups, unaligned token counts, a tail beyond 128 groups,
    # and an entirely empty middle chunk when experts=257.
    counts[1], counts[-1] = 17, 129
    x = torch.randn(sum(counts), 256, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(sum(counts), 128, device="cuda", dtype=torch.bfloat16)
    layout = fp8.GroupedLayout.from_counts(torch.tensor(counts), x.device)
    qa, sa = fp8.quantize_columns(g, layout)
    qb, sb = fp8.quantize_columns(x, layout)
    expected = torch.zeros(experts, 128, 256, device=x.device, dtype=torch.float32)
    begin = 0
    for first in range(0, experts, 128):
        last = min(first + 128, experts)
        ks = layout.padded_counts[first:last]
        end = begin + sum(ks)
        if end > begin:
            accumulator = expected[first:last]
            deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
                (qa[begin:end], sa[begin // 128 : end // 128]),
                (qb[begin:end], sb[begin // 128 : end // 128]),
                accumulator,
                ks,
                layout.ks[first:last],
                accumulator,
                recipe=(1, 1, 128),
                use_psum_layout=False,
            )
        begin = end

    # Poison uninitialized output so skipped empty experts cannot pass by luck.
    original_empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        value = original_empty(*args, **kwargs)
        if value.dtype == torch.bfloat16:
            value.fill_(float("nan"))
        return value

    with monkeypatch.context() as patcher:
        patcher.setattr(torch, "empty", poisoned_empty)
        actual = fp8._weight_gradient(x, g, layout, expected.shape, torch.bfloat16)
    torch.testing.assert_close(actual, expected.bfloat16(), atol=0, rtol=0)
    assert torch.count_nonzero(actual[torch.tensor(counts, device=x.device) == 0]) == 0


@pytest.mark.parametrize("use_ue8m0", [False, True])
def test_quantizers_match_deepgemm_reference(use_ue8m0):
    from deep_gemm.utils import per_block_cast_to_fp8, per_channel_cast_to_fp8, per_token_cast_to_fp8

    from veomni.ops.kernels.moe_experts.shared.deepgemm_fp8 import (
        GroupedLayout,
        quantize_columns,
        quantize_rows,
        quantize_weights,
    )

    torch.manual_seed(2026)
    counts = [0, 1, 127, 128, 129, 511, 513]
    x = torch.cat(
        [torch.randn(n, 256, device="cuda", dtype=torch.bfloat16) * 10 ** (i - 5) for i, n in enumerate(counts)]
    )
    x = x.T.contiguous().T  # Quantizers must honor non-contiguous input strides.
    layout = GroupedLayout.from_counts(torch.tensor(counts), x.device)
    layout.use_ue8m0 = use_ue8m0
    qr, sr = quantize_rows(x, layout)
    qc, sc = quantize_columns(x, layout)
    w = torch.randn(len(counts), 384, 256, device="cuda", dtype=torch.bfloat16) * 0.02
    qw, sw, qwt, swt = quantize_weights(w, use_ue8m0=use_ue8m0)
    source, target = 0, 0
    for expert, (n, p) in enumerate(zip(counts, layout.padded_counts)):
        expected_w, expected_s = per_block_cast_to_fp8(w[expert], use_ue8m0=use_ue8m0)
        torch.testing.assert_close(qw[expert].float(), expected_w.float(), atol=0, rtol=0)
        torch.testing.assert_close(sw[expert], expected_s, atol=0, rtol=0)
        torch.testing.assert_close(qwt[expert].float(), expected_w.float().T, atol=0, rtol=0)
        torch.testing.assert_close(swt[expert], expected_s.T, atol=0, rtol=0)
        if n:
            padded = F.pad(x[source : source + n], (0, 0, 0, p - n)).contiguous()
            expected_r, expected_rs = per_token_cast_to_fp8(padded, use_ue8m0=use_ue8m0)
            expected_c, expected_cs = per_channel_cast_to_fp8(padded, use_ue8m0=use_ue8m0)
            torch.testing.assert_close(qr[target : target + p].float(), expected_r.float(), atol=0, rtol=0)
            torch.testing.assert_close(sr[target : target + p], expected_rs, atol=0, rtol=0)
            if use_ue8m0:
                packed_columns = qc[target : target + p]
                column_scales = sc[target // 128 : (target + p) // 128]
            else:
                packed_columns = qc[target * 256 : (target + p) * 256].reshape(256, p).T
                column_scales = sc[:, target // 128 : (target + p) // 128].T
            torch.testing.assert_close(packed_columns.float(), expected_c.float(), atol=0, rtol=0)
            torch.testing.assert_close(column_scales, expected_cs, atol=0, rtol=0)
        source += n
        target += p


@pytest.mark.parametrize("merged,limit,routing_dtype", [(True, None, torch.float32), (False, 0.7, torch.bfloat16)])
def test_full_moe_forward_backward(merged, limit, routing_dtype):
    torch.manual_seed(19)
    e, t, h, intermediate = 4, 139, 256, 128
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)
    w1 = (torch.randn(e, 2 * intermediate, h, device="cuda") * 0.03).bfloat16()
    w2 = (torch.randn(e, h, intermediate, device="cuda") * 0.03).bfloat16()
    routing = torch.rand(t, 2, device="cuda", dtype=routing_dtype)
    # Expert 3 is empty. Duplicate assignments are legal on the non-EP path.
    selected = torch.randint(0, e - 1, (t, 2), device="cuda")
    grad = torch.randn_like(x) * 0.1
    backend = resolve_op("moe_experts", "standard", "fused_deepgemm_fp8").wrapper
    empty = torch.empty(0, device="cuda")
    results = []
    for reference in ("kernel", "bf16", "quantized"):
        leaves = [a.detach().clone().requires_grad_() for a in (x, routing, w1, w2)]
        xi, ri, wi, wo = leaves
        a, b = wi.chunk(2, dim=1)
        if reference == "bf16":
            y = standard_ep_reference(xi, ri, selected, a, b, wo, num_experts=e, swiglu_limit=limit)
        elif reference == "quantized":
            y = _quantized_moe_reference(xi, ri, selected, wi, wo, limit)
        else:
            y = backend(
                xi,
                ri,
                selected,
                empty if merged else a,
                empty if merged else b,
                wo,
                wi if merged else empty,
                num_experts=e,
                swiglu_limit=limit,
            )
        results.append((y, *torch.autograd.grad(y, leaves, grad)))
    for label, actual, bf16, quantized in zip(("output", "hidden", "router", "gate_up", "down"), *results):
        _relative_error(actual, quantized, 0.01, f"quantized {label}")
        # The derivative at the clamp boundary is discontinuous; use the
        # independent FP8 oracle above for that deliberately adversarial case.
        if limit is None or label not in ("hidden", "gate_up"):
            _relative_error(actual, bf16, 0.085, f"BF16 {label}")
    assert torch.count_nonzero(results[0][-1][-1]) == 0
    assert torch.count_nonzero(results[0][-2][-1]) == 0


def test_empty_moe_has_zero_gradients():
    from veomni.ops.kernels.moe_experts.standard.deepgemm import wrapper

    x = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    routing = torch.empty(0, 2, device="cuda", requires_grad=True)
    w1 = torch.randn(2, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    empty = torch.empty(0, device="cuda")
    y = wrapper(x, routing, routing.long(), empty, empty, w2, w1, num_experts=2)
    grads = torch.autograd.grad(y.sum(), (x, routing, w1, w2))
    assert y.shape == x.shape
    assert all(torch.count_nonzero(g) == 0 for g in grads)


def test_checkpointed_fp8_gradients():
    from torch.utils.checkpoint import checkpoint

    from veomni.ops.kernels.moe_experts.standard.deepgemm import wrapper

    torch.manual_seed(77)
    x = torch.randn(257, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = (torch.randn(2, 256, 128, device="cuda", dtype=torch.bfloat16) * 0.03).requires_grad_()
    w2 = (torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16) * 0.03).requires_grad_()
    routing = torch.rand(257, 2, device="cuda", requires_grad=True)
    indices = torch.tensor([0, 1], device="cuda").expand(257, -1)
    empty = x.new_empty(0)

    def run(x, routing, w1, w2):
        return wrapper(x, routing, indices, empty, empty, w2, w1, num_experts=2)

    grad = torch.randn_like(x)
    leaves = (x, routing, w1, w2)
    ordinary = torch.autograd.grad(run(*leaves), leaves, grad)
    recomputed = torch.autograd.grad(checkpoint(run, *leaves, use_reentrant=False), leaves, grad)
    for actual, expected in zip(recomputed, ordinary):
        _relative_error(actual, expected, 0.005, "checkpoint gradient")


@pytest.mark.parametrize("bad", ["shape", "dtype", "routing", "expert_id"])
def test_invalid_operands_fail_explicitly(bad):
    from veomni.ops.kernels.moe_experts.standard.deepgemm import wrapper

    h = 64 if bad == "shape" else 128
    x = torch.zeros(1, h, device="cuda", dtype=torch.float32 if bad == "dtype" else torch.bfloat16)
    w1 = torch.zeros(2, 256, h, device="cuda", dtype=torch.bfloat16)
    w2 = torch.zeros(2, h, 128, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([[2 if bad == "expert_id" else 0]], device="cuda")
    routing = torch.ones(1 if bad == "routing" else (1, 1), device="cuda")
    empty = x.new_empty(0)
    with pytest.raises(ValueError):
        wrapper(x, routing, indices, empty, empty, w2, w1, num_experts=2)


def test_quantization_offsets_above_int32():
    from veomni.ops.kernels.moe_experts.shared.deepgemm_fp8 import GroupedLayout, quantize_columns, quantize_rows

    # Only the last two rows are touched; the allocation makes their genuine
    # device addresses exceed 2^31 elements while strides themselves stay int32.
    # An unusually large stride alone would promote a Triton constexpr to int64
    # and would not expose multiplication overflow for ordinary contiguous rows.
    if torch.cuda.mem_get_info()[0] < 8 * 2**30:
        pytest.skip("Requires 8 GiB free for the large-address regression.")
    h, start = 5120, 2**31 // 5120 + 1
    x = torch.empty(start + 2, h, device="cuda", dtype=torch.bfloat16)
    x[-2:] = torch.linspace(-1, 1, 2 * h, device="cuda").reshape(2, h)
    layout = GroupedLayout.from_counts(torch.tensor([2]), x.device)
    layout.rows[:2] += start
    layout.metadata[0, 0] = start
    qc, sc = quantize_columns(x, layout)
    if layout.use_ue8m0:
        columns = qc[:2].float() * sc[0]
    else:
        columns = qc.view(h, 128).T[:2].float() * sc[:, 0]
    # Exercise overflow in Q stores as well as X loads. Invalid prefix rows
    # write zeros without reading the uninitialized prefix of X.
    padded_rows = (start + 2 + 127) // 128 * 128
    layout.rows = torch.full((padded_rows,), -1, device="cuda", dtype=torch.int32)
    layout.rows[-2:] = torch.arange(start, start + 2, device="cuda", dtype=torch.int32)
    qr, sr = quantize_rows(x, layout)
    rows = qr[-2:].float() * sr[-2:].repeat_interleave(128, dim=1)
    _relative_error(rows, _quantized_rows(x[-2:]), 0.001, "large-offset row quantization")
    _relative_error(columns, _quantized_columns(x[-2:]), 0.001, "large-offset column quantization")


def test_many_experts_with_empty_groups():
    from deep_gemm.testing import calc_diff

    from veomni.ops.kernels.moe_experts.shared.deepgemm_fp8 import GroupedLayout, GroupedLinear

    torch.manual_seed(51)
    counts = [0] * 384
    counts[0], counts[128], counts[383] = 17, 129, 1
    x = torch.randn(sum(counts), 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
    w = (torch.randn(384, 128, 128, device="cuda") * 0.04).bfloat16().requires_grad_()
    dy = torch.randn(sum(counts), 128, device="cuda", dtype=torch.bfloat16)
    y = GroupedLinear.apply(x, w, GroupedLayout.from_counts(torch.tensor(counts), x.device))
    dw = torch.autograd.grad(y, w, dy)[0]
    offset = 0
    for expert, n in enumerate(counts):
        if n:
            expected = dy[offset : offset + n].float().T @ x[offset : offset + n].float()
            assert calc_diff(dw[expert], expected.bfloat16()) < 0.001
        else:
            assert torch.count_nonzero(dw[expert]) == 0
        offset += n


@pytest.mark.launch_plan(provider=_DEVICE_CONTEXT_LAUNCH)
@pytest.mark.skipif(runner_device_count() < 2, reason="Two GPUs are required.")
def test_grouped_linear_uses_tensor_device():
    if torch.cuda.device_count() < 2 or torch.cuda.get_device_capability(1) != torch.cuda.get_device_capability(0):
        pytest.skip("Two GPUs of the same supported architecture are required.")
    # DeepGEMM caches device-bound CUfunction handles process-wide. Test device
    # guards in fresh workers, each using only the non-current device.
    mp.spawn(_non_current_device_worker, nprocs=_DEVICE_CONTEXT_LAUNCH.nproc, join=True)


def _non_current_device_worker(rank):
    from deep_gemm.testing import calc_diff

    from veomni.ops.kernels.moe_experts.shared.deepgemm_fp8 import GroupedLayout, GroupedLinear

    torch.manual_seed(104)
    operand_device = f"cuda:{1 - rank}"
    with torch.cuda.device(rank):
        x = torch.randn(129, 128, device=operand_device, dtype=torch.bfloat16).requires_grad_()
        w = (torch.randn(1, 128, 128, device=operand_device) * 0.04).bfloat16().requires_grad_()
        dy = torch.randn_like(x)
        y = GroupedLinear.apply(x, w, GroupedLayout.from_counts(torch.tensor([129]), x.device))
        dx, dw = torch.autograd.grad(y, (x, w), dy)
        assert torch.cuda.current_device() == rank
        for actual, expected in (
            (y, x.float() @ w[0].float().T),
            (dx, dy.float() @ w[0].float()),
            (dw[0], dy.float().T @ x.float()),
        ):
            assert actual.device == x.device
            assert calc_diff(actual, expected.bfloat16()) < 0.001

        from veomni.ops.kernels.moe_experts.standard.deepgemm import wrapper

        routing = torch.ones(129, 1, device=x.device, dtype=x.dtype).requires_grad_()
        selected = torch.zeros(129, 1, device=x.device, dtype=torch.int64)
        w1 = torch.cat((w.detach(), w.detach()), dim=1).requires_grad_()
        w2 = w.detach().clone().requires_grad_()
        empty = x.new_empty(0)
        actual = wrapper(x, routing, selected, empty, empty, w2, w1, num_experts=1)
        actual_grads = torch.autograd.grad(actual, (x, routing, w1, w2), dy)
        expected = _quantized_moe_reference(x, routing, selected, w1, w2, None)
        expected_grads = torch.autograd.grad(expected, (x, routing, w1, w2), dy)
        assert torch.cuda.current_device() == rank
        for label, a, b in zip(
            ("output", "input", "routing", "gate_up", "down"),
            (actual,) + actual_grads,
            (expected,) + expected_grads,
        ):
            assert a.device == x.device
            _relative_error(a, b, 0.01, f"non-current-device MoE {label}")

        # Reject a second device before reaching upstream's invalid-handle error.
        with pytest.raises(RuntimeError, match="one CUDA device per process"):
            GroupedLayout.from_counts(torch.tensor([129]), torch.device(f"cuda:{rank}"))
