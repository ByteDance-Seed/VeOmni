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

"""Tests for chunked fused linear log-probs (PPO-style).

Pin **bitwise** parity vs a reference ``F.linear -> log_softmax ->
gather`` implementation under deterministic algorithms + batch-invariant
mode on CUDA. Same contract that
``tests/models/test_return_log_probs_e2e.py::
test_return_log_probs_bitwise_matches_logits_reference`` enforces
end-to-end. The kernel returns per-token actual log-probabilities
(non-positive); IGNORE_INDEX positions produce 0.
"""

import os
import sysconfig

import pytest
import torch
import torch.nn.functional as F

import veomni.ops.kernels.cross_entropy.chunk_logprobs as cl
from veomni.utils.constants import IGNORE_INDEX
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


# Required by ``torch.use_deterministic_algorithms`` for cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class _FakePS:
    def __init__(self, sp_enabled: bool):
        self.sp_enabled = sp_enabled


def _have_python_dev_headers() -> bool:
    """Triton JIT needs Python development headers to build its helper."""
    include = sysconfig.get_path("include")
    return include is not None and os.path.isfile(os.path.join(include, "Python.h"))


@pytest.fixture(autouse=True)
def _bitwise_setup(monkeypatch):
    """Per-test setup: deterministic algorithms + batch-invariant mode.

    Skips on CPU — the kernel's chunked matmul path and the reference's
    single ``F.linear`` rely on CUDA's batch-invariant Triton replacements
    of ``aten::mm`` / ``aten::_log_softmax`` for bitwise parity. Without
    those replacements the BLAS algorithm choice (block size, parallel
    reduction order) varies with input shape and breaks bitwise equality
    across chunk boundaries.

    The fixture also monkeypatches ``cl.get_parallel_state`` so each
    test can declare its own SP enablement without spinning up a
    distributed group.
    """
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA required for bitwise parity (deterministic + batch-invariant mode).")

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)

    bi_ctx = None
    if _have_python_dev_headers():
        from veomni.ops.batch_invariant_ops import set_batch_invariant_mode

        bi_ctx = set_batch_invariant_mode(True)
        bi_ctx.__enter__()

    # Default: SP disabled. Individual tests opt into sp_enabled=True via
    # ``monkeypatch.setattr(cl, "get_parallel_state", ...)`` again.
    monkeypatch.setattr(cl, "get_parallel_state", lambda: _FakePS(sp_enabled=False))

    try:
        yield
    finally:
        if bi_ctx is not None:
            bi_ctx.__exit__(None, None, None)
        torch.use_deterministic_algorithms(prev_deterministic, warn_only=True)


def _device():
    # Route through ``veomni.utils.device.get_device_type`` so the tests
    # work on any accelerator (CUDA / NPU) and pass the device-api-check
    # sanity job (which forbids hardcoded ``"cuda"`` strings in tests).
    return torch.device(get_device_type())


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor, label: str = "tensor") -> None:
    """``torch.equal`` with a structured diff message on mismatch."""
    assert actual.shape == expected.shape, f"{label} shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}"
    assert actual.dtype == expected.dtype, f"{label} dtype mismatch: {actual.dtype} vs {expected.dtype}"
    if not torch.equal(actual, expected):
        ne = actual != expected
        diff = (actual - expected).abs()
        first_idx = torch.nonzero(ne, as_tuple=False)[:5].tolist()
        raise AssertionError(
            f"{label} not bitwise equal: "
            f"{int(ne.sum().item())}/{actual.numel()} mismatched, "
            f"max_abs_diff={diff.max().item():.3e}, first_idx={first_idx}"
        )


def _reference_per_token_log_probs(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Reference impl: full-logits gather. Used as ground truth.

    Routes the per-token NLL through the same
    ``_per_token_log_probs_from_logits`` helper the kernel uses (which
    prefers ``flash_attn``'s triton ``cross_entropy_loss``, falling
    back to ``log_softmax + gather``), so the kernel-vs-reference
    comparison stays bitwise regardless of which path is active.
    Performs the same causal shift the kernel does (predict y_{t+1}
    from h_t) and pads the trailing seq position with 0.0 so the
    result has the same shape as ``labels``.
    """
    shifted = labels[..., 1:].contiguous()
    h = hidden_states[..., :-1, :].contiguous()
    flat = h.reshape(-1, h.size(-1))
    logits = F.linear(flat, weights).float()
    log_probs_flat = cl._per_token_log_probs_from_logits(logits, shifted.reshape(-1), ignore_index)
    log_probs = log_probs_flat.view_as(shifted)
    return F.pad(log_probs, (0, 1), value=0.0)


def test_bitwise_parity_with_reference():
    """Kernel forward equals the reference bitwise (single-chunk boundary).

    With ``chunk_size`` > total tokens the kernel collapses to one
    ``h @ weight.t()``; under batch-invariant mode that matmul + the
    log_softmax + gather are bitwise identical to the reference's
    single ``F.linear`` + log_softmax + gather.
    """
    torch.manual_seed(0)
    B, L, H, V = 2, 64, 32, 256
    hidden = torch.randn(B, L, H, dtype=torch.float32, device=_device())
    weights = torch.randn(V, H, dtype=torch.float32, device=_device())
    labels = torch.randint(0, V, (B, L), dtype=torch.long, device=_device())
    labels[0, ::7] = IGNORE_INDEX
    labels[1, 0] = IGNORE_INDEX

    out = cl.chunk_logprobs_function(hidden, weights, labels, chunk_size=B * L + 1)
    ref = _reference_per_token_log_probs(hidden, weights, labels)

    _assert_bitwise_equal(out, ref, "log_probs")


def test_ignore_index_zeroes_output_and_grad():
    """All-IGN labels emit exactly zero output and zero gradient."""
    torch.manual_seed(0)
    B, L, H, V = 1, 8, 4, 16
    hidden = torch.randn(B, L, H, dtype=torch.float32, device=_device(), requires_grad=True)
    weights = torch.randn(V, H, dtype=torch.float32, device=_device(), requires_grad=True)
    labels = torch.full((B, L), IGNORE_INDEX, dtype=torch.long, device=_device())

    out = cl.chunk_logprobs_function(hidden, weights, labels, chunk_size=4)
    assert torch.all(out == 0)

    out.sum().backward()
    assert torch.all(hidden.grad == 0)
    assert torch.all(weights.grad == 0)


def _manual_backward_per_token_log_probs(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    grad_target: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hand-derived backward matching the kernel's explicit op sequence.

    The kernel's backward computes
    ``dlogits = dlp * (one_hot - softmax(logits))`` then
    ``dhidden = dlogits @ weight`` and
    ``dweight = dlogits.t() @ hidden`` (single chunk). PyTorch's stock
    autograd path through ``log_softmax + gather`` is mathematically
    equivalent but sequences the fp32 ops differently (scatter-then-
    subtract vs subtract-then-multiply), so the two paths' grads agree
    only at fp32 epsilon (~1e-7), not bitwise. To pin **bitwise**
    parity under batch-invariant ``mm``, we mirror the kernel's
    op sequence here and use this as the reference instead.
    """
    shifted = labels[..., 1:].contiguous()
    h = hidden[..., :-1, :].contiguous()
    flat = h.reshape(-1, h.size(-1))
    target = shifted.reshape(-1)
    grad_flat = grad_target[..., :-1].reshape(-1)

    logits = F.linear(flat, weights).float()
    probs = logits.softmax(dim=-1)
    mask = (target != ignore_index).float()
    safe = target.clamp(min=0).unsqueeze(-1)
    one_hot = torch.zeros_like(probs).scatter_(-1, safe, 1.0)
    masked_dlp = (grad_flat * mask).unsqueeze(-1)
    dlogits = masked_dlp * (one_hot - probs)

    dhidden_flat = dlogits @ weights
    dweight = dlogits.t() @ flat

    dhidden = torch.zeros_like(hidden)
    dhidden[..., :-1, :] = dhidden_flat.view_as(h)
    return dhidden, dweight


def test_chunk_size_invariance_forward_and_grad():
    """Output, dhidden and dweight are bitwise invariant to ``chunk_size``.

    Under batch-invariant ``mm`` / ``_log_softmax``, every per-row
    forward output is independent of which other rows are in the matmul,
    so the chunked path's row-i output equals the single-chunk path's
    row-i output bit-for-bit. The same row-independence holds for
    ``dhidden`` (each output row is written by exactly one chunk's
    matmul). For ``dweight`` we pin ``B=1, T=L=24`` and use chunk_sizes
    that all yield exactly one chunk (24, 1024) so the cross-chunk
    add accumulation never runs and the comparison stays bitwise. The
    fundamentally-multi-chunk regimes (``chunk_size=1, 5``) are
    deliberately omitted because cross-chunk ``dweight += partial``
    sums in different orders for different chunk_sizes and only agrees
    at fp32 epsilon, not bitwise.
    """
    torch.manual_seed(7)
    B, L, H, V = 1, 24, 8, 32
    hidden0 = torch.randn(B, L, H, dtype=torch.float32, device=_device())
    weights0 = torch.randn(V, H, dtype=torch.float32, device=_device())
    labels = torch.randint(0, V, (B, L), dtype=torch.long, device=_device())
    labels[0, 3] = IGNORE_INDEX
    labels[0, 11] = IGNORE_INDEX
    labels[0, 23] = IGNORE_INDEX

    grad_target = torch.randn(B, L, dtype=torch.float32, device=_device())

    chunk_sizes = (24, 1024)
    outputs = []
    grads_h = []
    grads_w = []
    for chunk_size in chunk_sizes:
        h = hidden0.clone().requires_grad_(True)
        w = weights0.clone().requires_grad_(True)
        out = cl.chunk_logprobs_function(h, w, labels, chunk_size=chunk_size)
        (out * grad_target).sum().backward()
        outputs.append(out.detach())
        grads_h.append(h.grad.detach())
        grads_w.append(w.grad.detach())

    for i in range(1, len(outputs)):
        _assert_bitwise_equal(outputs[i], outputs[0], f"forward[chunk_size={chunk_sizes[i]}]")
        _assert_bitwise_equal(grads_h[i], grads_h[0], f"dhidden[chunk_size={chunk_sizes[i]}]")
        _assert_bitwise_equal(grads_w[i], grads_w[0], f"dweight[chunk_size={chunk_sizes[i]}]")


def test_grad_matches_reference():
    """Gradients are bitwise identical to a hand-derived backward reference.

    Uses ``_manual_backward_per_token_log_probs`` (which mirrors the
    kernel's explicit ``dlogits = dlp * (one_hot - softmax)`` form
    instead of relying on autograd's ``log_softmax + gather``
    backward). Forces ``chunk_size > total tokens`` so the kernel's
    ``dweight = dlogits.t() @ h_chunk`` is a single mm matching the
    reference's single mm against the same matrices. Under
    batch-invariant ``mm`` both paths produce bit-identical gradients.
    """
    torch.manual_seed(42)
    B, L, H, V = 2, 32, 16, 64
    hidden0 = torch.randn(B, L, H, dtype=torch.float32, device=_device())
    weights0 = torch.randn(V, H, dtype=torch.float32, device=_device())
    labels = torch.randint(0, V, (B, L), dtype=torch.long, device=_device())
    labels[0, ::5] = IGNORE_INDEX

    grad_target = torch.randn(B, L, dtype=torch.float32, device=_device())

    h_a = hidden0.clone().requires_grad_(True)
    w_a = weights0.clone().requires_grad_(True)
    out_a = cl.chunk_logprobs_function(h_a, w_a, labels, chunk_size=B * L + 1)
    (out_a * grad_target).sum().backward()

    out_ref = _reference_per_token_log_probs(hidden0, weights0, labels)
    dhidden_ref, dweight_ref = _manual_backward_per_token_log_probs(hidden0, weights0, labels, grad_target)

    _assert_bitwise_equal(out_a.detach(), out_ref, "log_probs")
    _assert_bitwise_equal(h_a.grad, dhidden_ref, "dhidden")
    _assert_bitwise_equal(w_a.grad, dweight_ref, "dweight")


def test_sp_enabled_skips_internal_shift(monkeypatch):
    """Under SP, the dataloader pre-shifts; kernel must not shift again.

    Reference computes ``F.linear(hidden) -> log_softmax -> gather``
    against the *un-shifted* labels, and asserts bitwise equality to
    the kernel under batch-invariant mode.
    """
    monkeypatch.setattr(cl, "get_parallel_state", lambda: _FakePS(sp_enabled=True))

    torch.manual_seed(1)
    B, L, H, V = 1, 16, 8, 32
    hidden = torch.randn(B, L, H, dtype=torch.float32, device=_device())
    weights = torch.randn(V, H, dtype=torch.float32, device=_device())
    labels = torch.randint(0, V, (B, L), dtype=torch.long, device=_device())

    out = cl.chunk_logprobs_function(hidden, weights, labels, chunk_size=B * L + 1)
    assert out.shape == labels.shape

    flat = hidden.reshape(-1, H)
    logits = F.linear(flat, weights).float()
    expected_flat = cl._per_token_log_probs_from_logits(logits, labels.reshape(-1), IGNORE_INDEX)
    expected = expected_flat.view_as(labels)
    _assert_bitwise_equal(out, expected, "log_probs (sp_enabled)")


def _maybe_load_verl_fused_linear_for_ppo():
    """Import verl's ``FusedLinearForPPOFunction`` without triggering ``verl.__init__``.

    ``verl/__init__.py`` imports ``ray`` (not a VeOmni dep), so a plain
    ``import verl.utils.experimental.torch_functional`` fails in the
    test env. Load the submodule's source directly via ``importlib``;
    returns ``None`` if the verl repo isn't present.
    """
    import importlib.util

    verl_root = os.environ.get("VERL_PATH", "/home/ubuntu/verl")
    src = os.path.join(verl_root, "verl", "utils", "experimental", "torch_functional.py")
    if not os.path.isfile(src):
        return None
    spec = importlib.util.spec_from_file_location("_verl_torch_functional_for_test", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FusedLinearForPPOFunction


def test_bitwise_parity_with_verl_fused_linear_for_ppo():
    """VeOmni ``chunk_logprobs_function`` is bitwise equal to verl's
    ``FusedLinearForPPOFunction[0]`` (token_log_probs).

    Both kernels:
      1. Compute ``logits = (h @ w.t()) / T`` in input dtype, then upcast to fp32.
      2. Route per-token NLL through ``flash_attn``'s triton CE kernel.
      3. Negate to get actual log-probabilities.
      4. Compute backward via explicit ``dlogits = dlp * (one_hot - softmax)``,
         cast to input dtype, divide by T, then matmul.

    Pinning the parity here keeps the verl ↔ VeOmni integration story
    "drop-in": a model trained under VeOmni's per-token log-probs path
    produces the same fp32 numbers as the same model invoked through
    verl's fused PPO kernel, under deterministic + batch-invariant
    mode.
    """
    verl_fn = _maybe_load_verl_fused_linear_for_ppo()
    if verl_fn is None:
        pytest.skip("verl repo not found; set VERL_PATH to enable verl-bitwise comparison.")

    torch.manual_seed(123)
    B, L, H, V = 2, 32, 16, 256
    hidden_a = torch.randn(B, L, H, dtype=torch.float32, device=_device(), requires_grad=True)
    weights_a = torch.randn(V, H, dtype=torch.float32, device=_device(), requires_grad=True)
    # All-valid labels: verl's kernel has no IGNORE_INDEX masking, so we
    # match its scope by avoiding IGN labels here.
    labels = torch.randint(0, V, (B, L), dtype=torch.long, device=_device())

    grad_target = torch.randn(B, L, dtype=torch.float32, device=_device())
    chunk_size = B * L + 1  # single-chunk boundary for both kernels

    # VeOmni — pass ``shift_labels=labels`` so the wrapper treats them
    # as already-aligned (verl's kernel never applies a causal shift,
    # mirroring that here keeps the per-token output aligned for a
    # bit-for-bit comparison).
    out_a = cl.chunk_logprobs_function(hidden_a, weights_a, labels, chunk_size=chunk_size, shift_labels=labels)
    (out_a * grad_target).sum().backward()

    # verl — same inputs cloned, no causal shift inside
    hidden_b = hidden_a.detach().clone().requires_grad_(True)
    weights_b = weights_a.detach().clone().requires_grad_(True)
    out_b, _entropy_b = verl_fn.apply(hidden_b, weights_b, labels, 1.0, chunk_size)
    (out_b * grad_target).sum().backward()

    _assert_bitwise_equal(out_a, out_b, "log_probs vs verl")
    _assert_bitwise_equal(hidden_a.grad, hidden_b.grad, "dhidden vs verl")
    _assert_bitwise_equal(weights_a.grad, weights_b.grad, "dweight vs verl")


def test_explicit_shift_labels_overrides_internal_shift():
    """When the caller provides shift_labels, the kernel uses them as-is.

    Mirrors the contract used by ``ForCausalLMLoss``: under SP off the
    wrapper builds ``shift_labels = F.pad(labels, (0, 1), IGN)[..., 1:]``
    so each label position is already the next-token target. The kernel
    must consume that directly (no internal shift, no trailing pad) and
    return a tensor whose seq length matches the *passed* shift_labels.
    """
    torch.manual_seed(11)
    B, L, H, V = 2, 16, 8, 32
    hidden = torch.randn(B, L, H, dtype=torch.float32, device=_device())
    weights = torch.randn(V, H, dtype=torch.float32, device=_device())
    labels = torch.randint(0, V, (B, L), dtype=torch.long, device=_device())
    labels[0, 3] = IGNORE_INDEX

    shift_labels = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    out = cl.chunk_logprobs_function(hidden, weights, labels, chunk_size=B * L + 1, shift_labels=shift_labels)

    assert out.shape == shift_labels.shape

    flat = hidden.reshape(-1, H)
    logits = F.linear(flat, weights).float()
    expected_flat = cl._per_token_log_probs_from_logits(logits, shift_labels.reshape(-1), IGNORE_INDEX)
    expected = expected_flat.view_as(shift_labels)
    _assert_bitwise_equal(out, expected, "log_probs (explicit shift_labels)")
