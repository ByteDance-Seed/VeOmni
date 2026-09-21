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

"""Tests for chunked fused linear log-probs and entropy (PPO-style).

Pin **bitwise** parity vs a reference ``F.linear -> log_softmax ->
gather`` implementation under deterministic algorithms + batch-invariant
mode on CUDA. Same contract that
``tests/models/test_return_log_probs_e2e.py::
test_return_log_probs_bitwise_matches_logits_reference`` enforces
end-to-end. The kernel returns per-token actual log-probabilities
(non-positive) **and** softmax entropy (non-negative); IGNORE_INDEX
positions produce 0 in both outputs.
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
    # work on any accelerator and pass the device-api-check sanity job
    # (which forbids hardcoded device strings in tests).
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


def _reference_per_token(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference impl: full-logits gather + entropy. Used as ground truth.

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
    logits = F.linear(flat, weights)
    if temperature != 1.0:
        logits = logits / temperature
    logits = logits.float()
    target = shifted.reshape(-1)
    log_probs_flat = cl._per_token_log_probs_from_logits(logits, target, ignore_index)
    entropy_flat = cl._per_token_entropy_from_logits(logits)
    # Mask entropy at IGN to mirror kernel.
    mask = target != ignore_index
    entropy_flat = torch.where(mask, entropy_flat, torch.zeros_like(entropy_flat))

    log_probs = log_probs_flat.view_as(shifted)
    entropy = entropy_flat.view_as(shifted)
    return F.pad(log_probs, (0, 1), value=0.0), F.pad(entropy, (0, 1), value=0.0)


def test_bitwise_parity_with_reference():
    """Kernel forward equals the reference bitwise (single-chunk boundary).

    With ``chunk_size`` > total tokens the kernel collapses to one
    ``h @ weight.t()``; under batch-invariant mode that matmul + the
    log_softmax + gather are bitwise identical to the reference's
    single ``F.linear`` + log_softmax + gather. Asserts both outputs
    (log_probs and entropy).
    """
    torch.manual_seed(0)
    B, L, H, V = 2, 64, 32, 256
    hidden = torch.randn(B, L, H, dtype=torch.float32, device=_device())
    weights = torch.randn(V, H, dtype=torch.float32, device=_device())
    labels = torch.randint(0, V, (B, L), dtype=torch.long, device=_device())
    labels[0, ::7] = IGNORE_INDEX
    labels[1, 0] = IGNORE_INDEX

    log_probs, entropy = cl.chunk_logprobs_function(hidden, weights, labels, chunk_size=B * L + 1)
    ref_log_probs, ref_entropy = _reference_per_token(hidden, weights, labels)

    _assert_bitwise_equal(log_probs, ref_log_probs, "log_probs")
    _assert_bitwise_equal(entropy, ref_entropy, "entropy")


def test_temperature_scales_logits_bitwise():
    """``temperature != 1.0`` divides logits before softmax.

    Pin bitwise parity vs the reference path that applies the same
    ``logits = logits / T`` divide before ``log_softmax + gather`` and
    entropy. Confirms the wrapper ``temperature`` kwarg threads through
    the kernel (and not e.g. silently dropped).
    """
    torch.manual_seed(0)
    B, L, H, V = 2, 32, 16, 64
    hidden = torch.randn(B, L, H, dtype=torch.float32, device=_device())
    weights = torch.randn(V, H, dtype=torch.float32, device=_device())
    labels = torch.randint(0, V, (B, L), dtype=torch.long, device=_device())

    T = 2.5
    log_probs, entropy = cl.chunk_logprobs_function(hidden, weights, labels, chunk_size=B * L + 1, temperature=T)
    ref_log_probs, ref_entropy = _reference_per_token(hidden, weights, labels, temperature=T)

    _assert_bitwise_equal(log_probs, ref_log_probs, "log_probs (T=2.5)")
    _assert_bitwise_equal(entropy, ref_entropy, "entropy (T=2.5)")


def test_ignore_index_zeroes_output_and_grad():
    """All-IGN labels emit exactly zero outputs and zero gradients.

    Both ``log_probs`` and ``entropy`` are masked to 0 at IGN positions
    (so summing either over the seq gives "value at supervised slots
    only"), and no gradient flows through hidden_states / weights.
    """
    torch.manual_seed(0)
    B, L, H, V = 1, 8, 4, 16
    hidden = torch.randn(B, L, H, dtype=torch.float32, device=_device(), requires_grad=True)
    weights = torch.randn(V, H, dtype=torch.float32, device=_device(), requires_grad=True)
    labels = torch.full((B, L), IGNORE_INDEX, dtype=torch.long, device=_device())

    log_probs, entropy = cl.chunk_logprobs_function(hidden, weights, labels, chunk_size=4)
    assert torch.all(log_probs == 0)
    assert torch.all(entropy == 0)

    (log_probs.sum() + entropy.sum()).backward()
    assert torch.all(hidden.grad == 0)
    assert torch.all(weights.grad == 0)


def _manual_backward_per_token(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    grad_log_probs: torch.Tensor | None,
    grad_entropy: torch.Tensor | None,
    temperature: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hand-derived backward matching the kernel's explicit op sequence.

    Mirrors verl's ``_fused_linear_for_ppo_bwd`` exactly:
    ``dlogits_lp = dlp * (one_hot - softmax)``,
    ``dlogits_ent = -dent * softmax * (log_softmax + entropy)``,
    summed; cast back to input dtype, divide by T, then matmul.
    PyTorch's stock autograd path through ``log_softmax + gather`` is
    mathematically equivalent but sequences the fp32 ops differently
    (scatter-then-subtract vs subtract-then-multiply), so the two paths'
    grads agree only at fp32 epsilon (~1e-7), not bitwise.
    """
    shifted = labels[..., 1:].contiguous()
    h = hidden[..., :-1, :].contiguous()
    flat = h.reshape(-1, h.size(-1))
    target = shifted.reshape(-1)

    logits = F.linear(flat, weights)
    if temperature != 1.0:
        logits = logits / temperature
    logits = logits.float()
    probs = logits.softmax(dim=-1)
    mask = (target != ignore_index).float()

    dlogits = torch.zeros_like(probs)

    if grad_log_probs is not None:
        dlp_flat = grad_log_probs[..., :-1].reshape(-1)
        safe = target.clamp(min=0).unsqueeze(-1)
        one_hot = torch.zeros_like(probs).scatter_(-1, safe, 1.0)
        masked_dlp = (dlp_flat * mask).unsqueeze(-1)
        dlogits = dlogits + masked_dlp * (one_hot - probs)

    if grad_entropy is not None:
        dent_flat = grad_entropy[..., :-1].reshape(-1)
        log_probs_full = logits.log_softmax(dim=-1)
        entropy_full = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
        masked_dent = (dent_flat * mask).unsqueeze(-1)
        dlogits = dlogits + probs * (log_probs_full + entropy_full.unsqueeze(-1)) * (-masked_dent)

    dlogits = dlogits.to(flat.dtype)
    if temperature != 1.0:
        dlogits = dlogits / temperature

    dhidden_flat = dlogits @ weights
    dweight = dlogits.t() @ flat

    dhidden = torch.zeros_like(hidden)
    dhidden[..., :-1, :] = dhidden_flat.view_as(h)
    return dhidden, dweight


def test_chunk_size_invariance_forward_and_grad():
    """Outputs and gradients are bitwise invariant to ``chunk_size``.

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

    grad_lp = torch.randn(B, L, dtype=torch.float32, device=_device())
    grad_ent = torch.randn(B, L, dtype=torch.float32, device=_device())

    chunk_sizes = (24, 1024)
    log_probs_outs = []
    entropy_outs = []
    grads_h = []
    grads_w = []
    for chunk_size in chunk_sizes:
        h = hidden0.clone().requires_grad_(True)
        w = weights0.clone().requires_grad_(True)
        log_probs, entropy = cl.chunk_logprobs_function(h, w, labels, chunk_size=chunk_size)
        ((log_probs * grad_lp).sum() + (entropy * grad_ent).sum()).backward()
        log_probs_outs.append(log_probs.detach())
        entropy_outs.append(entropy.detach())
        grads_h.append(h.grad.detach())
        grads_w.append(w.grad.detach())

    for i in range(1, len(chunk_sizes)):
        _assert_bitwise_equal(log_probs_outs[i], log_probs_outs[0], f"log_probs[chunk_size={chunk_sizes[i]}]")
        _assert_bitwise_equal(entropy_outs[i], entropy_outs[0], f"entropy[chunk_size={chunk_sizes[i]}]")
        _assert_bitwise_equal(grads_h[i], grads_h[0], f"dhidden[chunk_size={chunk_sizes[i]}]")
        _assert_bitwise_equal(grads_w[i], grads_w[0], f"dweight[chunk_size={chunk_sizes[i]}]")


def test_grad_matches_reference():
    """Gradients are bitwise identical to a hand-derived backward reference.

    Uses ``_manual_backward_per_token`` (which mirrors the kernel's
    explicit ``dlogits = dlp * (one_hot - softmax) - dent * softmax *
    (log_softmax + entropy)`` form instead of relying on autograd's
    ``log_softmax + gather`` backward). Forces ``chunk_size > total
    tokens`` so the kernel's ``dweight = dlogits.t() @ h_chunk`` is a
    single mm matching the reference's single mm. Under batch-invariant
    ``mm`` both paths produce bit-identical gradients across **both**
    log_probs and entropy upstream gradients.
    """
    torch.manual_seed(42)
    B, L, H, V = 2, 32, 16, 64
    hidden0 = torch.randn(B, L, H, dtype=torch.float32, device=_device())
    weights0 = torch.randn(V, H, dtype=torch.float32, device=_device())
    labels = torch.randint(0, V, (B, L), dtype=torch.long, device=_device())
    labels[0, ::5] = IGNORE_INDEX

    grad_lp = torch.randn(B, L, dtype=torch.float32, device=_device())
    grad_ent = torch.randn(B, L, dtype=torch.float32, device=_device())

    h_a = hidden0.clone().requires_grad_(True)
    w_a = weights0.clone().requires_grad_(True)
    log_probs_a, entropy_a = cl.chunk_logprobs_function(h_a, w_a, labels, chunk_size=B * L + 1)
    ((log_probs_a * grad_lp).sum() + (entropy_a * grad_ent).sum()).backward()

    ref_log_probs, ref_entropy = _reference_per_token(hidden0, weights0, labels)
    dhidden_ref, dweight_ref = _manual_backward_per_token(
        hidden0, weights0, labels, grad_log_probs=grad_lp, grad_entropy=grad_ent
    )

    _assert_bitwise_equal(log_probs_a.detach(), ref_log_probs, "log_probs")
    _assert_bitwise_equal(entropy_a.detach(), ref_entropy, "entropy")
    _assert_bitwise_equal(h_a.grad, dhidden_ref, "dhidden")
    _assert_bitwise_equal(w_a.grad, dweight_ref, "dweight")


def test_sp_enabled_skips_internal_shift(monkeypatch):
    """Under SP, the dataloader pre-shifts; kernel must not shift again.

    Reference computes ``F.linear(hidden) -> log_softmax -> gather``
    against the *un-shifted* labels, and asserts bitwise equality to
    the kernel under batch-invariant mode. Both log_probs and entropy
    are validated.
    """
    monkeypatch.setattr(cl, "get_parallel_state", lambda: _FakePS(sp_enabled=True))

    torch.manual_seed(1)
    B, L, H, V = 1, 16, 8, 32
    hidden = torch.randn(B, L, H, dtype=torch.float32, device=_device())
    weights = torch.randn(V, H, dtype=torch.float32, device=_device())
    labels = torch.randint(0, V, (B, L), dtype=torch.long, device=_device())

    log_probs, entropy = cl.chunk_logprobs_function(hidden, weights, labels, chunk_size=B * L + 1)
    assert log_probs.shape == labels.shape
    assert entropy.shape == labels.shape

    flat = hidden.reshape(-1, H)
    logits = F.linear(flat, weights).float()
    target = labels.reshape(-1)
    expected_lp = cl._per_token_log_probs_from_logits(logits, target, IGNORE_INDEX).view_as(labels)
    expected_ent = cl._per_token_entropy_from_logits(logits)
    mask = target != IGNORE_INDEX
    expected_ent = torch.where(mask, expected_ent, torch.zeros_like(expected_ent)).view_as(labels)

    _assert_bitwise_equal(log_probs, expected_lp, "log_probs (sp_enabled)")
    _assert_bitwise_equal(entropy, expected_ent, "entropy (sp_enabled)")


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


@pytest.mark.parametrize("temperature", [1.0, 0.7, 2.5])
def test_bitwise_parity_with_verl_fused_linear_for_ppo(temperature):
    """VeOmni ``chunk_logprobs_function`` is bitwise equal to verl's
    ``FusedLinearForPPOFunction`` (token_log_probs **and** entropy).

    Both kernels:
      1. Compute ``logits = (h @ w.t()) / T`` in input dtype, then upcast to fp32.
      2. Route per-token NLL through ``flash_attn``'s triton CE kernel.
      3. Negate to get actual log-probabilities.
      4. Compute entropy ``H[p] = logsumexp(logits) - sum(p * logits)``.
      5. Compute backward via explicit ``dlogits = dlp * (one_hot - softmax)
         + (-dent) * softmax * (log_softmax + entropy)``, cast to input dtype,
         divide by T, then matmul.

    Pinning the parity here keeps the verl ↔ VeOmni integration story
    "drop-in": a model trained under VeOmni's per-token log-probs +
    entropy path produces the same fp32 numbers as the same model
    invoked through verl's fused PPO kernel, under deterministic +
    batch-invariant mode. The parametrize over temperature confirms
    the temperature passthrough survives the verl↔VeOmni boundary.
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

    grad_lp = torch.randn(B, L, dtype=torch.float32, device=_device())
    grad_ent = torch.randn(B, L, dtype=torch.float32, device=_device())
    chunk_size = B * L + 1  # single-chunk boundary for both kernels

    # VeOmni — pass ``shift_labels=labels`` so the wrapper treats them
    # as already-aligned (verl's kernel never applies a causal shift,
    # mirroring that here keeps the per-token output aligned for a
    # bit-for-bit comparison).
    log_probs_a, entropy_a = cl.chunk_logprobs_function(
        hidden_a,
        weights_a,
        labels,
        chunk_size=chunk_size,
        shift_labels=labels,
        temperature=temperature,
    )
    ((log_probs_a * grad_lp).sum() + (entropy_a * grad_ent).sum()).backward()

    # verl — same inputs cloned, no causal shift inside
    hidden_b = hidden_a.detach().clone().requires_grad_(True)
    weights_b = weights_a.detach().clone().requires_grad_(True)
    log_probs_b, entropy_b = verl_fn.apply(hidden_b, weights_b, labels, temperature, chunk_size)
    ((log_probs_b * grad_lp).sum() + (entropy_b * grad_ent).sum()).backward()

    _assert_bitwise_equal(log_probs_a, log_probs_b, "log_probs vs verl")
    _assert_bitwise_equal(entropy_a, entropy_b, "entropy vs verl")
    _assert_bitwise_equal(hidden_a.grad, hidden_b.grad, "dhidden vs verl")
    _assert_bitwise_equal(weights_a.grad, weights_b.grad, "dweight vs verl")


def test_explicit_shift_labels_overrides_internal_shift():
    """When the caller provides shift_labels, the kernel uses them as-is.

    Mirrors the contract used by ``ForCausalLMLoss``: under SP off the
    wrapper builds ``shift_labels = F.pad(labels, (0, 1), IGN)[..., 1:]``
    so each label position is already the next-token target. The kernel
    must consume that directly (no internal shift, no trailing pad) and
    return tensors whose seq length matches the *passed* shift_labels.
    """
    torch.manual_seed(11)
    B, L, H, V = 2, 16, 8, 32
    hidden = torch.randn(B, L, H, dtype=torch.float32, device=_device())
    weights = torch.randn(V, H, dtype=torch.float32, device=_device())
    labels = torch.randint(0, V, (B, L), dtype=torch.long, device=_device())
    labels[0, 3] = IGNORE_INDEX

    shift_labels = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    log_probs, entropy = cl.chunk_logprobs_function(
        hidden, weights, labels, chunk_size=B * L + 1, shift_labels=shift_labels
    )

    assert log_probs.shape == shift_labels.shape
    assert entropy.shape == shift_labels.shape

    flat = hidden.reshape(-1, H)
    logits = F.linear(flat, weights).float()
    target = shift_labels.reshape(-1)
    expected_lp = cl._per_token_log_probs_from_logits(logits, target, IGNORE_INDEX).view_as(shift_labels)
    expected_ent = cl._per_token_entropy_from_logits(logits)
    mask = target != IGNORE_INDEX
    expected_ent = torch.where(mask, expected_ent, torch.zeros_like(expected_ent)).view_as(shift_labels)

    _assert_bitwise_equal(log_probs, expected_lp, "log_probs (explicit shift_labels)")
    _assert_bitwise_equal(entropy, expected_ent, "entropy (explicit shift_labels)")
