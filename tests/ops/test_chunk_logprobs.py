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

These exercise correctness against a reference ``F.linear ->
log_softmax -> gather`` implementation. The kernel returns per-token
actual log-probabilities (non-positive); IGNORE_INDEX positions
produce 0. CPU-only — ``get_parallel_state`` is monkeypatched.
"""

import torch
import torch.nn.functional as F

import veomni.ops.kernels.cross_entropy.chunk_logprobs as cl
from veomni.utils.constants import IGNORE_INDEX


class _FakePS:
    def __init__(self, sp_enabled: bool):
        self.sp_enabled = sp_enabled


def _reference_per_token_log_probs(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Reference impl: full-logits gather. Used as ground truth.

    Returns per-token actual log-probabilities (non-positive). Performs
    the same causal shift the kernel does (predict y_{t+1} from h_t)
    and pads the trailing seq position with 0.0 so the result has the
    same shape as ``labels``.
    """
    shifted = labels[..., 1:].contiguous()
    h = hidden_states[..., :-1, :].contiguous()
    flat = h.reshape(-1, h.size(-1))
    logits = F.linear(flat, weights).float()
    log_softmax = logits.log_softmax(dim=-1)
    safe = shifted.reshape(-1).clamp(min=0).unsqueeze(-1)
    log_probs_flat = log_softmax.gather(-1, safe).squeeze(-1)
    mask = shifted.reshape(-1) != ignore_index
    log_probs_flat = torch.where(mask, log_probs_flat, torch.zeros_like(log_probs_flat))
    log_probs = log_probs_flat.view_as(shifted)
    return F.pad(log_probs, (0, 1), value=0.0)


def test_numerical_parity_with_reference(monkeypatch):
    monkeypatch.setattr(cl, "get_parallel_state", lambda: _FakePS(sp_enabled=False))

    torch.manual_seed(0)
    B, L, H, V = 2, 64, 32, 256
    hidden = torch.randn(B, L, H, dtype=torch.float32)
    weights = torch.randn(V, H, dtype=torch.float32)
    labels = torch.randint(0, V, (B, L), dtype=torch.long)
    # Salt with IGNORE_INDEX positions
    labels[0, ::7] = IGNORE_INDEX
    labels[1, 0] = IGNORE_INDEX

    out = cl.chunk_logprobs_function(hidden, weights, labels, chunk_size=16)
    ref = _reference_per_token_log_probs(hidden, weights, labels)

    assert out.shape == labels.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_bitwise_parity_with_reference_when_chunk_covers_seq(monkeypatch):
    """With ``chunk_size >= total tokens`` the kernel is bitwise identical to the reference.

    The chunk loop collapses to a single ``h @ weight.t()`` against the
    same weight tensor as the reference's ``F.linear``; ``log_softmax``
    and ``gather`` are row-wise pointwise ops. Both paths execute the
    same ops on the same data on CPU, so the per-token output must be
    bitwise equal — same contract that
    ``tests/models/test_return_log_probs_e2e.py::
    test_return_log_probs_bitwise_matches_logits_reference`` pins
    end-to-end on GPU under deterministic + batch-invariant mode.
    """
    monkeypatch.setattr(cl, "get_parallel_state", lambda: _FakePS(sp_enabled=False))

    torch.manual_seed(0)
    B, L, H, V = 2, 64, 32, 256
    hidden = torch.randn(B, L, H, dtype=torch.float32)
    weights = torch.randn(V, H, dtype=torch.float32)
    labels = torch.randint(0, V, (B, L), dtype=torch.long)
    labels[0, ::7] = IGNORE_INDEX
    labels[1, 0] = IGNORE_INDEX

    # `chunk_size > B * L` forces a single matmul boundary identical to
    # the reference's single ``F.linear`` call.
    out = cl.chunk_logprobs_function(hidden, weights, labels, chunk_size=B * L + 1)
    ref = _reference_per_token_log_probs(hidden, weights, labels)

    assert out.shape == labels.shape
    assert out.dtype == ref.dtype
    if not torch.equal(out, ref):
        ne = out != ref
        diff = (out - ref).abs()
        first_idx = torch.nonzero(ne, as_tuple=False)[:5].tolist()
        raise AssertionError(
            f"per-token log_probs not bitwise equal: "
            f"{int(ne.sum().item())}/{out.numel()} mismatched, "
            f"max_abs_diff={diff.max().item():.3e}, first_idx={first_idx}"
        )


def test_ignore_index_zeroes_output_and_grad(monkeypatch):
    monkeypatch.setattr(cl, "get_parallel_state", lambda: _FakePS(sp_enabled=False))

    torch.manual_seed(0)
    B, L, H, V = 1, 8, 4, 16
    hidden = torch.randn(B, L, H, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(V, H, dtype=torch.float64, requires_grad=True)
    labels = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)

    out = cl.chunk_logprobs_function(hidden, weights, labels, chunk_size=4)
    # All positions ignored → all-zero output, no gradient flow.
    assert torch.all(out == 0)

    # Confirm gradients are zero (no NaNs from masked log_softmax/gather).
    out.sum().backward()
    assert torch.all(hidden.grad == 0)
    assert torch.all(weights.grad == 0)


def test_chunk_size_invariance_forward_and_grad(monkeypatch):
    monkeypatch.setattr(cl, "get_parallel_state", lambda: _FakePS(sp_enabled=False))

    torch.manual_seed(7)
    B, L, H, V = 1, 24, 8, 32
    hidden0 = torch.randn(B, L, H, dtype=torch.float64)
    weights0 = torch.randn(V, H, dtype=torch.float64)
    labels = torch.randint(0, V, (B, L), dtype=torch.long)
    labels[0, 3] = IGNORE_INDEX
    labels[0, 11] = IGNORE_INDEX
    labels[0, 23] = IGNORE_INDEX

    grad_target = torch.randn(B, L, dtype=torch.float64)

    outputs = []
    grads_h = []
    grads_w = []
    for chunk_size in (1, 5, 24, 1000):
        h = hidden0.clone().requires_grad_(True)
        w = weights0.clone().requires_grad_(True)
        out = cl.chunk_logprobs_function(h, w, labels, chunk_size=chunk_size)
        (out * grad_target).sum().backward()
        outputs.append(out.detach())
        grads_h.append(h.grad.detach())
        grads_w.append(w.grad.detach())

    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i], atol=1e-9, rtol=1e-7)
        assert torch.allclose(grads_h[0], grads_h[i], atol=1e-9, rtol=1e-7)
        assert torch.allclose(grads_w[0], grads_w[i], atol=1e-9, rtol=1e-7)


def test_grad_matches_reference(monkeypatch):
    """End-to-end gradient parity vs. the reference (gather/log_softmax) impl."""
    monkeypatch.setattr(cl, "get_parallel_state", lambda: _FakePS(sp_enabled=False))

    torch.manual_seed(42)
    B, L, H, V = 2, 32, 16, 64
    hidden0 = torch.randn(B, L, H, dtype=torch.float64)
    weights0 = torch.randn(V, H, dtype=torch.float64)
    labels = torch.randint(0, V, (B, L), dtype=torch.long)
    labels[0, ::5] = IGNORE_INDEX

    grad_target = torch.randn(B, L, dtype=torch.float64)

    h_a = hidden0.clone().requires_grad_(True)
    w_a = weights0.clone().requires_grad_(True)
    out_a = cl.chunk_logprobs_function(h_a, w_a, labels, chunk_size=8)
    (out_a * grad_target).sum().backward()

    h_b = hidden0.clone().requires_grad_(True)
    w_b = weights0.clone().requires_grad_(True)
    out_b = _reference_per_token_log_probs(h_b, w_b, labels)
    (out_b * grad_target).sum().backward()

    assert torch.allclose(out_a.detach(), out_b.detach(), atol=1e-10, rtol=1e-7)
    # Backward goes through ``.float()`` cast inside the kernel (matches
    # verl's pattern of computing dlogits in fp32). For fp64 inputs
    # this caps gradient precision at ~fp32. Tolerance reflects that.
    assert torch.allclose(h_a.grad, h_b.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(w_a.grad, w_b.grad, atol=1e-5, rtol=1e-4)


def test_sp_enabled_skips_internal_shift(monkeypatch):
    """Under SP, the dataloader pre-shifts; kernel must not shift again."""
    monkeypatch.setattr(cl, "get_parallel_state", lambda: _FakePS(sp_enabled=True))

    torch.manual_seed(1)
    B, L, H, V = 1, 16, 8, 32
    hidden = torch.randn(B, L, H, dtype=torch.float32)
    weights = torch.randn(V, H, dtype=torch.float32)
    labels = torch.randint(0, V, (B, L), dtype=torch.long)

    out = cl.chunk_logprobs_function(hidden, weights, labels, chunk_size=4)
    # Same shape as input labels; computed against the un-shifted hidden.
    assert out.shape == labels.shape

    flat = hidden.reshape(-1, H)
    logits = F.linear(flat, weights).float()
    log_softmax = logits.log_softmax(dim=-1)
    safe = labels.reshape(-1).clamp(min=0).unsqueeze(-1)
    log_probs = log_softmax.gather(-1, safe).squeeze(-1)
    mask = labels.reshape(-1) != IGNORE_INDEX
    expected = torch.where(mask, log_probs, torch.zeros_like(log_probs)).view_as(labels)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4)


def test_explicit_shift_labels_overrides_internal_shift(monkeypatch):
    """When the caller provides shift_labels, the kernel uses them as-is.

    Mirrors the contract used by ``ForCausalLMLoss``: under SP off the
    wrapper builds ``shift_labels = F.pad(labels, (0, 1), IGN)[..., 1:]``
    so each label position is already the next-token target. The kernel
    must consume that directly (no internal shift, no trailing pad) and
    return a tensor whose seq length matches the *passed* shift_labels,
    not the original labels.
    """
    monkeypatch.setattr(cl, "get_parallel_state", lambda: _FakePS(sp_enabled=False))

    torch.manual_seed(11)
    B, L, H, V = 2, 16, 8, 32
    hidden = torch.randn(B, L, H, dtype=torch.float32)
    weights = torch.randn(V, H, dtype=torch.float32)
    labels = torch.randint(0, V, (B, L), dtype=torch.long)
    labels[0, 3] = IGNORE_INDEX

    # ForCausalLMLoss-style explicit shift: pad-right with IGN, drop first.
    shift_labels = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    out = cl.chunk_logprobs_function(hidden, weights, labels, chunk_size=4, shift_labels=shift_labels)

    # Output aligns with the *shifted* targets (seq length L because
    # shift_labels was constructed to length L), no internal shift.
    assert out.shape == shift_labels.shape

    flat = hidden.reshape(-1, H)
    logits = F.linear(flat, weights).float()
    log_softmax = logits.log_softmax(dim=-1)
    safe = shift_labels.reshape(-1).clamp(min=0).unsqueeze(-1)
    log_probs = log_softmax.gather(-1, safe).squeeze(-1)
    mask = shift_labels.reshape(-1) != IGNORE_INDEX
    expected = torch.where(mask, log_probs, torch.zeros_like(log_probs)).view_as(shift_labels)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4)
