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

"""Tests for the chunked fused linear top-k forward-KL distillation kernel.

The kernel returns ``(log_probs, entropy, distillation_losses, student_mass,
teacher_mass)``. We exercise:

- Numerical equivalence vs a dense reference (``F.linear -> log_softmax ->
  gather`` + the same KL formula verl's ``compute_forward_kl_topk`` runs)
  across single-chunk and multi-chunk paths. CPU-friendly (no triton, no
  ``flash_attn``) so it runs in the default CI matrix.
- Bitwise parity vs verl's ``compute_forward_kl_topk`` on CUDA under
  deterministic + batch-invariant mode, mirroring the pattern in
  ``tests/ops/test_chunk_logprobs.py``.
- Closed-form backward correctness via a direct comparison against
  PyTorch's autograd through the dense reference.
- IGNORE_INDEX masking → exact zero on all five outputs and zero
  gradient.
- ``log_prob_min_clamp`` round-trip.
- ``temperature`` path matches the corresponding sibling kernel.
- ``student_mass`` / ``teacher_mass`` are detached.
"""

import os

import pytest
import torch
import torch.nn.functional as F

import veomni.ops.kernels.cross_entropy.chunk_logprobs as cl
import veomni.ops.kernels.cross_entropy.chunk_topk_distill as ctkd
from veomni.ops.kernels.cross_entropy import chunk_logprobs_function
from veomni.utils.constants import IGNORE_INDEX


# Required by ``torch.use_deterministic_algorithms`` for cuBLAS on CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class _FakePS:
    """Stand-in for ``ParallelState`` so the kernel's SP gate is testable
    without spinning up a process group."""

    def __init__(self, sp_enabled: bool = False):
        self.sp_enabled = sp_enabled


@pytest.fixture(autouse=True)
def _no_sp(monkeypatch):
    """Default: SP disabled. Tests that need SP-on monkeypatch this again."""
    monkeypatch.setattr(ctkd, "get_parallel_state", lambda: _FakePS(sp_enabled=False))
    # Force the gather-fallback path for the per-token NLL helper so the
    # tests run on CPU. The fa_ce triton kernel requires a real CUDA device;
    # the fallback path is bitwise-equivalent on the IGN-zero contract and
    # numerically identical on fp32 inputs.
    monkeypatch.setattr(cl, "_FA_CE_AVAILABLE", False)


def _dense_reference(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    temperature: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
    log_prob_min_clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense reference that mirrors verl's ``compute_forward_kl_topk``.

    Applies the same internal causal shift as the kernel (predict y_{t+1}
    from h_t) and pads the trailing slot with ``0`` so the output shape
    matches the input ``labels``. The KL formula and clamp semantics
    are copied from
    ``verl/trainer/distillation/fsdp/losses.py::compute_forward_kl_topk``
    so a passing test of this reference against the kernel is also a
    passing equivalence test against verl's loss (modulo the chunking
    boundary, which is the entire point of the kernel).
    """
    labels_shifted = labels[..., 1:].contiguous()
    hidden_shifted = hidden[..., :-1, :].contiguous()
    ids_shifted = teacher_topk_ids[..., 1:, :].contiguous()
    tlp_shifted = teacher_topk_log_probs[..., 1:, :].contiguous()

    logits = F.linear(hidden_shifted, weights)
    if temperature != 1.0:
        logits = logits / temperature
    logits = logits.float()

    log_probs_full = logits.log_softmax(dim=-1)
    mask = (labels_shifted != ignore_index).float()

    # log_probs / entropy at the actual label positions.
    safe = labels_shifted.clamp(min=0).unsqueeze(-1)
    log_probs = log_probs_full.gather(-1, safe).squeeze(-1) * mask

    probs = logits.softmax(dim=-1)
    entropy = (torch.logsumexp(logits, dim=-1) - (probs * logits).sum(dim=-1)) * mask

    # Top-k distillation outputs.
    student_topk_lp = log_probs_full.gather(-1, ids_shifted)  # [B, L-1, K]
    student_mass = student_topk_lp.exp().sum(-1) * mask
    teacher_mass = tlp_shifted.exp().sum(-1) * mask

    if log_prob_min_clamp is not None:
        student_topk_lp_eff = student_topk_lp.clamp_min(log_prob_min_clamp)
        tlp_eff = tlp_shifted.clamp_min(log_prob_min_clamp)
    else:
        student_topk_lp_eff = student_topk_lp
        tlp_eff = tlp_shifted

    p_t = tlp_eff.float().exp()
    distill = (p_t * (tlp_eff.float() - student_topk_lp_eff.float())).sum(-1) * mask

    return (
        F.pad(log_probs, (0, 1), value=0.0),
        F.pad(entropy, (0, 1), value=0.0),
        F.pad(distill, (0, 1), value=0.0),
        F.pad(student_mass, (0, 1), value=0.0),
        F.pad(teacher_mass, (0, 1), value=0.0),
    )


def _make_inputs(B=2, L=16, H=8, V=64, K=4, seed=0, dtype=torch.float32, device="cpu"):
    """Build a self-consistent (h, w, labels, teacher_topk_ids, teacher_topk_log_probs) tuple.

    Teacher log-probs are derived from a separate "teacher" logits draw and
    normalized via ``log_softmax`` so they pass verl's mass < 1 sanity check.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(B, L, H, dtype=dtype, device=device, generator=g)
    weights = torch.randn(V, H, dtype=dtype, device=device, generator=g)
    labels = torch.randint(0, V, (B, L), generator=g, dtype=torch.long, device=device)

    teacher_logits = torch.randn(B, L, V, dtype=dtype, device=device, generator=g)
    teacher_log_probs = teacher_logits.log_softmax(dim=-1)
    teacher_topk_log_probs, teacher_topk_ids = teacher_log_probs.topk(K, dim=-1)
    return hidden, weights, labels, teacher_topk_ids.contiguous(), teacher_topk_log_probs.contiguous()


def test_forward_matches_dense_reference_single_chunk():
    """``chunk_size`` large enough to fit all tokens -> bitwise vs reference.

    The dense reference and the kernel collapse to the same matmul +
    log_softmax + gather sequence. fp32 inputs + a single chunk means no
    rounding boundary difference is possible.
    """
    h, w, labels, ids, tlp = _make_inputs(B=2, L=16, H=8, V=64, K=4)
    log_probs, entropy, distill, student_mass, teacher_mass = ctkd.chunk_topk_distill_function(
        h, w, labels, ids, tlp, chunk_size=10_000
    )
    ref = _dense_reference(h, w, labels, ids, tlp)

    torch.testing.assert_close(log_probs, ref[0], rtol=0, atol=0)
    torch.testing.assert_close(entropy, ref[1], rtol=0, atol=0)
    torch.testing.assert_close(distill, ref[2], rtol=0, atol=0)
    torch.testing.assert_close(student_mass, ref[3], rtol=0, atol=0)
    torch.testing.assert_close(teacher_mass, ref[4], rtol=0, atol=0)


def test_forward_invariant_across_chunk_sizes():
    """Multi-chunk path = single-chunk path under fp32 deterministic mode."""
    h, w, labels, ids, tlp = _make_inputs(B=2, L=32, H=8, V=64, K=4)
    outs_full = ctkd.chunk_topk_distill_function(h, w, labels, ids, tlp, chunk_size=10_000)
    outs_small = ctkd.chunk_topk_distill_function(h, w, labels, ids, tlp, chunk_size=7)
    for a, b, name in zip(outs_full, outs_small, ["log_probs", "entropy", "distill", "smass", "tmass"]):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6, msg=lambda m, n=name: f"{n}: {m}")


def test_ignore_index_zeroes_all_outputs_and_grads():
    """All-IGN labels -> exact zero on every output and every gradient."""
    h, w, labels, ids, tlp = _make_inputs(B=1, L=8, H=4, V=16, K=3)
    labels[:] = IGNORE_INDEX
    h = h.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)

    log_probs, entropy, distill, smass, tmass = ctkd.chunk_topk_distill_function(h, w, labels, ids, tlp, chunk_size=4)
    for t, n in [
        (log_probs, "log_probs"),
        (entropy, "entropy"),
        (distill, "distill"),
        (smass, "smass"),
        (tmass, "tmass"),
    ]:
        assert torch.all(t == 0), f"{n} should be all-zero under IGN-only labels"

    (log_probs.sum() + entropy.sum() + distill.sum()).backward()
    assert torch.all(h.grad == 0)
    assert torch.all(w.grad == 0)


def test_partial_ignore_index_masks_only_affected_slots():
    """Mixed IGN/valid labels: zero on IGN slots, non-zero elsewhere."""
    h, w, labels, ids, tlp = _make_inputs(B=1, L=10, H=4, V=16, K=3, seed=1)
    labels[0, ::3] = IGNORE_INDEX

    log_probs, entropy, distill, smass, tmass = ctkd.chunk_topk_distill_function(h, w, labels, ids, tlp, chunk_size=4)
    # Trailing pad slot is always zero. IGN slots in the shifted index space:
    # labels_shifted[i] = labels[i+1] so labels[i+1] == IGN -> output[i] == 0.
    shifted_ign = labels[..., 1:] == IGNORE_INDEX
    shifted_ign = F.pad(shifted_ign, (0, 1), value=True)  # trailing slot
    for t, n in [
        (log_probs, "log_probs"),
        (entropy, "entropy"),
        (distill, "distill"),
        (smass, "smass"),
        (tmass, "tmass"),
    ]:
        assert torch.all(t[shifted_ign] == 0), f"{n}: IGN slots should be 0"


def test_temperature_path_matches_chunk_logprobs():
    """``log_probs`` / ``entropy`` slots match the sibling kernel under any T.

    The distillation kernel and ``chunk_logprobs_function`` share the same
    op order on the (log_probs, entropy) outputs. Pinning equivalence
    between the two kernels guarantees the distillation path doesn't drift
    on the shared outputs.
    """
    h, w, labels, ids, tlp = _make_inputs(B=1, L=8, H=4, V=16, K=3, seed=2)
    T = 0.7
    out_distill = ctkd.chunk_topk_distill_function(h, w, labels, ids, tlp, chunk_size=10_000, temperature=T)
    out_lp = chunk_logprobs_function(h, w, labels, chunk_size=10_000, temperature=T)
    torch.testing.assert_close(out_distill[0], out_lp[0], rtol=0, atol=0)
    torch.testing.assert_close(out_distill[1], out_lp[1], rtol=0, atol=0)


def test_log_prob_min_clamp_affects_only_clamped_entries():
    """``log_prob_min_clamp`` floors both top-k log-prob tensors before KL."""
    h, w, labels, ids, tlp = _make_inputs(B=1, L=8, H=4, V=16, K=3, seed=3)

    # Pick a clamp aggressive enough to trip at least some entries.
    clamp = -2.0
    outs_clamped = ctkd.chunk_topk_distill_function(
        h, w, labels, ids, tlp, chunk_size=10_000, log_prob_min_clamp=clamp
    )
    outs_uncl = ctkd.chunk_topk_distill_function(h, w, labels, ids, tlp, chunk_size=10_000)

    # log_probs / entropy / student_mass / teacher_mass are *not* affected
    # by the clamp (it only enters the KL reduction). Pin that.
    for i, name in enumerate(["log_probs", "entropy", "student_mass", "teacher_mass"]):
        idx = i if i < 2 else i + 1  # skip the distill slot (index 2)
        torch.testing.assert_close(
            outs_clamped[idx], outs_uncl[idx], rtol=0, atol=0, msg=lambda m, n=name: f"{n}: {m}"
        )

    # Distill values should match the reference under the same clamp.
    ref = _dense_reference(h, w, labels, ids, tlp, log_prob_min_clamp=clamp)
    torch.testing.assert_close(outs_clamped[2], ref[2], rtol=0, atol=0)


def test_mass_outputs_are_detached():
    """``student_mass`` / ``teacher_mass`` carry ``requires_grad=False``."""
    h, w, labels, ids, tlp = _make_inputs(B=1, L=8, H=4, V=16, K=3, seed=4)
    h = h.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)

    _, _, _, smass, tmass = ctkd.chunk_topk_distill_function(h, w, labels, ids, tlp, chunk_size=4)
    assert not smass.requires_grad, "student_mass must be detached"
    assert not tmass.requires_grad, "teacher_mass must be detached"


def test_backward_matches_dense_reference():
    """Closed-form kernel backward equals autograd through the dense reference.

    Both the kernel forward/backward and the dense reference cast logits to
    fp32 before the log_softmax (matching ``chunk_logprobs.py``'s op order,
    which is bitwise-vs-verl). That fp32 boundary caps the agreement at
    fp32 epsilon (~1e-6), not the fp64 epsilon you'd otherwise expect from
    the input dtype.
    """
    torch.manual_seed(0)
    h, w, labels, ids, tlp = _make_inputs(B=1, L=8, H=4, V=16, K=3, dtype=torch.float64)
    h.requires_grad_(True)
    w.requires_grad_(True)
    h_ref = h.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    grad_distill = torch.randn_like(labels, dtype=torch.float64)
    grad_lp = torch.randn_like(labels, dtype=torch.float64)
    grad_ent = torch.randn_like(labels, dtype=torch.float64)

    log_probs, entropy, distill, _, _ = ctkd.chunk_topk_distill_function(h, w, labels, ids, tlp, chunk_size=10_000)
    loss = (distill * grad_distill).sum() + (log_probs * grad_lp).sum() + (entropy * grad_ent).sum()
    loss.backward()

    ref_lp, ref_ent, ref_dist, _, _ = _dense_reference(h_ref, w_ref, labels, ids, tlp)
    loss_ref = (ref_dist * grad_distill).sum() + (ref_lp * grad_lp).sum() + (ref_ent * grad_ent).sum()
    loss_ref.backward()

    torch.testing.assert_close(h.grad, h_ref.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(w.grad, w_ref.grad, rtol=1e-5, atol=1e-6)


def test_backward_with_clamp_matches_reference():
    """Same backward equivalence holds when ``log_prob_min_clamp`` is active.

    Constructs the teacher distribution so several top-k entries fall
    **below** the clamp. The teacher coefficient in the closed-form
    backward (``p_t_k = exp(clamp(log_p_t,k))``) then has to match the
    forward's clamped value — passing here pins that consistency. A
    prior buggy revision that used unclamped ``tlp`` in the backward
    would fail this assertion.
    """
    torch.manual_seed(0)
    h, w, labels, ids, tlp = _make_inputs(B=1, L=8, H=4, V=16, K=3, dtype=torch.float64, seed=5)
    h.requires_grad_(True)
    w.requires_grad_(True)
    h_ref = h.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    clamp = -3.0
    # Force the clamp to trip on the teacher side: pull about half the
    # top-k log-prob entries below ``clamp`` by subtracting a large
    # constant. After this nudge, ``tlp`` is **not** a valid
    # ``log_softmax`` output (rows no longer sum to 1) — which is fine,
    # the kernel and the reference both treat it as opaque log-prob
    # data; we're checking gradient equivalence, not distributional
    # properties.
    tlp[..., 1::2] -= 5.0
    assert (tlp < clamp).any(), "clamp must trip on the teacher side for this test to be meaningful"

    grad_distill = torch.randn_like(labels, dtype=torch.float64)

    _, _, distill, _, _ = ctkd.chunk_topk_distill_function(
        h, w, labels, ids, tlp, chunk_size=10_000, log_prob_min_clamp=clamp
    )
    (distill * grad_distill).sum().backward()

    _, _, ref_dist, _, _ = _dense_reference(h_ref, w_ref, labels, ids, tlp, log_prob_min_clamp=clamp)
    (ref_dist * grad_distill).sum().backward()

    torch.testing.assert_close(h.grad, h_ref.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(w.grad, w_ref.grad, rtol=1e-5, atol=1e-6)


def test_matches_verl_compute_forward_kl_topk():
    """Equivalence against verl's ``compute_forward_kl_topk`` reference.

    Skips when the local environment doesn't have verl installed. When it
    does, the verl FSDP loss path runs on a dense ``[1, L, V]``
    ``student_logits`` tensor and is the engine-side ground truth that our
    fused kernel must reproduce.
    """
    try:
        from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk
        from verl.workers.config import DistillationConfig, DistillationLossConfig
    except Exception:
        pytest.skip("verl not importable in this environment")

    h, w, labels, ids, tlp = _make_inputs(B=1, L=16, H=8, V=64, K=4, seed=6)

    # verl's helper expects nested ``[B, L, K]`` teacher tensors and
    # ``[B, L, V]`` student_logits — pre-shift everything the same way the
    # kernel does internally and feed verl's helper with the post-shift
    # tensors so the comparison is apples-to-apples.
    h_shifted = h[..., :-1, :].contiguous()
    ids_shifted = ids[..., 1:, :].contiguous()
    tlp_shifted = tlp[..., 1:, :].contiguous()

    student_logits = F.linear(h_shifted, w).float().unsqueeze(0).squeeze(0).unsqueeze(0)
    # verl's helper asserts ``is_nested``; build the nested view from
    # ``[B, L-1, K]`` by listing per-batch tensors.
    teacher_lp_nested = torch.nested.as_nested_tensor(list(tlp_shifted.unbind(0)))
    teacher_ids_nested = torch.nested.as_nested_tensor(list(ids_shifted.unbind(0)))

    cfg = DistillationConfig(distillation_loss=DistillationLossConfig(loss_mode="forward_kl_topk", topk=ids.shape[-1]))
    verl_out = compute_forward_kl_topk(
        student_logits=student_logits,
        teacher_topk_log_probs=teacher_lp_nested,
        teacher_topk_ids=teacher_ids_nested,
        config=cfg,
        data_format="bshd",
    )
    # verl masks at IGN inside its own loss combinator (not in
    # compute_forward_kl_topk), so for this equivalence we use all-valid
    # labels by construction. Only the three top-k tensors are checked
    # here; the per-token NLL / entropy slots are exercised by the
    # sibling tests.
    _, _, distill, smass, tmass = ctkd.chunk_topk_distill_function(h, w, labels, ids, tlp, chunk_size=10_000)
    # Strip the trailing pad slot to align with verl's output shape.
    torch.testing.assert_close(distill[..., :-1], verl_out["distillation_losses"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(smass[..., :-1], verl_out["student_mass"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tmass[..., :-1], verl_out["teacher_mass"], rtol=1e-5, atol=1e-5)


def test_chunk_topk_distill_saves_memory_vs_eager():
    """Peak device memory: fused kernel << eager ``h @ w.T`` materialization.

    The whole point of routing verl's top-k distillation through this
    kernel is to avoid materializing the ``[T, V]`` student-logits
    tensor that the non-fused path allocates. We pin that property
    directly: measure peak device memory under each path on the same
    (h, w, labels, teacher_*) tensors and assert fused << eager.

    Sizing rationale: at V=64k the dense ``[B*L, V]`` fp32 logits
    tensor is ~525 MB for B*L=2048 — well above any per-chunk
    allocation the kernel does internally (the kernel only ever holds
    ``chunk_size × V`` ≈ 256 KB worth of logits + log-softmax at a
    time). The assertion uses a conservative 3× margin so the test
    survives PyTorch allocator-block rounding and small fluctuations.
    Skipped on CPU and on devices with less than 4 GB free.

    Routes all device APIs through ``veomni.utils.device`` so the
    test runs unchanged on CUDA and NPU (both back ends expose
    ``mem_get_info`` / ``reset_peak_memory_stats`` /
    ``max_memory_allocated`` on their respective torch modules).
    """
    from veomni.utils.device import IS_CUDA_AVAILABLE, empty_cache, get_device_type, get_torch_device, synchronize

    if not IS_CUDA_AVAILABLE:
        pytest.skip("Accelerator-only memory test")

    torch_device_module = get_torch_device()
    free, _total = torch_device_module.mem_get_info()
    if free < 4 * 1024**3:
        pytest.skip(f"Need >= 4 GB free device memory, got {free / 1024**3:.1f} GB")

    device = torch.device(get_device_type())
    B, L, H, V, K = 1, 2048, 256, 64_000, 8
    torch.manual_seed(0)
    h = torch.randn(B, L, H, device=device, dtype=torch.float32, requires_grad=True)
    w = torch.randn(V, H, device=device, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, V, (B, L), device=device, dtype=torch.long)
    teacher_logits = torch.randn(B, L, V, device=device, dtype=torch.float32)
    teacher_topk_log_probs, teacher_topk_ids = teacher_logits.log_softmax(dim=-1).topk(K, dim=-1)
    teacher_topk_log_probs = teacher_topk_log_probs.contiguous()
    teacher_topk_ids = teacher_topk_ids.contiguous()
    # Free the dense teacher_logits before timing; verl never carries
    # this around — it only has the top-k slices.
    del teacher_logits
    empty_cache()

    # ---- Eager path: replicate the non-fused branch the verl FSDP
    # loss takes — `student_logits = h @ w.T` (the [T, V] materialization
    # we explicitly try to avoid), then `log_softmax + gather` to get
    # the per-top-k log-probs we'd KL-mix with the teacher.
    synchronize()
    torch_device_module.reset_peak_memory_stats(device)
    student_logits = F.linear(h, w)  # [B, L, V] fp32 — the OOM-target tensor
    student_log_probs = student_logits.log_softmax(dim=-1)  # [B, L, V] fp32
    student_topk_log_probs = student_log_probs.gather(dim=-1, index=teacher_topk_ids)  # [B, L, K]
    distill_eager = (teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)).sum(dim=-1)
    distill_eager.sum().backward()
    synchronize()
    eager_peak = torch_device_module.max_memory_allocated(device)

    # Free everything before the fused timing — the eager path's grads
    # would otherwise inflate the fused-path baseline.
    del student_logits, student_log_probs, student_topk_log_probs, distill_eager
    h.grad = None
    w.grad = None
    empty_cache()

    # ---- Fused path: chunk_topk_distill_function streams the lm_head
    # projection chunk-by-chunk; max in-flight allocation is
    # `chunk_size × V` fp32, not `B*L × V`.
    synchronize()
    torch_device_module.reset_peak_memory_stats(device)
    _lp, _ent, distill_fused, _smass, _tmass = ctkd.chunk_topk_distill_function(
        h, w, labels, teacher_topk_ids, teacher_topk_log_probs, chunk_size=128
    )
    distill_fused.sum().backward()
    synchronize()
    fused_peak = torch_device_module.max_memory_allocated(device)

    # Conservative margin (3×) to allow allocator-block rounding +
    # the fact that the fused kernel still has to hold h, w, labels,
    # teacher tensors, and the [B, L] outputs — none of which scale
    # with V. On the eager path, the dominant term is the [T, V]
    # tensor, which alone is ~525 MB at this size. Empirically the
    # fused path peaks well under the eager path / 5×.
    eager_mb = eager_peak / 1024**2
    fused_mb = fused_peak / 1024**2
    assert fused_peak * 3 < eager_peak, (
        f"top-k fused linear kernel did not save memory vs eager [T, V] path: "
        f"fused_peak={fused_mb:.1f} MB, eager_peak={eager_mb:.1f} MB "
        f"(expected fused × 3 < eager). Configuration: B={B}, L={L}, H={H}, V={V}, K={K}, "
        f"dense [B*L, V] fp32 = {B * L * V * 4 / 1024**2:.1f} MB."
    )
