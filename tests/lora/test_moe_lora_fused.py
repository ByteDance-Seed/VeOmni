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
"""Triton fused MoE-LoRA forward/backward parity tests (non-EP + EP single-rank).

Exercises the fused experts path in both
:class:`veomni.utils.moe_lora.LoraSharedExperts` (Mode 2 — shared LoRA) and
:class:`veomni.utils.moe_lora.LoraIndependentExperts` (Mode 1 — independent
per-expert LoRA). Both require the fused experts layout (single
``[E, 2I, H]`` ``gate_up_proj`` + single ``[E, H, I]`` ``down_proj``); see
:func:`veomni.utils.moe_lora._validate_fused_layout`.

The ``gate_up_proj`` parameter is covered by **two independent rank-r LoRA
pairs** (seed-style two-LoRA, see ``veomni.utils.moe_lora`` file
docstring). All four autograd classes carry the per-half ``(A_gate,
B_gate, A_up, B_up)`` tensors end-to-end; the EP-class tests below build
matching leaf tensors so backward parity covers each adapter
independently.

Uses the same :func:`build_toy` + :func:`load_lora_config` machinery as
``test_moe_lora_eager.py``. Each test flips ``moe_implementation`` between
``fused_triton`` (kernel path) and ``eager`` (reference) and compares:

1. forward outputs at small bf16 tolerance, and
2. d/dlora_A_*, d/dlora_B_* at small bf16 tolerance.

The EP-class single-rank parity test reproduces the non-EP class's
``preprocess`` + ``scatter`` machinery in-process so the EP autograd
classes can be exercised without spinning up a process group; the full
multi-process EP2 alignment is covered by the Phase 6 distributed tests.

Run:
    pytest -v tests/lora/test_moe_lora_fused.py
"""

from __future__ import annotations

import warnings

import pytest
import torch

from veomni.utils.moe_lora import (
    LoraIndependentExperts,
    LoraSharedExperts,
    apply_independent_moe_lora,
    apply_shared_moe_lora,
)

from .utils import (
    build_toy,
    experts_module_globs,
    find_first_matching_module,
    fused_triton_moe_ops,
    load_lora_config,
)


_TOY = "qwen3_moe_toy"  # Phase 4 / Phase 6 scope: qwen3_moe (fused experts layout).

# Forward and backward parity are checked via L2 relative error
# (``||fused - eager|| / ||eager||``) instead of element-wise atol/rtol.
# Rationale: in bf16 the per-expert group-gemm and the LoRA matmul chain
# have different reduction orders than the eager per-expert loop. Catastrophic
# cancellation can flip a single output near zero by O(1) units while the
# overall tensor stays accurate to <1% in L2. Element-wise allclose flags this
# as a divergence; L2-relative is the standard gradient-parity metric used in
# kernel correctness tests (matches what
# ``tests/ops/test_fused_moe_split_vs_merged.py`` is implicitly tolerating
# with its ``rtol=3e-2, atol=3e-2`` bounds, but expressed directly).
_FWD_L2REL_TOL = 0.02  # 2% — forward is one chain of group-gemm + add + matmul.
_GRAD_L2REL_TOL = 0.02  # 2% — backward stacks a bf16 matmul on top of the dgrad.


# Mode dispatch table — keeps the parametrised tests free of branch ladders.
# Each entry maps the parametrise id (``"shared"`` / ``"independent"``) to:
#   * the wrapper class (for isinstance checks),
#   * the ``apply_*`` factory used by ``_wrap_with_lora``,
#   * the global pointer name on ``veomni.ops.kernels.moe`` that the wrapper
#     dispatches against (used by the sanity / dispatch tests).
_MODES = {
    "shared": (LoraSharedExperts, apply_shared_moe_lora, "_fused_lora_moe_forward"),
    "independent": (LoraIndependentExperts, apply_independent_moe_lora, "_fused_independent_lora_moe_forward"),
}


def _l2_rel(actual: torch.Tensor, ref: torch.Tensor) -> float:
    """``||actual - ref||_F / ||ref||_F`` in fp32. Returns 0.0 when ``ref`` is exactly zero."""
    a = actual.float()
    r = ref.float()
    ref_norm = r.norm().item()
    if ref_norm == 0.0:
        return (a - r).norm().item()
    return ((a - r).norm() / ref_norm).item()


def _require_cuda_with_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("fused MoE-LoRA kernel requires CUDA.")
    try:
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("fused MoE-LoRA kernel requires triton.")


@pytest.fixture(autouse=True)
def _restore_moe_pointers():
    """Save / restore the global MoE pointers across each test.

    Tests in this file flip ``moe_implementation`` between fused and eager via
    ``build_toy(..., ops=fused_triton_moe_ops())`` which calls
    ``apply_veomni_fused_moe_patch("triton")`` and mutates the module-level
    ``_fused_moe_forward`` / ``_fused_lora_moe_forward`` /
    ``_fused_independent_lora_moe_forward`` pointers. Restoring afterwards
    keeps unrelated MoE tests deterministic regardless of run order.
    """
    from veomni.ops.kernels import moe as _moe_ops

    saved_base = _moe_ops._fused_moe_forward
    saved_lora = _moe_ops._fused_lora_moe_forward
    saved_indep = _moe_ops._fused_independent_lora_moe_forward
    try:
        yield
    finally:
        _moe_ops._fused_moe_forward = saved_base
        _moe_ops._fused_lora_moe_forward = saved_lora
        _moe_ops._fused_independent_lora_moe_forward = saved_indep


def _wrap_with_lora(model, lora_cfg, *, apply_fn, lora_b_perturb_std: float = 0.0):
    """Wrap ``model`` with the chosen MoE-LoRA mode; optionally bump lora_B off zero."""
    apply_fn(
        model,
        target_parameter_patterns=lora_cfg["target_parameters"],
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        freeze_base_model=True,
    )
    if lora_b_perturb_std > 0:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B_" in n:
                    p.add_(torch.randn_like(p) * lora_b_perturb_std)


def _make_inputs(experts_module, batch: int = 64, top_k: int = 2):
    """Synthetic experts-call inputs on the model's device/dtype.

    ``batch`` defaults to 64 (not the wrapper's default of 8) so each expert
    sees ~16 tokens on average — large enough for bf16 reduction noise to
    average out across the per-expert group-gemm and across the LoRA matmul,
    making fused-vs-eager parity meaningful at modest tolerances. With
    batch=8 the per-expert group has 1–2 rows and the bf16 noise dominates.
    """
    H, E = experts_module.hidden_dim, experts_module.num_experts
    p0 = next(experts_module.parameters())
    dtype, dev = p0.dtype, p0.device
    h = torch.randn(batch, H, dtype=dtype, device=dev)
    top_k_index = torch.randint(0, E, (batch, top_k), device=dev)
    top_k_weights = torch.softmax(torch.randn(batch, top_k, dtype=torch.float32, device=dev), dim=-1).to(dtype)
    return h, top_k_index, top_k_weights


def _build_wrapped(*, mode: str, fused: bool, lora_b_perturb_std: float = 0.02):
    """Build a fresh wrapped Qwen3-MoE toy + apply the chosen LoRA mode and ops backend.

    Same RNG seed for both invocations → identical base + LoRA tensors. The
    only difference is which ``moe_implementation`` was patched at build time,
    which determines whether the fused LoRA pointer is bound.
    """
    _, apply_fn, _ = _MODES[mode]
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_toy(_TOY, ops=fused_triton_moe_ops() if fused else None)
    lora_cfg = load_lora_config(_TOY)
    _wrap_with_lora(model, lora_cfg, apply_fn=apply_fn, lora_b_perturb_std=lora_b_perturb_std)
    sample_fqn, exp = find_first_matching_module(model, experts_module_globs(lora_cfg["target_parameters"]))
    return model, sample_fqn, exp, lora_cfg


def test_fused_pointer_bound_after_fused_triton_build():
    """Sanity check: building with ``moe_implementation=fused_triton`` binds both LoRA pointers."""
    _require_cuda_with_triton()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_toy(_TOY, ops=fused_triton_moe_ops())
    from veomni.ops.kernels import moe as _moe_ops

    assert _moe_ops._fused_lora_moe_forward is not None, (
        "apply_veomni_fused_moe_patch('triton') must bind _fused_lora_moe_forward (Mode 2)."
    )
    assert _moe_ops._fused_independent_lora_moe_forward is not None, (
        "apply_veomni_fused_moe_patch('triton') must bind _fused_independent_lora_moe_forward (Mode 1)."
    )
    assert _moe_ops._fused_moe_forward is not None, "Sanity: base fused MoE pointer should also be bound."


@pytest.mark.parametrize("mode", list(_MODES.keys()))
def test_wrapper_dispatch_chooses_fused_when_pointer_bound(mode):
    """Wrapper.forward must take the fused branch when its pointer is bound, eager otherwise."""
    _require_cuda_with_triton()
    wrapper_cls, _apply, pointer_name = _MODES[mode]

    # Path A: fused build → wrapper.forward calls the fused branch.
    model_f, fqn_f, exp_f, _ = _build_wrapped(mode=mode, fused=True, lora_b_perturb_std=0.0)
    wrapper_f = model_f.get_submodule(fqn_f)
    assert isinstance(wrapper_f, wrapper_cls)
    h, idx, w = _make_inputs(exp_f)
    out_f = wrapper_f(h, idx, w)
    assert out_f.shape == h.shape

    # Path B: eager build → pointer is None → wrapper falls back to eager.
    from veomni.ops.kernels import moe as _moe_ops

    _moe_ops._fused_moe_forward = None
    setattr(_moe_ops, pointer_name, None)
    model_e, fqn_e, exp_e, _ = _build_wrapped(mode=mode, fused=False, lora_b_perturb_std=0.0)
    wrapper_e = model_e.get_submodule(fqn_e)
    out_e = wrapper_e(h.cpu().to(next(exp_e.parameters()).device), idx, w)
    assert out_e.shape == h.shape


@pytest.mark.parametrize("mode", list(_MODES.keys()))
def test_fused_vs_eager_forward_parity(mode):
    """Forward output of the triton fused path matches the eager wrapper at bf16 tol."""
    _require_cuda_with_triton()

    # Eager reference first (so the autouse fixture's saved pointers match the eager state).
    model_e, fqn_e, exp_e, _ = _build_wrapped(mode=mode, fused=False, lora_b_perturb_std=0.02)
    h, idx, w = _make_inputs(exp_e)
    wrapper_e = model_e.get_submodule(fqn_e)
    with torch.no_grad():
        out_eager = wrapper_e(h, idx, w).clone()

    # Fused path — rebuild so apply_veomni_fused_moe_patch("triton") binds the kernel.
    model_f, fqn_f, _exp_f, _ = _build_wrapped(mode=mode, fused=True, lora_b_perturb_std=0.02)
    wrapper_f = model_f.get_submodule(fqn_f)
    # Sanity: identical seed / wrap → identical LoRA tensors → makes the parity check meaningful.
    # Iterate the wrapper's LoRA targets (gate_up_proj, down_proj — fused experts layout).
    for pname in wrapper_e._lora_specs:
        assert torch.equal(wrapper_e.get_lora_A_weight(pname), wrapper_f.get_lora_A_weight(pname))
        assert torch.equal(wrapper_e.get_lora_B_weight(pname), wrapper_f.get_lora_B_weight(pname))
    with torch.no_grad():
        out_fused = wrapper_f(h, idx, w)

    l2 = _l2_rel(out_fused, out_eager)
    assert l2 <= _FWD_L2REL_TOL, (
        f"[{mode}] forward parity broken: L2 relative error {l2:.4%} > {_FWD_L2REL_TOL:.2%} "
        f"(eager_norm={out_eager.float().norm().item():.3e})"
    )


@pytest.mark.parametrize("mode", list(_MODES.keys()))
def test_fused_vs_eager_backward_parity(mode):
    """Gradients on lora_A_* / lora_B_* match between fused and eager at bf16 tol."""
    _require_cuda_with_triton()

    def _grads(*, fused: bool):
        model, fqn, exp, _ = _build_wrapped(mode=mode, fused=fused, lora_b_perturb_std=0.02)
        wrapper = model.get_submodule(fqn)
        wrapper.train()
        h, idx, w = _make_inputs(exp)
        # Fixed loss (sum-of-squares) so both paths see the same upstream grad pattern.
        loss = wrapper(h, idx, w).float().pow(2).sum()
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in wrapper.named_parameters() if p.grad is not None}

    grads_eager = _grads(fused=False)
    grads_fused = _grads(fused=True)

    assert set(grads_eager) == set(grads_fused), (
        f"[{mode}] different param sets received grad: only-eager={set(grads_eager) - set(grads_fused)}, "
        f"only-fused={set(grads_fused) - set(grads_eager)}"
    )
    # Spot-check the LoRA grads — these are the only ones that should be non-zero
    # (base is frozen, perturbed lora_B → kaiming A still gets gradient via B).
    lora_param_names = sorted(n for n in grads_eager if n.startswith("lora_A_") or n.startswith("lora_B_"))
    assert lora_param_names, f"[{mode}] expected lora_A_*/lora_B_* params to receive gradients"
    for n in lora_param_names:
        ge, gf = grads_eager[n], grads_fused[n]
        assert ge.shape == gf.shape, f"[{mode}] {n}: shape mismatch eager={ge.shape} fused={gf.shape}"
        l2 = _l2_rel(gf, ge)
        assert l2 <= _GRAD_L2REL_TOL, (
            f"[{mode}] {n}: grad parity broken — L2 relative error {l2:.4%} > {_GRAD_L2REL_TOL:.2%} "
            f"(eager_norm={ge.float().norm().item():.3e}, max|fused-eager|={(ge - gf).abs().max().item():.3e})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# EP autograd-class single-rank parity (Phase 5).
#
# These exercise the LoRA math inside ``EPMergedFc1{Shared,Independent}LoRAGroupGemm``
# without spinning up a process group. We hand-build the same scatter / cumsum
# the non-EP class would compute internally, run the EP class on that permuted
# block, then replay the routing-weight + gather to compare against the non-EP
# reference. The actual all-to-all + permute-reorder plumbing in
# ``_ep_dispatch_fused_lora_moe`` is exercised by the existing non-LoRA EP tests
# (``EPMergedFc1GroupGemm`` reuses ``preprocess`` / ``token_pre_all2all`` /
# ``tokens_post_all2all`` 1:1) and by the multi-process Phase 6 EP2 tests; this
# unit-level check just locks in the LoRA math added on top.
# ──────────────────────────────────────────────────────────────────────────────


def _make_lora_leaf(*shape: int, dtype: torch.dtype, device: torch.device, scale: float = 0.02) -> torch.Tensor:
    """Make a leaf parameter-shaped tensor with ``requires_grad=True`` after scaling.

    ``torch.randn(...) * scale`` would create a non-leaf because the scaling op
    captures autograd; ``.detach()`` then ``requires_grad_()`` keeps it a leaf
    so ``.grad`` populates after ``backward()``.
    """
    return (torch.randn(*shape, dtype=dtype, device=device) * scale).detach().requires_grad_(True)


def _build_lora_leaves(mode: str, *, E: int, H: int, I: int, r: int, dtype: torch.dtype, device: torch.device):
    """Build a fresh, deterministic set of LoRA leaf tensors for ``mode``.

    Mirrors the seed-style two-LoRA layout of the wrappers / autograd
    classes: ``gate`` and ``up`` each get their own rank-r adapter on the
    fused gate_up base weight, plus a single ``down`` adapter. Re-seeded by
    the caller for cross-branch reproducibility (same seed → same initial
    values) so EP and non-EP branches start from identical LoRA state.
    """
    if mode == "shared":
        return {
            "lora_a_gate": _make_lora_leaf(r, H, dtype=dtype, device=device),
            "lora_b_gate": _make_lora_leaf(I, r, dtype=dtype, device=device),
            "lora_a_up": _make_lora_leaf(r, H, dtype=dtype, device=device),
            "lora_b_up": _make_lora_leaf(I, r, dtype=dtype, device=device),
            "lora_a_down": _make_lora_leaf(r, I, dtype=dtype, device=device),
            "lora_b_down": _make_lora_leaf(H, r, dtype=dtype, device=device),
        }
    return {
        "lora_a_gate": _make_lora_leaf(E, r, H, dtype=dtype, device=device),
        "lora_b_gate": _make_lora_leaf(E, I, r, dtype=dtype, device=device),
        "lora_a_up": _make_lora_leaf(E, r, H, dtype=dtype, device=device),
        "lora_b_up": _make_lora_leaf(E, I, r, dtype=dtype, device=device),
        "lora_a_down": _make_lora_leaf(E, r, I, dtype=dtype, device=device),
        "lora_b_down": _make_lora_leaf(E, H, r, dtype=dtype, device=device),
    }


@pytest.mark.parametrize("mode", ["shared", "independent"])
def test_ep_class_matches_nonep_class_single_rank(mode):
    """EP autograd class output AND LoRA grads match the non-EP class on the same permuted token block.

    Math equivalence: routing-weight scaling commutes with the linear ``down``
    + LoRA-down chain, so applying ``scattered_gate_weights`` *after* fc2 (the
    EP convention, via ``tokens_post_all2all`` → ``unpermute``) and applying
    it *before* fc2 (the non-EP convention, baked into the class) produce the
    same forward output and same LoRA gradients up to bf16 reduction-order
    noise. The forward leg is the easy half; the backward leg also closes
    over ``grad_lora_a_*`` / ``grad_lora_b_*`` per-expert chains, which is
    where any bug in the EP autograd backward would surface.
    """
    _require_cuda_with_triton()

    from veomni.ops.kernels.moe._kernels.kernel.moe import (
        expert_histogram,
        moe_gather,
        moe_scatter,
    )
    from veomni.ops.kernels.moe.lora_group_gemm import (
        EPMergedFc1IndependentLoRAGroupGemm,
        EPMergedFc1SharedLoRAGroupGemm,
        MergedFc1IndependentTritonFusedLoRAMoeExpertFunction,
        MergedFc1TritonFusedLoRAMoeExpertFunction,
    )

    _CLASSES = {
        "shared": (EPMergedFc1SharedLoRAGroupGemm, MergedFc1TritonFusedLoRAMoeExpertFunction),
        "independent": (EPMergedFc1IndependentLoRAGroupGemm, MergedFc1IndependentTritonFusedLoRAMoeExpertFunction),
    }
    ep_cls, nonep_cls = _CLASSES[mode]

    dev = torch.device("cuda")
    dtype = torch.bfloat16
    B, H, I, E, top_k, r = 32, 64, 96, 4, 2, 8
    scale_gate, scale_up, scale_down = 0.5, 0.5, 0.5

    # Build the shared inputs once — base weights stay frozen, the EP-side
    # permuted view is computed by manually replicating the non-EP class's
    # internal scatter/cumsum so we can hand the EP class the same token block.
    torch.manual_seed(0)
    hidden_states = torch.randn(B, H, dtype=dtype, device=dev)
    top_k_index = torch.randint(0, E, (B, top_k), device=dev)
    top_k_weights = torch.softmax(torch.randn(B, top_k, dtype=torch.float32, device=dev), dim=-1).to(dtype)

    splits = expert_histogram(top_k_index, E)
    scatter_index = top_k_index.flatten().argsort(stable=True).argsort().int().view(top_k_index.shape)
    permute_tokens = moe_scatter(hidden_states, scatter_index)  # [T, H], grouped by expert
    cumsum = torch.cumsum(splits, dim=0)
    T = permute_tokens.shape[0]
    scattered_gate_weights = torch.empty(T, 1, dtype=dtype, device=dev)
    scattered_gate_weights[scatter_index.flatten()] = top_k_weights.reshape(-1, 1)

    gate_up_proj = (torch.randn(E, 2 * I, H, dtype=dtype, device=dev) * 0.05).detach()
    down_proj = (torch.randn(E, H, I, dtype=dtype, device=dev) * 0.05).detach()

    # Argument order for both EP and non-EP autograd.Function.apply, after
    # the leading positional args (permute_tokens/cumsum or
    # num_experts/top_k_weights/top_k_index/hidden_states + base weights).
    # Matches ``MergedFc1*Function.forward`` and ``EPMergedFc1*GroupGemm.forward``.
    _LORA_KEYS = ("lora_a_gate", "lora_b_gate", "lora_a_up", "lora_b_up", "lora_a_down", "lora_b_down")

    def _build_branch(*, ep: bool):
        """Build one branch's LoRA leaves + run its forward; return (output, lora_dict).

        Re-seeding ``torch.manual_seed(123)`` before ``_build_lora_leaves`` makes
        both branches start from byte-identical LoRA values, which is the only
        way the per-tensor grad parity below is meaningful. Each branch owns
        its own leaf tensors so ``torch.autograd.grad`` calls are isolated.
        """
        torch.manual_seed(123)
        lora = _build_lora_leaves(mode, E=E, H=H, I=I, r=r, dtype=dtype, device=dev)
        if ep:
            out = ep_cls.apply(
                permute_tokens,
                cumsum,
                gate_up_proj,
                down_proj,
                *(lora[k] for k in _LORA_KEYS),
                scale_gate,
                scale_up,
                scale_down,
            )
        else:
            out = nonep_cls.apply(
                E,
                top_k_weights,
                top_k_index,
                hidden_states,
                gate_up_proj,
                down_proj,
                *(lora[k] for k in _LORA_KEYS),
                scale_gate,
                scale_up,
                scale_down,
            )
        return out, lora

    nonep_out, nonep_lora = _build_branch(ep=False)  # [B, H], routing weight baked in
    ep_permuted, ep_lora = _build_branch(ep=True)  # [T, H], routing weight applied later

    # ── Forward parity ───────────────────────────────────────────────────────
    # Replay the EP post-step (sgw multiply + unpermute) under no_grad — only
    # needed for the forward comparison; the backward leg uses a synthetic
    # upstream grad below to avoid having to differentiate ``moe_gather``.
    with torch.no_grad():
        ep_out = moe_gather(ep_permuted.detach() * scattered_gate_weights, scatter_index).reshape(hidden_states.shape)
    fwd_l2 = _l2_rel(ep_out, nonep_out.detach())
    assert fwd_l2 <= _FWD_L2REL_TOL, (
        f"[{mode}] EP-vs-non-EP single-rank forward parity broken: "
        f"L2 rel {fwd_l2:.4%} > {_FWD_L2REL_TOL:.2%} (ref_norm={nonep_out.float().norm().item():.3e})"
    )

    # ── Backward parity (per LoRA tensor) ────────────────────────────────────
    # Use a single synthetic upstream grad on the [B, H] output instead of a
    # pow(2).sum() loss, so both branches see *identical* upstream signal (no
    # bf16-forward noise leaking into the grad comparison).
    #
    # For the EP class, autograd doesn't reach back through ``moe_gather``
    # (raw Triton kernel, no backward registered), so we hand-derive the
    # equivalent upstream grad on the permuted-token tensor:
    #   ``ep_out[b]   = sum_k permuted[scatter_index[b,k]] * sgw[scatter_index[b,k]]``
    #   ``∂loss/∂permuted[t] = grad_out[b] * sgw[t]``
    # which is exactly ``moe_scatter(grad_out, scatter_index) * sgw`` because
    # ``scatter_index`` is a permutation in this synthetic setup.
    torch.manual_seed(456)
    grad_out = (torch.randn(B, H, dtype=dtype, device=dev) * 0.1).detach()
    grad_permuted = (moe_scatter(grad_out, scatter_index) * scattered_gate_weights).detach()

    nonep_grads = dict(
        zip(
            _LORA_KEYS,
            torch.autograd.grad(nonep_out, [nonep_lora[k] for k in _LORA_KEYS], grad_outputs=grad_out),
        )
    )
    ep_grads = dict(
        zip(
            _LORA_KEYS,
            torch.autograd.grad(ep_permuted, [ep_lora[k] for k in _LORA_KEYS], grad_outputs=grad_permuted),
        )
    )

    for name in _LORA_KEYS:
        g_nonep, g_ep = nonep_grads[name], ep_grads[name]
        assert g_nonep.shape == g_ep.shape, (
            f"[{mode}] {name}: grad shape mismatch nonep={g_nonep.shape} ep={g_ep.shape}"
        )
        l2 = _l2_rel(g_ep, g_nonep)
        assert l2 <= _GRAD_L2REL_TOL, (
            f"[{mode}] {name}: EP-vs-non-EP backward parity broken — L2 rel {l2:.4%} > {_GRAD_L2REL_TOL:.2%} "
            f"(nonep_norm={g_nonep.float().norm().item():.3e}, max|Δ|={(g_nonep - g_ep).abs().max().item():.3e})"
        )
