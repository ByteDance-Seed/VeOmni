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
"""Triton fused MoE-LoRA forward parity tests (Phase 4, non-EP).

Covers both layouts handled by
:class:`veomni.utils.moe_lora.LoraSharedExperts`:

* **v5 fused** (``_fused_forward_fused_v5``) — exercised when the installed
  ``transformers`` is on the v5 modeling path (>= 5.0.0 for qwen3_moe).
* **v4 split** (``_fused_forward_split_v4``) — exercised when the installed
  ``transformers`` is on the v4 modeling path (< 5.0.0 for qwen3_moe).

Both branches use the same :func:`build_toy` + :func:`load_lora_config`
machinery as ``test_moe_lora_eager.py``; the version-routed
``veomni/models/transformers/qwen3_moe/__init__.py`` selects the matching
modeling code at import time, so a single test body covers both layouts.

Each branch flips ``moe_implementation`` between ``fused_triton`` (kernel
path) and ``eager`` (reference) and compares:

1. forward outputs at small bf16 tolerance, and
2. d/dlora_A, d/dlora_B at small bf16 tolerance.

Out of scope (planned later):
* EP path — Phase 5.
* Per-expert (Mode 1) LoRA — wrapper currently only supports Mode 2 (shared).

Run:
    pytest -v tests/lora/test_moe_lora_fused.py
"""

from __future__ import annotations

import warnings

import pytest
import torch

from veomni.utils.moe_lora import LoraSharedExperts, apply_shared_moe_lora

from .utils import (
    build_toy,
    experts_module_globs,
    find_first_matching_module,
    fused_triton_moe_ops,
    load_lora_config,
)


_TOY = "qwen3_moe_toy"  # Phase 4 / Phase 6 scope: qwen3_moe (v4 split + v5 fused, both via toy).

# Forward and backward parity are checked via L2 relative error
# (``||fused - eager|| / ||eager||``) instead of element-wise atol/rtol.
# Rationale: in bf16 the per-expert group-gemm and the LoRA F.linear chain
# have different reduction orders than the eager per-expert loop. Catastrophic
# cancellation can flip a single output near zero by O(1) units while the
# overall tensor stays accurate to <1% in L2. Element-wise allclose flags this
# as a divergence; L2-relative is the standard gradient-parity metric used in
# kernel correctness tests (matches what
# ``tests/ops/test_fused_moe_split_vs_merged.py`` is implicitly tolerating
# with its ``rtol=3e-2, atol=3e-2`` bounds, but expressed directly).
_FWD_L2REL_TOL = 0.02  # 2% — forward is one chain of group-gemm + add + matmul.
_GRAD_L2REL_TOL = 0.02  # 2% — backward stacks a bf16 matmul on top of the dgrad.


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
    ``_fused_moe_forward`` / ``_fused_lora_moe_forward``. Restoring afterwards
    keeps unrelated MoE tests deterministic regardless of run order.
    """
    from veomni.ops.kernels import moe as _moe_ops

    saved_base = _moe_ops._fused_moe_forward
    saved_lora = _moe_ops._fused_lora_moe_forward
    try:
        yield
    finally:
        _moe_ops._fused_moe_forward = saved_base
        _moe_ops._fused_lora_moe_forward = saved_lora


def _wrap_with_lora(model, lora_cfg, *, lora_b_perturb_std: float = 0.0):
    """Wrap ``model`` with shared MoE-LoRA matching the toy's yaml; optionally bump lora_B off zero."""
    apply_shared_moe_lora(
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


def _build_wrapped(*, fused: bool, lora_b_perturb_std: float = 0.02):
    """Build a fresh wrapped Qwen3-MoE toy with the chosen ops backend.

    Same RNG seed for both invocations → identical base + LoRA tensors. The
    only difference is which ``moe_implementation`` was patched at build time,
    which determines whether ``_fused_lora_moe_forward`` is bound. Layout
    (v5 fused vs v4 split) is whatever the version-routed
    ``veomni/models/transformers/qwen3_moe/__init__.py`` resolves to for the
    installed ``transformers``.
    """
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_toy(_TOY, ops=fused_triton_moe_ops() if fused else None)
    lora_cfg = load_lora_config(_TOY)
    _wrap_with_lora(model, lora_cfg, lora_b_perturb_std=lora_b_perturb_std)
    sample_fqn, exp = find_first_matching_module(model, experts_module_globs(lora_cfg["target_parameters"]))
    return model, sample_fqn, exp, lora_cfg


def test_fused_pointer_bound_after_fused_triton_build():
    """Sanity check: building with ``moe_implementation=fused_triton`` actually binds the LoRA pointer."""
    _require_cuda_with_triton()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_toy(_TOY, ops=fused_triton_moe_ops())
    from veomni.ops.kernels import moe as _moe_ops

    assert _moe_ops._fused_lora_moe_forward is not None, (
        "apply_veomni_fused_moe_patch('triton') must bind _fused_lora_moe_forward."
    )
    assert _moe_ops._fused_moe_forward is not None, "Sanity: base fused MoE pointer should also be bound."


def test_wrapper_dispatch_chooses_fused_when_pointer_bound():
    """``LoraSharedExperts.forward`` must take the fused branch when the pointer is bound, eager otherwise."""
    _require_cuda_with_triton()

    # Path A: fused build → wrapper.forward calls the layout's fused branch.
    model_f, fqn_f, exp_f, lora_cfg = _build_wrapped(fused=True, lora_b_perturb_std=0.0)
    wrapper_f = model_f.get_submodule(fqn_f)
    assert isinstance(wrapper_f, LoraSharedExperts)
    assert wrapper_f._layout in ("fused_v5", "split_v4"), f"unexpected layout {wrapper_f._layout!r}"
    h, idx, w = _make_inputs(exp_f)
    out_f = wrapper_f(h, idx, w)
    assert out_f.shape == h.shape

    # Path B: eager build → pointer is None → wrapper falls back to eager.
    from veomni.ops.kernels import moe as _moe_ops

    _moe_ops._fused_moe_forward = None
    _moe_ops._fused_lora_moe_forward = None
    model_e, fqn_e, exp_e, _ = _build_wrapped(fused=False, lora_b_perturb_std=0.0)
    wrapper_e = model_e.get_submodule(fqn_e)
    out_e = wrapper_e(h.cpu().to(next(exp_e.parameters()).device), idx, w)
    assert out_e.shape == h.shape


def test_fused_vs_eager_forward_parity():
    """Forward output of the triton fused path matches the eager wrapper at bf16 tol."""
    _require_cuda_with_triton()

    # Eager reference first (so the autouse fixture's saved pointers match the eager state).
    model_e, fqn_e, exp_e, lora_cfg = _build_wrapped(fused=False, lora_b_perturb_std=0.02)
    h, idx, w = _make_inputs(exp_e)
    wrapper_e = model_e.get_submodule(fqn_e)
    with torch.no_grad():
        out_eager = wrapper_e(h, idx, w).clone()

    # Fused path — rebuild so apply_veomni_fused_moe_patch("triton") binds the kernel.
    model_f, fqn_f, _exp_f, _ = _build_wrapped(fused=True, lora_b_perturb_std=0.02)
    wrapper_f = model_f.get_submodule(fqn_f)
    # Sanity: identical seed / wrap → identical LoRA tensors → makes the parity check meaningful.
    # Iterate the layout's actual LoRA targets (v5: gate_up_proj/down_proj; v4: gate_proj/up_proj/down_proj).
    for pname in wrapper_e._lora_specs:
        assert torch.equal(wrapper_e.get_lora_A_weight(pname), wrapper_f.get_lora_A_weight(pname))
        assert torch.equal(wrapper_e.get_lora_B_weight(pname), wrapper_f.get_lora_B_weight(pname))
    with torch.no_grad():
        out_fused = wrapper_f(h, idx, w)

    l2 = _l2_rel(out_fused, out_eager)
    assert l2 <= _FWD_L2REL_TOL, (
        f"forward parity broken ({wrapper_e._layout}): L2 relative error {l2:.4%} > {_FWD_L2REL_TOL:.2%} "
        f"(eager_norm={out_eager.float().norm().item():.3e})"
    )


def test_fused_vs_eager_backward_parity():
    """Gradients on lora_A_* / lora_B_* match between fused and eager at bf16 tol."""
    _require_cuda_with_triton()

    def _grads(*, fused: bool):
        model, fqn, exp, _ = _build_wrapped(fused=fused, lora_b_perturb_std=0.02)
        wrapper = model.get_submodule(fqn)
        wrapper.train()
        h, idx, w = _make_inputs(exp)
        # Fixed loss (sum-of-squares) so both paths see the same upstream grad pattern.
        loss = wrapper(h, idx, w).float().pow(2).sum()
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in wrapper.named_parameters() if p.grad is not None}
        return wrapper._layout, grads

    layout_e, grads_eager = _grads(fused=False)
    layout_f, grads_fused = _grads(fused=True)
    assert layout_e == layout_f, f"layout mismatch eager={layout_e!r} fused={layout_f!r}"

    assert set(grads_eager) == set(grads_fused), (
        f"different param sets received grad: only-eager={set(grads_eager) - set(grads_fused)}, "
        f"only-fused={set(grads_fused) - set(grads_eager)}"
    )
    # Spot-check the LoRA grads — these are the only ones that should be non-zero
    # (base is frozen, perturbed lora_B → kaiming A still gets gradient via B).
    lora_param_names = sorted(n for n in grads_eager if n.startswith("lora_A_") or n.startswith("lora_B_"))
    assert lora_param_names, "expected lora_A_*/lora_B_* params to receive gradients"
    for n in lora_param_names:
        ge, gf = grads_eager[n], grads_fused[n]
        assert ge.shape == gf.shape, f"{n}: shape mismatch eager={ge.shape} fused={gf.shape}"
        l2 = _l2_rel(gf, ge)
        assert l2 <= _GRAD_L2REL_TOL, (
            f"{n} ({layout_e}): grad parity broken — L2 relative error {l2:.4%} > {_GRAD_L2REL_TOL:.2%} "
            f"(eager_norm={ge.float().norm().item():.3e}, max|fused-eager|={(ge - gf).abs().max().item():.3e})"
        )
