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
"""Smoke tests for VeOmni's MoE-LoRA wrappers (eager, single-process, bf16).

Covers BOTH wrapper flavours, parametrised on the ``mode`` axis:

* ``mode="shared"`` → :class:`veomni.lora.moe_layers.LoraSharedExperts` (Mode 2,
  one LoRA pair per layer, broadcast across all experts).
* ``mode="independent"`` → :class:`veomni.lora.moe_layers.LoraIndependentExperts`
  (Mode 1 — the trainer default — one LoRA pair *per expert*, 3-D tensors).

What this exercises (per ``mode`` × per toy):

1. The wrapper validates the experts layout (fused ``gate_up_proj`` /
   ``down_proj`` — see :func:`veomni.lora.moe_layers._validate_fused_layout`)
   and matches the yaml-declared ``target_parameters``.
2. The wrapper is a true no-op at init (per-expert kaiming-uniform A, zero B).
3. Backward only flows through ``<spec>.lora_A`` / ``<spec>.lora_B`` parameters
   (PEFT-aligned per-spec sub-modules); the base experts module is fully frozen.
4. End-to-end save/reload round-trip via PEFT + the
   ``veomni_moe_lora.json`` sidecar produces a model whose forward output
   is bit-identical to the in-memory trained model — exercised for both
   modes on Qwen3-MoE.

Transformers version gating:
    VeOmni MoE-LoRA is v5-only. Each toy declares a
    ``min_transformers_version`` in ``tests/lora/utils.py::TOY_LORA_SPECS``
    that mirrors the cutoff in each model's ``__init__.py``;
    :func:`tests.lora.utils.select_lora_yaml` calls ``pytest.skip`` on
    older envs (e.g. ``qwen3_5_moe`` requires transformers >= 5.2.0).

Whole-model build is fragile for some toy configs (e.g. Qwen3.5-MoE's
toy expects a ``mm_token_type_ids`` kwarg in forward; Qwen3-Omni-MoE's toy
historically failed at module import on a docstring validator). To stay
robust against those unrelated issues, the per-model checks call
``experts.forward`` *directly* on randomly-initialised inputs after wrapping;
the round-trip test uses Qwen3-MoE (the most stable toy model) for whole-model
``model.save_pretrained`` / ``PeftModel.from_pretrained``.

Device policy:
    Model build dominates total runtime (~30s/toy on CPU vs ~0.7s on a single
    A100). The toy configs are also relatively large (~1B params) so CPU build
    is impractical for local iteration. We therefore build / run on CUDA when
    available and fall back to CPU otherwise. CI without a GPU still works
    but is slow; mark the suite as ``cuda`` for selective runs.

Build / yaml / glob helpers live in ``tests/lora/utils.py`` and are shared
with future LoRA tests (e.g. EP alignment in Phase 6).

Run:
    pytest -v tests/lora/test_moe_lora_eager.py
"""

from __future__ import annotations

import warnings

import pytest
import torch

from veomni.lora import resolve_fused_moe_lora_targets
from veomni.lora.moe_layers import (
    _LORA_SPEC_KEYS,
    LoraIndependentExperts,
    LoraSharedExperts,
    apply_independent_moe_lora,
    apply_shared_moe_lora,
)

from .utils import (
    build_toy,
    experts_module_globs,
    find_all_matching_modules,
    find_first_matching_module,
    load_lora_config,
)


# ---------------------------------------------------------------------------
# Mode dispatch table: keep test bodies mode-agnostic by routing through here.
# ---------------------------------------------------------------------------

# (mode_label, apply_fn, wrapper_cls, expected_lora_param_ndim)
_MODE_TABLE = {
    "shared": (apply_shared_moe_lora, LoraSharedExperts, 2),
    "independent": (apply_independent_moe_lora, LoraIndependentExperts, 3),
}


def _apply(mode: str, *args, **kwargs):
    return _MODE_TABLE[mode][0](*args, **kwargs)


def _wrapper_cls(mode: str):
    return _MODE_TABLE[mode][1]


def _expected_ndim(mode: str) -> int:
    return _MODE_TABLE[mode][2]


# ---------------------------------------------------------------------------
# Parametrised per-model wrapper tests
# ---------------------------------------------------------------------------

# Toys to exercise. The expected count of experts modules to wrap is derived
# *from the built model* via :func:`find_all_matching_modules` so a toy adding
# or removing MoE layers doesn't silently desync from a hardcoded number here.
_MODEL_CASES = [
    pytest.param("qwen3_moe_toy", id="qwen3_moe"),
    pytest.param("qwen3_5_moe_toy", id="qwen3_5_moe"),
    pytest.param("qwen3vlmoe_toy", id="qwen3_vl_moe"),
    pytest.param("qwen3omni_toy", id="qwen3_omni_moe"),
    pytest.param("deepseek_v3_toy", id="deepseek_v3"),
]

_MODE_CASES = [
    pytest.param("shared", id="shared"),
    pytest.param("independent", id="independent"),
]


def _select_yaml_then_build(toy_dir: str):
    """``load_lora_config`` first → ``build_toy`` second.

    The ``load_lora_config`` call resolves the yaml via ``select_lora_yaml``,
    which raises ``pytest.skip`` when the installed transformers version is
    older than the toy's ``min_transformers_version`` (e.g. ``qwen3_5_moe``
    requires transformers >= 5.2.0). Doing that BEFORE ``build_toy`` matters
    because some toys reference model architectures that simply don't exist
    on older transformers — without this order, ``build_toy`` would raise
    ``Unknown ModelConfig`` instead of producing a clean skip.
    """
    lora_cfg = load_lora_config(toy_dir)  # may pytest.skip here
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_toy(toy_dir)
    # Mirror BaseTrainer._setup_lora: map any semantic MoE module names
    # (gate_proj / up_proj / down_proj) onto the model's fused expert
    # target_parameters. No-op for configs that already list explicit patterns.
    lora_cfg = resolve_fused_moe_lora_targets(model, lora_cfg)
    return model, lora_cfg


@pytest.mark.parametrize("mode", _MODE_CASES)
@pytest.mark.parametrize("toy_dir", _MODEL_CASES)
def test_layout_validate_and_wrap(toy_dir: str, mode: str):
    """Wrapping with the paired yaml's ``target_parameters`` must replace exactly the experts modules.

    Also asserts the wrapper exposes the canonical seed-style LoRA spec set
    (``gate_proj`` / ``up_proj`` / ``down_proj`` — three logical pairs per
    wrapped module, with ``gate_up_proj`` decomposing into two independent
    rank-r adapters) and that LoRA tensor rank matches the mode (2-D for
    shared, 3-D for independent).
    """
    model, lora_cfg = _select_yaml_then_build(toy_dir)
    patterns = lora_cfg["target_parameters"]
    # Count experts modules in the *built* model that match the yaml patterns —
    # the wrapper should replace every one of them and nothing else.
    expected_fqns = find_all_matching_modules(model, experts_module_globs(patterns))
    assert expected_fqns, (
        f"{toy_dir}: yaml patterns {patterns} matched no experts module in the built model — stale yaml?"
    )
    wrapped = _apply(
        mode,
        model,
        target_parameter_patterns=patterns,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        freeze_base_model=True,
    )
    assert sorted(wrapped) == expected_fqns, (
        f"{toy_dir}/{mode}: wrapped FQNs {sorted(wrapped)} ≠ expected (matching) {expected_fqns}"
    )
    # Wrapper allocates the canonical 3 logical specs unconditionally for the
    # v5 fused experts layout — ``gate_up_proj`` matched in the yaml expands
    # to ``gate_proj`` + ``up_proj`` (seed-style two-LoRA), and ``down_proj``
    # stays a single pair.
    expected_specs = set(_LORA_SPEC_KEYS)
    expected_cls = _wrapper_cls(mode)
    expected_lora_ndim = _expected_ndim(mode)
    for fqn in wrapped:
        w = model.get_submodule(fqn)
        assert isinstance(w, expected_cls), f"{fqn}: expected {expected_cls.__name__}, got {type(w).__name__}"
        assert set(w._lora_specs) == expected_specs, (
            f"{fqn}: lora_specs {set(w._lora_specs)} ≠ canonical {expected_specs}"
        )
        # Spot-check tensor rank on every logical spec: 2-D for shared (one
        # matrix per layer), 3-D for independent (leading expert dim).
        # Catches accidental cross-mode regressions and any half (gate / up /
        # down) that fails to allocate.
        for spec_name in expected_specs:
            a_w = w.get_lora_A_weight(spec_name)
            assert a_w.ndim == expected_lora_ndim, (
                f"{fqn}/{mode}/{spec_name}: expected lora_A ndim={expected_lora_ndim}, "
                f"got {a_w.ndim} (shape={tuple(a_w.shape)})"
            )


@pytest.mark.parametrize("mode", _MODE_CASES)
@pytest.mark.parametrize("toy_dir", _MODEL_CASES)
def test_eager_forward_no_op_at_init(toy_dir: str, mode: str):
    """Direct experts.forward() with kaiming-A / zero-B must reproduce the base output exactly."""
    model, lora_cfg = _select_yaml_then_build(toy_dir)
    patterns = lora_cfg["target_parameters"]
    sample_fqn, exp = find_first_matching_module(model, experts_module_globs(patterns))
    # Build a synthetic experts call on the model's device.
    H, E = exp.hidden_dim, exp.num_experts
    top_k, N = 2, 8
    p0 = next(exp.parameters())
    dtype, dev = p0.dtype, p0.device
    h = torch.randn(N, H, dtype=dtype, device=dev)
    top_k_index = torch.randint(0, E, (N, top_k), device=dev)
    top_k_weights = torch.softmax(torch.randn(N, top_k, dtype=torch.float32, device=dev), dim=-1).to(dtype)

    with torch.no_grad():
        base_out = exp(h, top_k_index, top_k_weights).clone()

    _apply(
        mode,
        model,
        target_parameter_patterns=patterns,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        freeze_base_model=True,
    )
    wrapper = model.get_submodule(sample_fqn)
    with torch.no_grad():
        wrap_out = wrapper(h, top_k_index, top_k_weights)
    diff = (wrap_out - base_out).abs().max().item()
    assert diff == 0.0, f"{toy_dir}/{mode}: LoRA must be no-op at init, got max|delta|={diff}"


@pytest.mark.parametrize("mode", _MODE_CASES)
@pytest.mark.parametrize("toy_dir", _MODEL_CASES)
def test_backward_isolates_to_lora_params(toy_dir: str, mode: str):
    """Backward through a wrapped experts module must only fill grads for ``<spec>.lora_A`` / ``<spec>.lora_B``."""
    model, lora_cfg = _select_yaml_then_build(toy_dir)
    patterns = lora_cfg["target_parameters"]
    sample_fqn, exp = find_first_matching_module(model, experts_module_globs(patterns))
    H, E = exp.hidden_dim, exp.num_experts
    p0 = next(exp.parameters())
    dtype, dev = p0.dtype, p0.device
    _apply(
        mode,
        model,
        target_parameter_patterns=patterns,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        freeze_base_model=True,
    )
    wrapper = model.get_submodule(sample_fqn)
    wrapper.train()

    h = torch.randn(8, H, dtype=dtype, device=dev)
    top_k_index = torch.randint(0, E, (8, 2), device=dev)
    top_k_weights = torch.softmax(torch.randn(8, 2, dtype=torch.float32, device=dev), dim=-1).to(dtype)
    out = wrapper(h, top_k_index, top_k_weights)
    out.float().pow(2).sum().backward()

    # PEFT-aligned wrapper layout: LoRA params live at
    # ``<spec>.lora_A.<adapter>.weight`` / ``<spec>.lora_B.<adapter>.weight``,
    # base params at ``<spec>.base_layer.weight``. We classify by checking
    # for the canonical ``lora_A`` / ``lora_B`` / ``base_layer`` segment in
    # the dot-split path so the check is robust to any FSDP renaming.
    n_lora_with_grad = 0
    n_base_with_grad = 0
    for n, p in wrapper.named_parameters():
        if p.grad is None or p.grad.abs().sum().item() == 0:
            continue
        parts = n.split(".")
        if "lora_A" in parts or "lora_B" in parts:
            n_lora_with_grad += 1
        elif "base_layer" in parts:
            n_base_with_grad += 1
    # At init lora_B == 0 ⇒ dL/dlora_A = 0 (chain through B). So only lora_B
    # params gain grad. The seed-style fused experts layout has three logical
    # targets (gate + up + down), so we expect at least 3 lora_B params with
    # non-zero grad in both modes.
    assert n_base_with_grad == 0, f"{toy_dir}/{mode}: base layer must stay frozen, got {n_base_with_grad}"
    assert n_lora_with_grad >= 3, (
        f"{toy_dir}/{mode}: expected at least 3 lora_B params with grad "
        f"(gate + up + down halves), got {n_lora_with_grad}"
    )


# ---------------------------------------------------------------------------
# Whole-model save/reload round-trip — exercised for BOTH modes on Qwen3-MoE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", _MODE_CASES)
def test_save_reload_round_trip_qwen3_moe(tmp_path, mode: str):
    """End-to-end (native stack): VeOmniLoraModel (linears + MoE-LoRA patterns) → save → reload → identical fwd.

    Run for both ``shared`` (Mode 2) and ``independent`` (Mode 1) flavours so the
    PEFT-free ``VeOmniLoraModel`` save / ``from_pretrained`` + ``load_lora_weights``
    path is covered for each. Asserts:

    * Init delta vs base is < 1e-3 (kaiming-A / zero-B no-op).
    * After perturbing ``<spec>.lora_B`` (so the LoRA contribution is non-trivial),
      reloading from disk produces a *bit-identical* forward + LoRA state-dict.
    """
    from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
    from veomni.lora.state_dict import get_lora_state_dict
    from veomni.lora.weight_loading import load_lora_weights

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_toy("qwen3_moe_toy")

    lora_cfg = resolve_fused_moe_lora_targets(model, load_lora_config("qwen3_moe_toy"))
    rank, alpha = lora_cfg["rank"], lora_cfg["alpha"]
    patterns = lora_cfg["target_parameters"]
    linear_targets = lora_cfg["lora_modules"]

    model.eval()
    dev = next(model.parameters()).device
    input_ids = torch.randint(0, 1000, (1, 16), device=dev)
    with torch.no_grad():
        base_out = model(input_ids=input_ids).logits.clone()
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    expected_n_experts = len(find_all_matching_modules(model, experts_module_globs(patterns)))
    assert expected_n_experts > 0, f"qwen3_moe_toy: yaml patterns {patterns} matched no experts module — stale yaml?"

    # Native wrap: dense linears + MoE-LoRA in one config. ``share_expert_lora``
    # selects the wrapper flavour (shared→Mode 2, independent→Mode 1).
    cfg = VeOmniLoraConfig.from_yaml(
        {
            "rank": rank,
            "alpha": alpha,
            "lora_modules": linear_targets,
            "target_parameters": patterns,
            "share_expert_lora": mode == "shared",
        }
    )
    wrapped_model = VeOmniLoraModel(model, cfg)
    from veomni.lora.moe_layers import is_lora_moe_experts

    n_wrapped = sum(1 for _, m in wrapped_model.named_modules() if is_lora_moe_experts(m))
    assert n_wrapped == expected_n_experts, (
        f"{mode}: wrapped {n_wrapped} experts modules, expected {expected_n_experts}"
    )

    # No-op at init.
    wrapped_model.eval()
    with torch.no_grad():
        delta_init = (wrapped_model(input_ids=input_ids).logits - base_out).abs().max().item()
    assert delta_init < 1e-3, f"{mode}: LoRA must be no-op at init, got {delta_init}"

    # Perturb to make the post-train state non-trivial.
    with torch.no_grad():
        for n, p in wrapped_model.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.02)

    wrapped_model.eval()
    with torch.no_grad():
        trained_out = wrapped_model(input_ids=input_ids).logits.clone()

    # Save the PEFT-format adapter (adapter_config.json + adapter_model files).
    save_dir = str(tmp_path / "adapter")
    wrapped_model.save_pretrained(save_dir)

    # Reload into a fresh model with the SAME base weights via from_pretrained.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model2 = build_toy("qwen3_moe_toy")
    model2.load_state_dict(base_state)
    model2.eval()
    reloaded = VeOmniLoraModel.from_pretrained(model2, save_dir, is_trainable=True)
    load_lora_weights(reloaded, save_dir, init_device="cpu")
    reloaded.eval()
    with torch.no_grad():
        reload_out = reloaded(input_ids=input_ids).logits

    # Bit-identical forward output.
    delta = (reload_out - trained_out).abs().max().item()
    assert delta == 0.0, f"{mode}: reload parity broken: max|reload-trained|={delta}"

    # Bit-identical LoRA parameter tensors (PEFT on-disk key format).
    s1 = get_lora_state_dict(wrapped_model, config=wrapped_model.get_lora_config())
    s2 = get_lora_state_dict(reloaded, config=reloaded.get_lora_config())
    assert set(s1) == set(s2), f"{mode}: reload key set mismatch"
    for k in s1:
        assert torch.equal(s1[k], s2[k]), f"{mode}: value mismatch: {k}"
