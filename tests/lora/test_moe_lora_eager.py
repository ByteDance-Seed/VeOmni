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

* ``mode="shared"`` → :class:`veomni.utils.moe_lora.LoraSharedExperts` (Mode 2,
  one LoRA pair per layer, broadcast across all experts).
* ``mode="independent"`` → :class:`veomni.utils.moe_lora.LoraIndependentExperts`
  (Mode 1 — the trainer default — one LoRA pair *per expert*, 3-D tensors).

What this exercises (per ``mode`` × per toy):

1. The wrapper validates the experts layout (fused ``gate_up_proj`` /
   ``down_proj`` — see :func:`veomni.utils.moe_lora._validate_fused_layout`)
   and matches the yaml-declared ``target_parameters``.
2. The wrapper is a true no-op at init (per-expert kaiming-uniform A, zero B).
3. Backward only flows through ``lora_A_*`` / ``lora_B_*`` parameters; the
   base experts module is fully frozen.
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

import os
import warnings

import pytest
import torch

from veomni.utils.moe_lora import (
    LoraIndependentExperts,
    LoraSharedExperts,
    apply_independent_moe_lora,
    apply_moe_lora_from_sidecar,
    apply_shared_moe_lora,
    read_moe_lora_sidecar,
    write_moe_lora_sidecar,
)

from .utils import (
    build_toy,
    experts_module_globs,
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

# Each entry is (toy_config_dir, expected_layer_count_for_assertion).
_MODEL_CASES = [
    pytest.param("qwen3_moe_toy", 4, id="qwen3_moe"),
    pytest.param("qwen3_5_moe_toy", 4, id="qwen3_5_moe"),
    pytest.param("qwen3vlmoe_toy", 2, id="qwen3_vl_moe"),
    pytest.param("qwen3omni_toy", 2, id="qwen3_omni_moe"),
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
    return model, lora_cfg


@pytest.mark.parametrize("mode", _MODE_CASES)
@pytest.mark.parametrize("toy_dir,expected_n_layers", _MODEL_CASES)
def test_layout_validate_and_wrap(toy_dir: str, expected_n_layers: int, mode: str):
    """Wrapping with the paired yaml's ``target_parameters`` must replace exactly the experts modules.

    Also asserts the wrapper picks up the yaml-declared LoRA specs
    (``gate_up_proj`` / ``down_proj``) and that LoRA tensor rank matches
    the mode (2-D for shared, 3-D for independent).
    """
    model, lora_cfg = _select_yaml_then_build(toy_dir)
    patterns = lora_cfg["target_parameters"]
    wrapped = _apply(
        mode,
        model,
        target_parameter_patterns=patterns,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        freeze_base_model=True,
    )
    assert len(wrapped) == expected_n_layers, (
        f"{toy_dir}/{mode}: expected {expected_n_layers} wrapped experts modules, got {len(wrapped)}: {wrapped}"
    )
    expected_specs = {p.rsplit(".", 1)[1] for p in patterns}
    expected_cls = _wrapper_cls(mode)
    expected_lora_ndim = _expected_ndim(mode)
    for fqn in wrapped:
        w = model.get_submodule(fqn)
        assert isinstance(w, expected_cls), f"{fqn}: expected {expected_cls.__name__}, got {type(w).__name__}"
        assert set(w._lora_specs) == expected_specs, (
            f"{fqn}: lora_specs {set(w._lora_specs)} ≠ yaml-declared {expected_specs}"
        )
        # Spot-check tensor rank: 2-D for shared (one matrix per layer), 3-D for
        # independent (leading expert dim). Catches accidental cross-mode
        # regressions early without poking into ParameterDict guts.
        a_w = w.get_lora_A_weight(next(iter(expected_specs)))
        assert a_w.ndim == expected_lora_ndim, (
            f"{fqn}/{mode}: expected lora_A ndim={expected_lora_ndim}, got {a_w.ndim} (shape={tuple(a_w.shape)})"
        )


@pytest.mark.parametrize("mode", _MODE_CASES)
@pytest.mark.parametrize("toy_dir,expected_n_layers", _MODEL_CASES)
def test_eager_forward_no_op_at_init(toy_dir: str, expected_n_layers: int, mode: str):
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
@pytest.mark.parametrize("toy_dir,expected_n_layers", _MODEL_CASES)
def test_backward_isolates_to_lora_params(toy_dir: str, expected_n_layers: int, mode: str):
    """Backward through a wrapped experts module must only fill grads for ``lora_A_*`` / ``lora_B_*``."""
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

    n_lora_with_grad = 0
    n_base_with_grad = 0
    for n, p in wrapper.named_parameters():
        if p.grad is None or p.grad.abs().sum().item() == 0:
            continue
        if n.startswith("lora_A_") or n.startswith("lora_B_"):
            n_lora_with_grad += 1
        elif n.startswith("base_layer."):
            n_base_with_grad += 1
    # At init lora_B == 0 ⇒ dL/dlora_A = 0 (chain through B). So only lora_B
    # params gain grad. The fused experts layout has two targets
    # (gate_up + down), so we expect at least 2 lora_B params with non-zero
    # grad in both modes.
    assert n_base_with_grad == 0, f"{toy_dir}/{mode}: base layer must stay frozen, got {n_base_with_grad}"
    assert n_lora_with_grad >= 2, (
        f"{toy_dir}/{mode}: expected at least 2 lora_B params with grad, got {n_lora_with_grad}"
    )


# ---------------------------------------------------------------------------
# Whole-model save/reload round-trip — exercised for BOTH modes on Qwen3-MoE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", _MODE_CASES)
def test_save_reload_round_trip_qwen3_moe(tmp_path, mode: str):
    """End-to-end: PEFT (yaml-declared linears) + MoE-LoRA (yaml patterns) → save+sidecar → reload → identical fwd.

    Run for both ``shared`` (Mode 2) and ``independent`` (Mode 1) flavours so
    the parametrised PEFT round-trip + sidecar dispatch logic in
    :func:`veomni.utils.moe_lora.apply_moe_lora_from_sidecar` is covered for
    each one. Asserts:

    * Init delta vs base is < 1e-3 (kaiming-A / zero-B no-op modulo activation
      checkpointing rounding).
    * After perturbing ``lora_B_*`` (so the LoRA contribution is non-trivial),
      reloading from disk produces a *bit-identical* forward + state-dict for
      every saved LoRA tensor.
    """
    from peft import LoraConfig, PeftModel, get_peft_model

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_toy("qwen3_moe_toy")

    lora_cfg = load_lora_config("qwen3_moe_toy")
    rank, alpha = lora_cfg["rank"], lora_cfg["alpha"]
    patterns = lora_cfg["target_parameters"]
    linear_targets = lora_cfg["lora_modules"]

    model.eval()
    dev = next(model.parameters()).device
    input_ids = torch.randint(0, 1000, (1, 16), device=dev)
    with torch.no_grad():
        base_out = model(input_ids=input_ids).logits.clone()
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Wrap with the same hyper-params the user-facing yaml declares.
    peft_cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=linear_targets)
    wrapped_model = get_peft_model(model, peft_cfg)
    wrapped = _apply(
        mode,
        wrapped_model,
        target_parameter_patterns=patterns,
        r=rank,
        lora_alpha=alpha,
        freeze_base_model=False,  # PEFT already froze
    )
    assert len(wrapped) == 4

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

    # Save (PEFT writes adapter_config + adapter_model.safetensors; we add the sidecar).
    save_dir = str(tmp_path / "adapter")
    wrapped_model.save_pretrained(save_dir)
    sidecar_path = write_moe_lora_sidecar(wrapped_model, save_dir)
    assert sidecar_path is not None and os.path.isfile(sidecar_path)

    # Reload into a fresh model with the SAME base weights.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model2 = build_toy("qwen3_moe_toy")
    model2.load_state_dict(base_state)
    model2.eval()
    sidecar = read_moe_lora_sidecar(save_dir)
    assert sidecar is not None
    assert sidecar["mode"] == mode, f"sidecar mode={sidecar['mode']!r}, expected {mode!r}"
    apply_moe_lora_from_sidecar(model2, sidecar, freeze_base_model=False)
    reloaded = PeftModel.from_pretrained(model2, save_dir, is_trainable=True)
    reloaded.eval()
    with torch.no_grad():
        reload_out = reloaded(input_ids=input_ids).logits

    # Bit-identical forward output.
    delta = (reload_out - trained_out).abs().max().item()
    assert delta == 0.0, f"{mode}: reload parity broken: max|reload-trained|={delta}"

    # Bit-identical LoRA parameter tensors.
    s1, s2 = wrapped_model.state_dict(), reloaded.state_dict()
    for k in s1:
        if "lora_A_" in k or "lora_B_" in k or "lora_A.default" in k or "lora_B.default" in k:
            assert k in s2, f"{mode}: missing in reload: {k}"
            assert torch.equal(s1[k], s2[k]), f"{mode}: value mismatch: {k}"
