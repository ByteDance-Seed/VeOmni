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
"""Smoke tests for VeOmni's shared-experts MoE LoRA wrapper (eager, single-process, bf16).

What this exercises:

1. ``LoraSharedExperts`` autodetects the experts layout on every Qwen3-family
   MoE model — v5 fused (``gate_up_proj`` / ``down_proj``) or v4 split
   (``gate_proj`` / ``up_proj`` / ``down_proj``) — and matches the yaml-declared
   ``target_parameters`` for the installed transformers version.
2. The wrapper is a true no-op at init (kaiming-uniform A, zero B).
3. Backward only flows through ``lora_A_*`` / ``lora_B_*`` parameters; the
   base experts module is fully frozen.
4. End-to-end save/reload round-trip via PEFT + the
   ``veomni_share_lora.json`` sidecar produces a model whose forward output
   is bit-identical to the in-memory trained model.

Transformers version routing:
    Each toy is mapped to one yaml per transformers branch
    (``configs/.../<model>_lora_v4.yaml`` and ``..._v5.yaml``) via
    ``tests/lora/utils.py::TOY_LORA_SPECS``. ``load_lora_config`` picks the
    yaml matching the live ``transformers.__version__`` (cutoffs mirror each
    model's ``__init__.py``). v5-only families (e.g. ``qwen3_5_moe``, added
    in transformers 5.2.0) skip when run on an older transformers.

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
    LoraSharedExperts,
    apply_shared_moe_lora,
    apply_shared_moe_lora_from_sidecar,
    read_shared_lora_sidecar,
    write_shared_lora_sidecar,
)

from .utils import (
    build_toy,
    experts_module_globs,
    find_first_matching_module,
    load_lora_config,
    transformers_branch,
)


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


@pytest.mark.parametrize("toy_dir,expected_n_layers", _MODEL_CASES)
def test_layout_autodetect_and_wrap(toy_dir: str, expected_n_layers: int):
    """Wrapping with the paired yaml's ``target_parameters`` must replace exactly the experts modules."""
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_toy(toy_dir)
    lora_cfg = load_lora_config(toy_dir)
    patterns = lora_cfg["target_parameters"]
    wrapped = apply_shared_moe_lora(
        model,
        target_parameter_patterns=patterns,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        freeze_base_model=True,
    )
    assert len(wrapped) == expected_n_layers, (
        f"{toy_dir}: expected {expected_n_layers} wrapped experts modules, got {len(wrapped)}: {wrapped}"
    )
    # The detected layout is dictated by the modeling code which the yaml is paired with.
    # Most v4 families use split (gate/up/down) and v5 uses fused (gate_up/down), but
    # qwen3_vl_moe v4 also uses fused — so derive the expected layout from the yaml's
    # trailing parameter names rather than from the transformers branch.
    expected_specs = {p.rsplit(".", 1)[1] for p in patterns}
    expected_layout = "fused_v5" if "gate_up_proj" in expected_specs else "split_v4"
    layouts = set()
    for fqn in wrapped:
        w = model.get_submodule(fqn)
        assert isinstance(w, LoraSharedExperts)
        layouts.add(w._layout)
        assert set(w._lora_specs) == expected_specs, (
            f"{fqn}: lora_specs {set(w._lora_specs)} ≠ yaml-declared {expected_specs}"
        )
    assert layouts == {expected_layout}, (
        f"{toy_dir} (transformers {transformers_branch(toy_dir)}): "
        f"expected layout {{{expected_layout!r}}}, got {layouts}"
    )


@pytest.mark.parametrize("toy_dir,expected_n_layers", _MODEL_CASES)
def test_eager_forward_no_op_at_init(toy_dir: str, expected_n_layers: int):
    """Direct experts.forward() with kaiming-A / zero-B must reproduce the base output exactly."""
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_toy(toy_dir)

    lora_cfg = load_lora_config(toy_dir)
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

    apply_shared_moe_lora(
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
    assert diff == 0.0, f"{toy_dir}: LoRA must be no-op at init, got max|delta|={diff}"


@pytest.mark.parametrize("toy_dir,expected_n_layers", _MODEL_CASES)
def test_backward_isolates_to_lora_params(toy_dir: str, expected_n_layers: int):
    """Backward through a wrapped experts module must only fill grads for ``lora_A_*`` / ``lora_B_*``."""
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_toy(toy_dir)

    lora_cfg = load_lora_config(toy_dir)
    patterns = lora_cfg["target_parameters"]
    sample_fqn, exp = find_first_matching_module(model, experts_module_globs(patterns))
    H, E = exp.hidden_dim, exp.num_experts
    p0 = next(exp.parameters())
    dtype, dev = p0.dtype, p0.device
    apply_shared_moe_lora(
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
    # At init lora_B == 0 ⇒ dL/dlora_A = 0 (chain through B). So only lora_B params gain grad.
    # Both targets have one lora_B each ⇒ expect exactly 2.
    assert n_base_with_grad == 0, f"{toy_dir}: base layer must stay frozen, got {n_base_with_grad}"
    assert n_lora_with_grad >= 2, f"{toy_dir}: expected at least 2 lora_B params with grad, got {n_lora_with_grad}"


# ---------------------------------------------------------------------------
# Whole-model save/reload round-trip
# ---------------------------------------------------------------------------


def test_save_reload_round_trip_qwen3_moe(tmp_path):
    """End-to-end: PEFT (yaml-declared linears) + shared-MoE-LoRA (yaml patterns) → save+sidecar → reload → identical fwd."""
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
    wrapped = apply_shared_moe_lora(
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
    assert delta_init < 1e-3, f"LoRA must be no-op at init, got {delta_init}"

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
    sidecar_path = write_shared_lora_sidecar(wrapped_model, save_dir)
    assert sidecar_path is not None and os.path.isfile(sidecar_path)

    # Reload into a fresh model with the SAME base weights.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model2 = build_toy("qwen3_moe_toy")
    model2.load_state_dict(base_state)
    model2.eval()
    sidecar = read_shared_lora_sidecar(save_dir)
    assert sidecar is not None
    apply_shared_moe_lora_from_sidecar(model2, sidecar, freeze_base_model=False)
    reloaded = PeftModel.from_pretrained(model2, save_dir, is_trainable=True)
    reloaded.eval()
    with torch.no_grad():
        reload_out = reloaded(input_ids=input_ids).logits

    # Bit-identical forward output.
    delta = (reload_out - trained_out).abs().max().item()
    assert delta == 0.0, f"reload parity broken: max|reload-trained|={delta}"

    # Bit-identical LoRA parameter tensors.
    s1, s2 = wrapped_model.state_dict(), reloaded.state_dict()
    for k in s1:
        if "lora_A_" in k or "lora_B_" in k or "lora_A.default" in k or "lora_B.default" in k:
            assert k in s2, f"missing in reload: {k}"
            assert torch.equal(s1[k], s2[k]), f"value mismatch: {k}"
