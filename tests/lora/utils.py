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
"""Shared helpers for VeOmni LoRA tests.

Conventions:
    * Toy models live under ``tests/toy_config/<toy_dir>/`` and are paired
      with a user-facing ``configs/.../<model>_lora.yaml``. The yaml is the
      source of truth for ``target_parameters`` (fused experts layout —
      ``gate_up_proj`` / ``down_proj``), ``lora_modules`` (PEFT linear
      targets), ``rank`` and ``alpha``.
    * VeOmni MoE-LoRA is v5-only (the wrapper validates a fused experts
      layout in :func:`veomni.lora.moe_layers._validate_fused_layout`).
      Toys whose model family was added in a later transformers release —
      e.g. ``qwen3_5_moe`` (5.2.0) — declare ``min_transformers_version``
      so :func:`select_lora_yaml` skips cleanly on older envs.
    * Tests load their patterns from the yaml so a stale yaml fails the
      suite loudly via ``apply_shared_moe_lora``'s ``fail_on_no_match``
      (default).
    * Build runs on CUDA when available (~0.7 s for a 1B-param toy) and
      falls back to CPU otherwise (~30 s/build, slow but functional).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import transformers
import yaml

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models import build_foundation_model
from veomni.utils.device import get_device_type
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOY_CONFIG_ROOT = os.path.join(REPO_ROOT, "tests", "toy_config")

# Accelerator build of a 1B-param toy is ~0.7 s vs ~30 s on CPU. Use the
# active accelerator (CUDA / NPU) via ``get_device_type`` when present, else
# fall back to CPU. ``get_device_type`` already returns ``"cpu"`` when no
# accelerator is available.
DEVICE = torch.device(get_device_type())


# ---------------------------------------------------------------------------
# Toy → lora.yaml mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoraYamlSpec:
    """How to find the matching lora.yaml for a toy.

    Args:
        yaml: Path (relative to repo root) of the LoRA fine-tuning yaml.
        min_transformers_version: Minimum ``transformers`` version where
            this model family is registered (mirrors the gate in
            ``veomni/models/transformers/<model>/__init__.py``). Tests skip
            on older envs rather than failing during model build.
    """

    yaml: str
    min_transformers_version: str


TOY_LORA_SPECS: dict[str, LoraYamlSpec] = {
    "qwen3_moe_toy": LoraYamlSpec(
        yaml="configs/text/qwen3_moe_lora.yaml",
        min_transformers_version="5.2.0",
    ),
    "qwen3_5_moe_toy": LoraYamlSpec(
        yaml="configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_lora.yaml",
        min_transformers_version="5.2.0",
    ),
    "qwen3vlmoe_toy": LoraYamlSpec(
        yaml="configs/multimodal/qwen3_vl/qwen3_vl_moe_lora.yaml",
        min_transformers_version="5.2.0",
    ),
    "qwen3omni_toy": LoraYamlSpec(
        yaml="configs/multimodal/qwen3_omni/qwen3_omni_lora.yaml",
        min_transformers_version="5.2.0",
    ),
    "deepseek_v3_toy": LoraYamlSpec(
        yaml="configs/text/deepseek_v3_lora.yaml",
        min_transformers_version="5.2.0",
    ),
}


# ---------------------------------------------------------------------------
# Model build
# ---------------------------------------------------------------------------


def full_eager_ops() -> OpsImplementationConfig:
    """Force every operator implementation to ``eager`` so wrapper code paths are exercised."""
    return OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
    )


def fused_triton_moe_ops() -> OpsImplementationConfig:
    """Eager everywhere *except* MoE, which uses the Triton group-gemm backend.

    Selecting ``moe_implementation="fused_triton"`` triggers
    ``apply_veomni_fused_moe_patch("triton")`` during ``build_foundation_model``,
    which is what installs ``veomni.lora.ops._fused_lora_moe_forward``.
    The fused MoE-LoRA tests need that pointer to be non-``None`` to actually
    exercise the kernel path inside ``LoraSharedExperts.forward``.
    """
    return OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="fused_triton",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
    )


def build_toy(toy_dir: str, *, ops: OpsImplementationConfig | None = None):
    """Build a bf16 toy model from ``tests/toy_config/<toy_dir>/`` on the active device.

    Args:
        toy_dir: Subdir under ``tests/toy_config/`` (e.g. ``"qwen3_moe_toy"``).
        ops: Optional ops backend selection. Defaults to :func:`full_eager_ops`
            so eager wrapper code paths are exercised; pass
            :func:`fused_triton_moe_ops` for kernel-path tests.

    Skips the calling test when the toy config dir is missing.
    """
    cfg_path = os.path.join(TOY_CONFIG_ROOT, toy_dir)
    if not os.path.isfile(os.path.join(cfg_path, "config.json")):
        pytest.skip(f"toy config not found: {cfg_path}")
    return build_foundation_model(
        config_path=cfg_path,
        weights_path=None,
        torch_dtype="bfloat16",
        init_device=DEVICE.type,
        ops_implementation=ops if ops is not None else full_eager_ops(),
    )


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------


def select_lora_yaml(toy_dir: str) -> str:
    """Return the absolute yaml path for ``toy_dir``.

    Skips the calling test when the installed ``transformers`` version is
    older than ``spec.min_transformers_version`` (i.e. the model family
    isn't registered there yet — VeOmni MoE-LoRA is v5-only).
    """
    if toy_dir not in TOY_LORA_SPECS:
        pytest.skip(f"no lora.yaml registered for toy {toy_dir!r}")
    spec = TOY_LORA_SPECS[toy_dir]
    if not is_transformers_version_greater_or_equal_to(spec.min_transformers_version):
        pytest.skip(
            f"{toy_dir}: requires transformers >= {spec.min_transformers_version}; got {transformers.__version__}."
        )
    abs_path = os.path.join(REPO_ROOT, spec.yaml)
    if not os.path.isfile(abs_path):
        pytest.skip(f"lora.yaml not found: {abs_path}")
    return abs_path


def load_lora_config(toy_dir: str) -> dict[str, Any]:
    """Return the ``model.lora_config`` block from the toy's ``lora.yaml``.

    The yaml is selected via :func:`select_lora_yaml`, which gates on the
    installed ``transformers`` version. Yamls are the source of truth for
    ``target_parameters``, ``lora_modules``, ``rank`` and ``alpha``.
    """
    yaml_path = select_lora_yaml(toy_dir)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["model"]["lora_config"]


# ---------------------------------------------------------------------------
# Glob / FQN utilities
# ---------------------------------------------------------------------------


def experts_module_globs(target_parameter_patterns: list[str]) -> list[str]:
    """Strip the trailing parameter name from each pattern → glob over experts modules.

    e.g. ``model.layers.*.mlp.experts.gate_up_proj`` → ``model.layers.*.mlp.experts``.
    Multiple patterns covering the same module collapse to a single glob.
    """
    return sorted({p.rsplit(".", 1)[0] for p in target_parameter_patterns})


def glob_to_regex(glob: str) -> re.Pattern:
    """Translate a PEFT-style glob (``*`` matches one FQN segment) to a regex."""
    return re.compile("^" + re.escape(glob).replace(r"\*", r"[^.]+") + "$")


def find_first_matching_module(model: torch.nn.Module, module_globs: list[str]) -> tuple[str, torch.nn.Module]:
    """Return ``(fqn, module)`` for the first module whose FQN matches any glob.

    Raises a verbose ``AssertionError`` on miss; useful for catching yaml drift
    against the toy modeling code.
    """
    regs = [glob_to_regex(g) for g in module_globs]
    for fqn, mod in model.named_modules():
        if any(r.match(fqn) for r in regs):
            return fqn, mod
    raise AssertionError(
        f"no module in {type(model).__name__} matched any of {module_globs!r} — "
        "is the paired lora.yaml stale w.r.t. the toy modeling code?"
    )


def find_all_matching_modules(model: torch.nn.Module, module_globs: list[str]) -> list[str]:
    """Return every FQN in ``model`` matching any of ``module_globs``, sorted.

    Used by tests to derive the *expected* count of experts modules to wrap
    straight from the built model rather than hardcoding a per-toy number —
    so adding/removing MoE layers in a toy config doesn't silently desync
    from the test's expected count.
    """
    regs = [glob_to_regex(g) for g in module_globs]
    return sorted(fqn for fqn, _ in model.named_modules() if any(r.match(fqn) for r in regs))
