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
    * Toy models live under ``tests/toy_config/<toy_dir>/`` and are paired with
      a user-facing ``configs/.../<model>_lora{,_v4,_v5}.yaml``. The yaml is the
      source of truth for ``target_parameters`` (v4 split vs v5 fused layout),
      ``lora_modules`` (PEFT linear targets), ``rank`` and ``alpha``.
    * Models that exist on both transformers v4 and v5 (different expert tensor
      layouts) ship two yamls; ``load_lora_config`` picks the one matching the
      installed ``transformers`` version. v5-only models (e.g. ``qwen3_5_moe``,
      added in transformers 5.2.0) ship a single yaml.
    * Tests load their patterns from the yaml so a stale yaml fails the suite
      loudly via ``apply_shared_moe_lora``'s ``fail_on_no_match`` (default).
    * Build runs on CUDA when available (~0.7 s for a 1B-param toy) and falls
      back to CPU otherwise (~30 s/build, slow but functional).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import transformers
import yaml

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models import build_foundation_model
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOY_CONFIG_ROOT = os.path.join(REPO_ROOT, "tests", "toy_config")

# CUDA build of a 1B-param toy is ~0.7 s vs ~30 s on CPU. Use GPU when present.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Toy → lora.yaml mapping (transformers-version-aware)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoraYamlSpec:
    """How to find the matching lora.yaml for a toy across transformers versions.

    Args:
        v5_min_version: Minimum ``transformers`` version that selects the v5
            modeling path for this family (matches the cutoff in each model's
            ``veomni/models/transformers/<model>/__init__.py``).
        v5_yaml: Path (relative to repo root) of the v5-layout yaml. Required.
        v4_yaml: Path of the v4-layout yaml. ``None`` for v5-only families
            (e.g. ``qwen3_5_moe`` exists in transformers >= 5.2.0 only).
    """

    v5_min_version: str
    v5_yaml: str
    v4_yaml: Optional[str]


# Cutoffs mirror veomni/models/transformers/<model>/__init__.py::register_*_modeling
# (qwen3_moe: >=5.0.0 → v5; qwen3_vl_moe / qwen3_omni_moe / qwen3_5_moe: >=5.2.0 → v5).
TOY_LORA_SPECS: Dict[str, LoraYamlSpec] = {
    "qwen3_moe_toy": LoraYamlSpec(
        v5_min_version="5.0.0",
        v5_yaml="configs/text/qwen3_moe_lora_v5.yaml",
        v4_yaml="configs/text/qwen3_moe_lora_v4.yaml",
    ),
    "qwen3_5_moe_toy": LoraYamlSpec(
        v5_min_version="5.2.0",
        v5_yaml="configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_lora.yaml",
        v4_yaml=None,  # v5-only; qwen3_5_moe was added in transformers 5.2.0.
    ),
    "qwen3vlmoe_toy": LoraYamlSpec(
        v5_min_version="5.2.0",
        v5_yaml="configs/multimodal/qwen3_vl/qwen3_vl_moe_lora_v5.yaml",
        v4_yaml="configs/multimodal/qwen3_vl/qwen3_vl_moe_lora_v4.yaml",
    ),
    "qwen3omni_toy": LoraYamlSpec(
        v5_min_version="5.2.0",
        v5_yaml="configs/multimodal/qwen3_omni/qwen3_omni_lora_v5.yaml",
        v4_yaml="configs/multimodal/qwen3_omni/qwen3_omni_lora_v4.yaml",
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


def build_toy(toy_dir: str):
    """Build a bf16 toy model from ``tests/toy_config/<toy_dir>/`` on the active device.

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
        ops_implementation=full_eager_ops(),
    )


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------


def transformers_branch(toy_dir: str) -> str:
    """Return ``"v5"`` if installed transformers >= ``spec.v5_min_version`` else ``"v4"``.

    Used by tests both for selecting the right yaml and for human-readable
    parametrize ids / log messages.
    """
    spec = TOY_LORA_SPECS[toy_dir]
    return "v5" if is_transformers_version_greater_or_equal_to(spec.v5_min_version) else "v4"


def select_lora_yaml(toy_dir: str) -> Tuple[str, str]:
    """Return ``(absolute_path, "v4"|"v5")`` for the yaml matching the live transformers.

    Skips the calling test when the toy is registered as v5-only and the
    installed transformers version pre-dates ``v5_min_version``.
    """
    if toy_dir not in TOY_LORA_SPECS:
        pytest.skip(f"no lora.yaml registered for toy {toy_dir!r}")
    spec = TOY_LORA_SPECS[toy_dir]
    branch = transformers_branch(toy_dir)
    rel = spec.v5_yaml if branch == "v5" else spec.v4_yaml
    if rel is None:
        pytest.skip(
            f"{toy_dir}: requires transformers >= {spec.v5_min_version} "
            f"for v5 path; got {transformers.__version__} and no v4 yaml exists "
            "(family is v5-only — see veomni/models/transformers/<model>/__init__.py)."
        )
    abs_path = os.path.join(REPO_ROOT, rel)
    if not os.path.isfile(abs_path):
        pytest.skip(f"lora.yaml not found: {abs_path}")
    return abs_path, branch


def load_lora_config(toy_dir: str) -> Dict[str, Any]:
    """Return the ``model.lora_config`` block from the version-matched ``lora.yaml``.

    The yaml is selected via :func:`select_lora_yaml`, which honours the
    installed ``transformers`` version. Yamls are the source of truth for
    ``target_parameters`` (v4 vs v5 layout), ``lora_modules`` (PEFT linear
    targets), ``rank`` and ``alpha``.
    """
    yaml_path, _branch = select_lora_yaml(toy_dir)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["model"]["lora_config"]


# ---------------------------------------------------------------------------
# Glob / FQN utilities
# ---------------------------------------------------------------------------


def experts_module_globs(target_parameter_patterns: List[str]) -> List[str]:
    """Strip the trailing parameter name from each pattern → glob over experts modules.

    e.g. ``model.layers.*.mlp.experts.gate_up_proj`` → ``model.layers.*.mlp.experts``.
    Multiple patterns covering the same module collapse to a single glob.
    """
    return sorted({p.rsplit(".", 1)[0] for p in target_parameter_patterns})


def glob_to_regex(glob: str) -> re.Pattern:
    """Translate a PEFT-style glob (``*`` matches one FQN segment) to a regex."""
    return re.compile("^" + re.escape(glob).replace(r"\*", r"[^.]+") + "$")


def find_first_matching_module(model: torch.nn.Module, module_globs: List[str]) -> Tuple[str, torch.nn.Module]:
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
