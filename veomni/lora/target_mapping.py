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

"""Map semantic LoRA module names to fused MoE expert parameter patterns.

Some models (the Qwen3-MoE family) store expert weights as fused 3-D
``nn.Parameter`` s (``gate_up_proj`` / ``down_proj``) rather than one
``nn.Linear`` per expert. Users still describe LoRA targets with the familiar
semantic module names (``gate_proj`` / ``up_proj`` / ``down_proj``) in
``lora_config['lora_modules']``. Each fused model registers a
``_convert_lora_targets_to_parameters`` staticmethod (see the model
``__init__.py``) that rewrites those names into the physical fused-parameter
glob patterns consumed by :func:`veomni.lora.moe_layers.inject_moe_lora`.

This is the native (PEFT-free) counterpart of the mapping that used to live in
``veomni.utils.lora_utils``: the model hook produces glob patterns that flow
straight into ``VeOmniLoraConfig.target_parameters`` — the native MoE injector
matches those patterns directly, so no expansion to concrete parameter names is
required (unlike the PEFT ``target_parameters`` path).
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

import torch.nn as nn

from ..utils import logging


logger = logging.get_logger(__name__)

# Semantic MLP module names that a fused-MoE model owns as bare parameters
# rather than ``nn.Linear`` submodules.
_FUSED_GATE_UP = {"gate_proj", "up_proj"}
_FUSED_DOWN = "down_proj"


def convert_fused_moe_lora_targets(
    lora_modules: list[str],
    target_parameter_patterns: list[str],
    gate_up_proj_pattern: str,
    down_proj_pattern: str,
) -> tuple[list[str], list[str]]:
    """Rewrite semantic MLP LoRA module names into fused MoE expert patterns.

    ``gate_proj`` / ``up_proj`` collapse to ``gate_up_proj_pattern`` (the fused
    gate+up parameter) and ``down_proj`` to ``down_proj_pattern``; both are
    removed from ``target_modules``. Any other module name (``q_proj``, router,
    ...) is left untouched. Explicit ``target_parameter_patterns`` are preserved.
    """
    target_modules = list(lora_modules)
    target_parameter_patterns = list(target_parameter_patterns)
    if _FUSED_GATE_UP & set(target_modules):
        target_parameter_patterns.append(gate_up_proj_pattern)
        target_modules = [name for name in target_modules if name not in _FUSED_GATE_UP]
    if _FUSED_DOWN in target_modules:
        target_parameter_patterns.append(down_proj_pattern)
        target_modules = [name for name in target_modules if name != _FUSED_DOWN]
    return target_modules, target_parameter_patterns


def _unwrap_model_for_lora_targets(model: nn.Module) -> nn.Module:
    """Return the raw base model, unwrapping a LoRA wrapper if present."""
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def _assert_patterns_match(model: nn.Module, patterns: list[str]) -> None:
    """Raise if a produced parameter glob matches no parameter of ``model``.

    Fused MoE expert patterns that silently match nothing would otherwise
    inject no LoRA at all, which is easy to miss. Parameter names are available
    even under ``init_device='meta'`` (only the storage is on ``meta``).
    """
    parameter_names = [name for name, _ in model.named_parameters()]
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            matched = any(fnmatch(name, pattern) for name in parameter_names)
        else:
            matched = any(name == pattern or name.endswith(f".{pattern}") for name in parameter_names)
        if not matched:
            raise ValueError(f"LoRA target parameter pattern did not match any parameter: {pattern}")


def resolve_fused_moe_lora_targets(model: nn.Module, lora_config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``lora_config`` with semantic MoE names mapped to params.

    Models that fuse their experts register a ``_convert_lora_targets_to_parameters``
    hook translating ``gate_proj`` / ``up_proj`` / ``down_proj`` (and leaving
    everything else in place). Dense models — and fused models whose config lists
    no semantic MoE module — are returned unchanged, so ``gate_proj`` etc. stay as
    ordinary ``nn.Linear`` LoRA targets. The result feeds
    :meth:`VeOmniLoraConfig.from_yaml`, which derives ``moe_mode`` from the
    resulting ``target_parameters`` and ``share_expert_lora``.
    """
    resolved = dict(lora_config)
    base = _unwrap_model_for_lora_targets(model)

    converter = getattr(base, "_convert_lora_targets_to_parameters", None)
    if converter is None:
        return resolved
    if not callable(converter):
        logger.warning_rank0("Ignoring invalid `_convert_lora_targets_to_parameters`: not callable.")
        return resolved

    lora_modules = resolved.get("lora_modules")
    if lora_modules is None:
        lora_modules = resolved.get("target_modules")
    # A regex string target spec is opaque to name-based mapping; leave it as-is.
    if not lora_modules or isinstance(lora_modules, str):
        return resolved

    explicit_parameters = list(resolved.get("target_parameters") or [])
    new_modules, new_parameters = converter(base, list(lora_modules), explicit_parameters)
    if new_parameters == explicit_parameters:
        # No semantic MoE name was present; nothing to rewrite.
        return resolved

    _assert_patterns_match(base, new_parameters)

    resolved["lora_modules"] = new_modules
    if "target_modules" in resolved:
        resolved["target_modules"] = new_modules
    resolved["target_parameters"] = new_parameters
    return resolved
