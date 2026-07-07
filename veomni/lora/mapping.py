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

"""Target matching + in-place LoRA injection for the VeOmni-native stack.

Given a base model and a :class:`~veomni.lora.config.VeOmniLoraConfig`, walk the
module tree once and replace matched ``nn.Linear`` layers with
:class:`~veomni.lora.layers.LoraLinear`. Matching follows PEFT semantics: a
module matches when its FQN equals a target or ends with ``.<target>`` (list
form), or fully matches the regex (string form); ``exclude_modules`` prunes.
Per-module ``r`` / ``alpha`` overrides come from ``rank_pattern`` /
``alpha_pattern``.

MoE ``target_parameters`` injection (the 3-D expert wrappers) is handled
separately in :mod:`veomni.lora.moe_layers`; this module only covers the dense
``nn.Linear`` path.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch.nn as nn

from ..utils import logging
from .layers import LoraLinear


if TYPE_CHECKING:
    from .config import VeOmniLoraConfig


logger = logging.get_logger(__name__)


def _module_matches(fqn: str, spec: list[str] | str | None) -> bool:
    """PEFT-style membership test of ``fqn`` against a target/exclude spec."""
    if not spec:
        return False
    if isinstance(spec, str):
        return re.fullmatch(spec, fqn) is not None
    if fqn in spec:
        return True
    return any(fqn.endswith(f".{t}") for t in spec)


def find_target_linear_names(model: nn.Module, config: VeOmniLoraConfig) -> list[str]:
    """Return FQNs of ``nn.Linear`` modules that should receive dense LoRA."""
    names: list[str] = []
    for fqn, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _module_matches(fqn, config.target_modules):
            continue
        if _module_matches(fqn, config.exclude_modules):
            continue
        names.append(fqn)
    return names


def _replace_module(root: nn.Module, fqn: str, new_module: nn.Module) -> None:
    """Set ``root.<fqn> = new_module`` via its immediate parent."""
    parent_fqn, _, attr = fqn.rpartition(".")
    parent = root.get_submodule(parent_fqn) if parent_fqn else root
    setattr(parent, attr, new_module)


def inject_dense_lora(
    model: nn.Module,
    config: VeOmniLoraConfig,
    adapter_name: str,
) -> list[str]:
    """Replace matched ``nn.Linear`` layers in ``model`` with ``LoraLinear``.

    ``model`` is the *inner* base model (``VeOmniLoraModel.base_model.model``);
    the returned FQNs are relative to it. Does not touch ``requires_grad`` —
    the caller (:class:`~veomni.lora.model.VeOmniLoraModel`) runs the freeze /
    unfreeze pass after all injection (dense + MoE) is done.
    """
    if not config.target_modules:
        return []

    target_names = find_target_linear_names(model, config)
    if not target_names:
        logger.warning_rank0(
            f"No nn.Linear modules matched target_modules={config.target_modules!r}; dense LoRA injected nothing."
        )
        return []

    wrapped: list[str] = []
    for fqn in target_names:
        base_linear: nn.Linear = model.get_submodule(fqn)
        lora_linear = LoraLinear(
            base_layer=base_linear,
            adapter_name=adapter_name,
            r=config.rank_for(fqn),
            lora_alpha=config.alpha_for(fqn),
            lora_dropout=config.lora_dropout,
            use_rslora=config.use_rslora,
            init_lora_weights=config.init_lora_weights,
        )
        _replace_module(model, fqn, lora_linear)
        wrapped.append(fqn)

    logger.info_rank0(f"Injected dense LoRA into {len(wrapped)} nn.Linear module(s) (showing first 5): {wrapped[:5]}")
    return wrapped


def mark_lora_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze everything, then unfreeze LoRA params (and biases per ``bias``).

    ``bias`` follows PEFT semantics:
      * ``none``      — only ``lora_A`` / ``lora_B`` are trainable.
      * ``all``       — every bias in the model stays trainable.
      * ``lora_only`` — biases of modules that carry LoRA stay trainable.
    """
    for p in model.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if ".lora_A." in n or ".lora_B." in n or n.startswith("lora_A.") or n.startswith("lora_B."):
            p.requires_grad = True

    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if n.endswith("bias") or ".bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for module in model.modules():
            if isinstance(module, LoraLinear) and getattr(module.base_layer, "bias", None) is not None:
                module.base_layer.bias.requires_grad = True
    else:
        raise ValueError(f"Unknown bias policy {bias!r}.")
