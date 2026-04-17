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

"""Table-driven device patch engine.

Each model declares a list of ``OpsPatch`` descriptors that map
``OpsImplementationConfig`` field values to concrete kernel replacements.
The shared ``apply_device_patches()`` function handles the dispatch,
lazy import, attribute replacement, and logging.

For model-specific logic that cannot be expressed as a simple
``(module, attr, setattr)`` triple (e.g. DeepSeek V3 Triton RoPE),
an optional ``custom_patches`` callback is invoked after the table
patches are applied.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Callable

from ..utils import logging
from .ops_config import get_ops_config


if TYPE_CHECKING:
    from ..arguments.arguments_types import OpsImplementationConfig

logger = logging.get_logger(__name__)


@dataclass
class ImplSpec:
    """One concrete kernel implementation.

    Attributes:
        module: Fully-qualified module path for lazy import.
        attr: Attribute name to fetch from the imported module.
        replace_forward: If ``True``, replace ``target.forward`` instead of
            replacing the target attribute on the parent module entirely.
    """

    module: str
    attr: str
    replace_forward: bool = False


@dataclass
class OpsPatch:
    """Declarative patch descriptor for one op on one model.

    For well-known ops, prefer the factory helpers ``rope_patch``,
    ``rms_norm_patch``, and ``swiglu_patch`` which fill in ``config_field``
    and ``op_label`` automatically.  Use this class directly only for
    custom / non-standard ops.

    Attributes:
        config_field: ``OpsImplementationConfig`` field name to read.
        op_label: Human-readable label for log messages (e.g. ``"RMSNorm"``).
        target_attr: Attribute name on the HF module to patch.
        impls: Mapping from implementation value to ``ImplSpec``.
    """

    config_field: str
    op_label: str
    target_attr: str
    impls: dict[str, ImplSpec] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Factory helpers for well-known ops
# ---------------------------------------------------------------------------


def rope_patch(target_attr: str, impls: dict[str, ImplSpec]) -> OpsPatch:
    """Create an ``OpsPatch`` for ``rotary_pos_emb_implementation``."""
    return OpsPatch("rotary_pos_emb_implementation", "RoPE", target_attr, impls)


def rms_norm_patch(target_attr: str, impls: dict[str, ImplSpec]) -> OpsPatch:
    """Create an ``OpsPatch`` for ``rms_norm_implementation``."""
    return OpsPatch("rms_norm_implementation", "RMSNorm", target_attr, impls)


def swiglu_patch(target_attr: str, impls: dict[str, ImplSpec]) -> OpsPatch:
    """Create an ``OpsPatch`` for ``swiglu_mlp_implementation``."""
    return OpsPatch("swiglu_mlp_implementation", "SwiGLU", target_attr, impls)


def apply_device_patches(
    hf_module: ModuleType,
    patches: list[OpsPatch],
    model_name: str,
    *,
    custom_patches: Callable[[OpsImplementationConfig, list[str]], None] | None = None,
) -> None:
    """Apply kernel patches to *hf_module* based on ``OpsImplementationConfig``.

    Args:
        hf_module: The HuggingFace (or VeOmni) modeling module to patch.
        patches: List of ``OpsPatch`` descriptors.
        model_name: Display name used in log messages.
        custom_patches: Optional callback for model-specific logic that
            cannot be expressed as a table entry.  Receives
            ``(ops_config, applied_list)`` so it can append to the shared
            log line.
    """
    ops_config = get_ops_config()
    if ops_config is None:
        return

    applied: list[str] = []

    for patch in patches:
        impl_value = getattr(ops_config, patch.config_field)
        spec = patch.impls.get(impl_value)
        if spec is None:
            continue
        mod = importlib.import_module(spec.module)
        replacement = getattr(mod, spec.attr)
        if spec.replace_forward:
            target = getattr(hf_module, patch.target_attr)
            target.forward = replacement
        else:
            setattr(hf_module, patch.target_attr, replacement)
        applied.append(f"{patch.op_label} ({impl_value})")

    if custom_patches is not None:
        custom_patches(ops_config, applied)

    if applied:
        logger.info_rank0(f"Apply ops patches to {model_name}: {', '.join(applied)}.")
