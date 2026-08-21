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

"""VeOmni runtime helpers over wrapped :class:`~veomni.models.seed_omni.modeling_omni.OmniModel` modules."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch.nn as nn

from ..configuration_omni import OmniConfig
from ..mixins.base_mixin import BaseMixin
from .dispatch import unwrap_module_chain


def save_module_assets(module: nn.Module, module_dir: str) -> None:
    """Save a wrapped module's config plus processor / tokenizer sidecars (no weights).

    Unwraps DDP / LoRA wrappers first so ``config`` and assets resolve regardless
    of ``dp_mode``. FSDP2 composes in place and needs no special case here.
    """
    module = unwrap_module_chain(module)
    cfg = getattr(module, "config", None)
    if cfg is not None and hasattr(cfg, "save_pretrained"):
        cfg.save_pretrained(module_dir)
    for attr in ("_processor", "_image_processor", "_video_processor", "_tokenizer"):
        asset = getattr(module, attr, None)
        if asset is not None and hasattr(asset, "save_pretrained"):
            asset.save_pretrained(module_dir)


def save_module_subdirectory(
    config: OmniConfig,
    name: str,
    module: nn.Module,
    save_directory: str,
    *,
    save_module_weights: bool,
    **kwargs: Any,
) -> None:
    """Write one module subfolder under an omni checkpoint root (runtime path)."""
    subfolder = config.module_checkpoint_subfolder(name)
    module_dir = os.path.join(save_directory, subfolder)
    os.makedirs(module_dir, exist_ok=True)
    save_module_assets(module, module_dir)
    if save_module_weights:
        if not hasattr(module, "save_pretrained"):
            raise TypeError(
                f"OmniModelRuntime.save_pretrained: sub-module '{name}' ({type(module).__name__}) "
                "has no save_pretrained()."
            )
        module.save_pretrained(module_dir, **kwargs)


def iter_named_omni_modules(
    module_names: Iterable[str],
    modules: Mapping[str, nn.Module],
) -> Iterator[tuple[str, BaseMixin]]:
    """Yield ``(name, raw BaseMixin)`` for every graph participant behind wrappers."""
    for name in module_names:
        wrapped = modules.get(name)
        if wrapped is None:
            continue
        raw = unwrap_module_chain(wrapped)
        if isinstance(raw, BaseMixin):
            yield name, raw


__all__ = [
    "iter_named_omni_modules",
    "save_module_assets",
    "save_module_subdirectory",
]
