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
from collections.abc import Sequence
from typing import Any

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def save_module_subdirectory(
    name: str,
    module: nn.Module,
    save_directory: str,
    *,
    assets: Sequence[Any],
    save_module_weights: bool,
    **kwargs: Any,
) -> None:
    """Write one module subfolder under an omni checkpoint root (runtime path).

    ``assets`` is the module's config + processor / tokenizer sidecars
    (:attr:`ModuleRuntime.model_assets`), written whether or not weights are.

    The subfolder is the module's name, matching
    :meth:`OmniModel._save_module_subdirectory` — an entry's ``model_path`` says
    where a module was *loaded* from, which may be another checkpoint entirely,
    and must not decide where this one writes.
    """
    from ....module_utils import save_model_assets

    module_dir = os.path.join(save_directory, name)
    os.makedirs(module_dir, exist_ok=True)
    save_model_assets(module_dir, assets)
    if isinstance(module, DistributedDataParallel):
        # DDP does not forward ``save_pretrained``. Strip only DDP: a LoRA
        # wrapper's own ``save_pretrained`` is what writes the adapter.
        module = module.module
    if save_module_weights:
        if not hasattr(module, "save_pretrained"):
            raise TypeError(
                f"OmniModelRuntime.save_pretrained: sub-module '{name}' ({type(module).__name__}) "
                "has no save_pretrained()."
            )
        module.save_pretrained(module_dir, **kwargs)


__all__ = [
    "save_module_subdirectory",
]
