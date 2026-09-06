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

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils import logging
from ..utils.env import get_env
from .config.singleton import set_ops_config
from .dispatch import OpSlot


if TYPE_CHECKING:
    from ..arguments.arguments_types import OpsImplementationConfig

__all__ = ["OpSlot"]

logger = logging.get_logger(__name__)


def apply_ops_patch():
    """Leftover import-time hook. Attention now installs via ``apply_kernel_patch``."""
    from ..kernels import apply_kernel_patch

    apply_kernel_patch()


def apply_ops_config(ops_config: OpsImplementationConfig) -> None:
    """Apply kernel patches based on resolved ``OpsImplementationConfig``.

    Single install point for config-driven dispatch:

    1. Binds the cross-entropy kernel into ``LOSS_MAPPING`` via
       ``install_loss_mapping`` (pre-bound ``partial`` — no runtime resolution).
    2. Populates the ops-config singleton so per-model ``device_patch.py`` and
       ``OpSlot.bind`` can read the user's selections.

    Per-model kernels are applied by each model's ``device_patch.py``.
    """
    set_ops_config(ops_config)

    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend == "hf":
        return

    from .kernels.cross_entropy import install_loss_mapping

    ce_label = install_loss_mapping(ops_config.cross_entropy_loss_implementation)
    logger.info_rank0(f"✅ VeOmni ops config applied: {ce_label}.")
    logger.info_rank0(format_kernel_functions())


def format_kernel_functions() -> str:
    lines = []
    lines.append("\n=========== OPS ============")

    # Cross-entropy is bound via LOSS_MAPPING (partial-wrapped), not a module
    # global — surface it here so the log still shows the active CE kernel.
    lines.append(f"cross_entropy = {_current_cross_entropy_name()}")

    lines.append("==============================")
    return "\n".join(lines)


def _current_cross_entropy_name() -> str:
    from functools import partial

    from transformers.loss.loss_utils import LOSS_MAPPING

    entry = LOSS_MAPPING.get("ForCausalLM")
    if entry is None:
        return "unset"
    if isinstance(entry, partial):
        ce_fn = entry.keywords.get("cross_entropy_fn")
        return getattr(ce_fn, "__name__", repr(ce_fn)) if ce_fn is not None else "unset"
    return getattr(entry, "__name__", repr(entry))
