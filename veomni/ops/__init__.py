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
from . import flash_attn, fused_cross_entropy, fused_load_balancing_loss, fused_moe
from .fused_load_balancing_loss import load_balancing_loss_func
from .fused_moe import fused_moe_forward
from .ops_config import set_ops_config


if TYPE_CHECKING:
    from ..arguments.arguments_types import OpsImplementationConfig

__all__ = [
    "fused_moe_forward",
    "load_balancing_loss_func",
]

logger = logging.get_logger(__name__)


def build_ALL_OPS():
    return [
        ("_fused_moe_forward", fused_moe._fused_moe_forward),
        ("_flash_attention_forward", flash_attn._flash_attention_forward),
        ("_cross_entropy", fused_cross_entropy._cross_entropy),
        ("_load_balancing_loss", fused_load_balancing_loss._load_balancing_loss),
    ]


def apply_ops_patch():
    """Import-time ops patch: only registers attention implementations.

    Loss, load-balancing, and MoE patches are deferred to
    ``apply_ops_config()`` which is called after the YAML config is parsed
    and ``OpsImplementationConfig`` is available.
    """
    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend == "hf":
        logger.info_rank0("⚠️ Skip applying ops patch. Using huggingface transformers backend.")
    else:
        from .flash_attn import apply_veomni_attention_patch

        apply_veomni_attention_patch()
        logger.info_rank0("✅ VeOmni attention patch applied.")


def apply_ops_config(ops_config: OpsImplementationConfig) -> None:
    """Apply loss / load-balancing / model patches based on resolved config.

    This must be called after ``OpsImplementationConfig`` is constructed
    (i.e. after YAML parsing) and before ``build_foundation_model()``.
    """
    set_ops_config(ops_config)

    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend == "hf":
        return

    from .fused_cross_entropy import apply_veomni_loss_patch
    from .fused_load_balancing_loss import apply_veomni_load_balancing_loss_patch

    apply_veomni_loss_patch(
        cross_entropy_loss_implementation=ops_config.cross_entropy_loss_implementation,
    )
    apply_veomni_load_balancing_loss_patch(
        load_balancing_loss_implementation=ops_config.load_balancing_loss_implementation,
    )
    # NOTE: fused MoE patch is applied in build_foundation_model() based on
    # the moe_implementation parameter.
    logger.info_rank0("✅ VeOmni ops config applied.")
    logger.info_rank0(format_kernel_functions())


def format_kernel_functions() -> str:
    lines = []
    lines.append("\n=========== OPS ============")

    for alias, func in build_ALL_OPS():
        impl = func.__name__ if func is not None else "None"
        lines.append(f"{alias} = {impl}")

    lines.append("==============================")
    return "\n".join(lines)
