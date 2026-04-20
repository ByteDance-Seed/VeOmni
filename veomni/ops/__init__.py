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

# Eagerly import kernel packages so that every op registers itself with the
# registry.  Order does not matter; each ``register_op`` call is idempotent.
from . import kernels, liger  # noqa: F401  triggers all register_op() calls
from .config.registry import apply_global_ops
from .config.singleton import set_ops_config
from .dispatch import OpSlot
from .kernels import attention, cross_entropy, load_balancing_loss, moe  # noqa: F401
from .kernels.load_balancing_loss import load_balancing_loss_func
from .kernels.moe import fused_moe_forward


if TYPE_CHECKING:
    from ..arguments.arguments_types import OpsImplementationConfig

__all__ = [
    "fused_moe_forward",
    "OpSlot",
    "load_balancing_loss_func",
]

logger = logging.get_logger(__name__)


def build_ALL_OPS():
    return [
        ("_fused_moe_forward", moe._fused_moe_forward),
        ("_flash_attention_forward", attention._flash_attention_forward),
        ("_cross_entropy", cross_entropy._cross_entropy),
        ("_load_balancing_loss", load_balancing_loss._load_balancing_loss),
    ]


def apply_ops_patch():
    """Import-time ops patch.

    Registers attention implementations and installs VeOmni's loss wrappers in
    HuggingFace's ``LOSS_MAPPING``.  The loss install MUST happen at import
    time (not only inside ``apply_ops_config``) because several code paths
    build models directly via ``build_foundation_model`` without going through
    ``BaseTrainer`` (e.g. unit tests, ad-hoc scripts).  VeOmni modeling code
    calls ``self.loss_function(hidden_states=..., logits=None, ...)`` which
    HF's stock ``ForCausalLMLoss`` cannot handle.

    Load-balancing loss and per-model kernel patches remain deferred to
    ``apply_ops_config()`` / each model's ``device_patch.py``.
    """
    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend == "hf":
        logger.info_rank0("⚠️ Skip applying ops patch. Using huggingface transformers backend.")
    else:
        from .kernels.attention import apply_veomni_attention_patch
        from .kernels.cross_entropy import install_loss_mapping

        apply_veomni_attention_patch()
        install_loss_mapping()
        logger.info_rank0("✅ VeOmni attention + loss patches applied.")


def apply_ops_config(ops_config: OpsImplementationConfig) -> None:
    """Apply kernel patches based on resolved ``OpsImplementationConfig``.

    Walks all registered GLOBAL ops (cross-entropy loss, load-balancing loss)
    and binds the selected backend to each op's ``global_slot``.  Per-model
    patches are applied separately by each model's ``device_patch.py`` via
    ``apply_per_model_patches``.
    """
    set_ops_config(ops_config)

    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend == "hf":
        return

    # Re-install VeOmni's loss wrappers before walking the registry (already
    # installed at import time by ``apply_ops_patch``; this is idempotent and
    # makes the ordering explicit): the NPU cross-entropy backend's
    # side-effect then overrides LOSS_MAPPING["ForCausalLM"] to the
    # chunked-loss variant.
    from .kernels.cross_entropy import install_loss_mapping

    install_loss_mapping()

    applied = apply_global_ops(ops_config)
    # NOTE: fused MoE patch is applied in build_foundation_model() based on
    # the moe_implementation parameter; per-model kernels are applied by each
    # model's device_patch.py.
    logger.info_rank0(f"✅ VeOmni ops config applied: {', '.join(applied) if applied else '(defaults only)'}.")
    logger.info_rank0(format_kernel_functions())


def format_kernel_functions() -> str:
    lines = []
    lines.append("\n=========== OPS ============")

    for alias, func in build_ALL_OPS():
        impl = func.__name__ if func is not None else "None"
        lines.append(f"{alias} = {impl}")

    lines.append("==============================")
    return "\n".join(lines)
