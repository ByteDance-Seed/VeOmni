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
    "format_kernel_functions",
    "format_kernel_selection_summary",
]

logger = logging.get_logger(__name__)


def build_ALL_OPS():
    return [
        ("_fused_moe_forward", moe._fused_moe_forward),
        ("_flash_attention_forward", attention._flash_attention_forward),
        ("_load_balancing_loss", load_balancing_loss._load_balancing_loss),
    ]


def apply_ops_patch():
    """Import-time ops patch — attention only.

    Registers VeOmni's SP-aware attention variants into the shared
    ``ALL_ATTENTION_FUNCTIONS`` registry. Loss dispatch (``LOSS_MAPPING``) is
    deferred to ``apply_ops_config`` so there is a single binding point that
    consumes ``OpsImplementationConfig``; ``build_foundation_model`` invokes
    it automatically when callers pass ``ops_implementation=...`` (and
    installs defaults otherwise).
    """
    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend == "hf":
        logger.info_rank0("⚠️ Skip applying ops patch. Using huggingface transformers backend.")
    else:
        from .kernels.attention import apply_veomni_attention_patch

        apply_veomni_attention_patch()
        logger.info_rank0("✅ VeOmni attention patches applied.")


def apply_ops_config(ops_config: OpsImplementationConfig) -> None:
    """Apply kernel patches based on resolved ``OpsImplementationConfig``.

    Single install point for config-driven dispatch:

    1. Binds the cross-entropy kernel into ``LOSS_MAPPING`` via
       ``install_loss_mapping`` (pre-bound ``partial`` — no runtime resolution).
    2. Walks GLOBAL ops (e.g. load-balancing loss) and binds each selected
       backend to its ``global_slot``.
    3. Populates the ops-config singleton so per-model ``device_patch.py`` and
       ``OpSlot.bind`` can read the user's selections.

    MoE dispatch is applied in ``build_foundation_model`` (via
    ``moe_implementation`` ∈ {``eager``, ``fused_triton``, ``fused_quack``,
    ``fused_npu``}); per-model kernels are applied by each model's
    ``device_patch.py``.
    """
    set_ops_config(ops_config)

    modeling_backend = get_env("MODELING_BACKEND")
    if modeling_backend == "hf":
        return

    from .kernels.cross_entropy import install_loss_mapping

    ce_label = install_loss_mapping(ops_config.cross_entropy_loss_implementation)

    applied = apply_global_ops(ops_config)
    applied.insert(0, ce_label)
    logger.info_rank0(f"✅ VeOmni ops config applied: {', '.join(applied)}.")
    logger.info_rank0(format_kernel_functions())


def format_kernel_functions() -> str:
    lines = []
    lines.append("\n=========== OPS ============")

    for alias, func in build_ALL_OPS():
        impl = func.__name__ if func is not None else "None"
        lines.append(f"{alias} = {impl}")

    # Cross-entropy is bound via LOSS_MAPPING (partial-wrapped), not a module
    # global — surface it here so the log still shows the active CE kernel.
    lines.append(f"cross_entropy = {_current_cross_entropy_name()}")

    lines.append("==============================")
    return "\n".join(lines)


def format_kernel_selection_summary(model=None, modeling_module=None) -> str:
    """Final kernel-selection summary, intended to be logged after model build.

    Unlike :func:`format_kernel_functions` (which runs at ``apply_ops_config``
    time and only sees GLOBAL pointers), this summary includes:

    - the attention implementation actually wired onto ``model.config``,
    - the cross-entropy kernel installed into ``LOSS_MAPPING``,
    - all GLOBAL function pointers (``_fused_moe_forward``,
      ``_flash_attention_forward``, ``_load_balancing_loss``) — bound by
      ``apply_ops_config`` and ``_bind_veomni_ops``,
    - every :class:`OpSlot` declared on the model's generated modeling module
      (RMSNorm / RoPE / SwiGLU / MoE experts / per-task loss slots).
    """
    lines = ["", "============= Kernel selection summary ============="]

    if model is not None:
        cfg = getattr(model, "config", None)
        cls = model.__class__
        model_type = getattr(cfg, "model_type", None) if cfg is not None else None
        model_label = f"{cls.__module__}.{cls.__name__}"
        if model_type:
            model_label = f"{model_label} (model_type={model_type})"
        lines.append(f"  model                     = {model_label}")
        for attn_label, attn_value in _collect_attn_implementations(cfg):
            lines.append(f"  {attn_label:<25} = {attn_value}")

    lines.append(f"  cross_entropy             = {_current_cross_entropy_name()}")

    for alias, func in build_ALL_OPS():
        impl = getattr(func, "__name__", repr(func)) if func is not None else "None (eager / unbound)"
        lines.append(f"  {alias:<25} = {impl}")

    if modeling_module is not None:
        slot_entries: list[tuple[str, OpSlot]] = []
        for attr_name in dir(modeling_module):
            obj = getattr(modeling_module, attr_name, None)
            if isinstance(obj, OpSlot):
                slot_entries.append((attr_name, obj))
        if slot_entries:
            module_label = modeling_module.__name__.rsplit(".", 1)[-1]
            lines.append(f"  -- OpSlot bindings ({module_label}) --")
            for attr_name, slot in sorted(slot_entries):
                impl_label = slot.impl_name if slot.impl_name is not None else "<unbound>"
                kernel_label = (
                    getattr(slot.kernel, "__name__", repr(slot.kernel)) if slot.kernel is not None else "eager"
                )
                lines.append(f"  {attr_name:<35} = {impl_label} -> {kernel_label}")

    lines.append("=====================================================")
    return "\n".join(lines)


def _collect_attn_implementations(cfg) -> list[tuple[str, str]]:
    """Return ``[(label, value), ...]`` for every attn impl reachable from ``cfg``.

    Handles three shapes:
    - flat ``cfg._attn_implementation`` (string),
    - HF v5 dict form ``cfg._attn_implementation = {"text_config": ..., ...}``,
    - composite multimodal configs whose sub-configs (``text_config``,
      ``vision_config``, ``audio_config``) each carry their own
      ``_attn_implementation``.
    """
    if cfg is None:
        return []

    out: list[tuple[str, str]] = []
    top = getattr(cfg, "_attn_implementation", None)
    if isinstance(top, dict):
        for k, v in top.items():
            out.append((f"attn_implementation[{k}]", str(v)))
    elif top is not None:
        out.append(("attn_implementation", str(top)))

    for sub_attr in ("text_config", "vision_config", "audio_config", "thinker_config", "talker_config"):
        sub = getattr(cfg, sub_attr, None)
        sub_attn = getattr(sub, "_attn_implementation", None) if sub is not None else None
        if sub_attn is not None and not isinstance(sub_attn, dict):
            out.append((f"attn_implementation[{sub_attr}]", str(sub_attn)))

    return out


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
