# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# See the License for the specific language governing limitations
# under the License.

"""Register ``veomni_*`` attention names and matching mask builders on HF dicts."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ....utils import logging
from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from .mask.flash import flash_attention_mask_builder
from .mask.flex import flex_attention_mask_builder
from .mask.magi import magi_attention_mask_builder
from .mask.sdpa import sdpa_attention_mask_builder
from .standard.flash import flash_attention_forward
from .standard.flex import flex_attention_forward
from .standard.magi import magi_attention_forward
from .standard.sage import sage_attention_forward
from .standard.sdpa import sdpa_attention_forward


logger = logging.get_logger(__name__)

# Every ``veomni_*`` attention name has a matching mask builder. Flash and
# Sage return ``None``; causal stays the ``is_causal`` kwarg.
_VEOMNI_HF_PATCHES: tuple[tuple[str, Callable[..., Any], Callable[..., Any]], ...] = (
    ("veomni_flash_attention_2", flash_attention_forward, flash_attention_mask_builder),
    ("veomni_flash_attention_3", flash_attention_forward, flash_attention_mask_builder),
    ("veomni_flash_attention_4", flash_attention_forward, flash_attention_mask_builder),
    ("veomni_flex_attention", flex_attention_forward, flex_attention_mask_builder),
    ("veomni_magi_attention", magi_attention_forward, magi_attention_mask_builder),
    ("veomni_sage_attention", sage_attention_forward, flash_attention_mask_builder),
    ("veomni_sdpa", sdpa_attention_forward, sdpa_attention_mask_builder),
)

VEOMNI_FLASH_ATTN_IMPL_MAPPING = {
    "veomni_flash_attention_2": "flash_attention_2",
    "veomni_flash_attention_3": "flash_attention_3",
    "veomni_flash_attention_4": "flash_attention_4",
}

_original_load_and_register_attn_kernel: Callable | None = None
_veomni_hub_kernel_loader_patch_applied = False


def _load_veomni_local_flash_kernel(implementation: str) -> SimpleNamespace:
    """Build a local kernel-like object for VeOmni flash attention names.

    Mimics the minimal interface expected by Transformers ``_lazy_imports``:
    ``flash_attn_func`` and ``flash_attn_varlen_func``.
    """
    stock = VEOMNI_FLASH_ATTN_IMPL_MAPPING.get(implementation)
    if stock == "flash_attention_2":
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except ImportError as e:
            raise ImportError(
                f"VeOmni attention implementation `{implementation}` requires `flash_attn` (FA2) to be importable."
            ) from e
    elif stock == "flash_attention_3":
        try:
            from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        except ImportError as e:
            raise ImportError(
                f"VeOmni attention implementation `{implementation}` requires "
                "`flash_attn_interface` (FA3) to be importable."
            ) from e
    elif stock == "flash_attention_4":
        try:
            from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
        except ImportError as e:
            raise ImportError(
                f"VeOmni attention implementation `{implementation}` requires `flash_attn.cute` (FA4) to be importable."
            ) from e
    else:
        raise ValueError(f"Unknown VeOmni flash attention implementation: {implementation}")

    return SimpleNamespace(
        flash_attn_func=flash_attn_func,
        flash_attn_varlen_func=flash_attn_varlen_func,
    )


def patch_transformers_hub_kernel_loader_for_veomni() -> None:
    """Intercept VeOmni flash names before Transformers treats them as hub ids.

    FA2 and FA3 have explicit ``_lazy_imports`` branches. FA4 does not, so it
    falls through to ``load_and_register_attn_kernel``; this patch loads
    ``flash_attn.cute`` locally instead of fetching from the hub.
    """
    global _veomni_hub_kernel_loader_patch_applied
    global _original_load_and_register_attn_kernel

    if _veomni_hub_kernel_loader_patch_applied:
        return

    try:
        import transformers.integrations.hub_kernels as hub_kernels
    except ImportError as e:
        logger.warning_rank0(f"Failed to patch Transformers hub kernel loader for VeOmni attention: {e}")
        return

    _original_load_and_register_attn_kernel = getattr(hub_kernels, "load_and_register_attn_kernel", None)
    if not callable(_original_load_and_register_attn_kernel):
        logger.warning_rank0("Transformers hub kernel loader is unavailable; VeOmni attention loader patch skipped.")
        return

    def _veomni_load_and_register_attn_kernel(
        attn_implementation: str,
        attention_wrapper: Callable | None = None,
        allow_all_kernels: bool = False,
    ) -> SimpleNamespace | object:
        if attn_implementation in VEOMNI_FLASH_ATTN_IMPL_MAPPING:
            return _load_veomni_local_flash_kernel(attn_implementation)

        if is_transformers_version_greater_or_equal_to("5.3.0"):
            return _original_load_and_register_attn_kernel(
                attn_implementation, attention_wrapper, allow_all_kernels=allow_all_kernels
            )
        return _original_load_and_register_attn_kernel(attn_implementation, attention_wrapper)

    hub_kernels.load_and_register_attn_kernel = _veomni_load_and_register_attn_kernel
    _veomni_hub_kernel_loader_patch_applied = True


def apply_veomni_attention_patch() -> None:
    """Register Ulysses-aware ``veomni_*`` keys on HF dicts.

    Safe to call more than once. Overwrites the same names.
    """
    patch_transformers_hub_kernel_loader_for_veomni()
    for name, _, mask_builder in _VEOMNI_HF_PATCHES:
        ALL_MASK_ATTENTION_FUNCTIONS.register(name, mask_builder)
    for name, forward, _ in _VEOMNI_HF_PATCHES:
        ALL_ATTENTION_FUNCTIONS.register(name, forward)
