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

from types import SimpleNamespace
from typing import Callable, Optional

import torch
from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward as _transformers_flash_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...distributed.parallel_state import get_parallel_state
from ...distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
)
from ...utils import logging
from ...utils.import_utils import is_transformers_version_greater_or_equal_to


logger = logging.get_logger(__name__)
_flash_attention_forward = None
_original_load_and_register_attn_kernel: Callable | None = None
_veomni_hub_kernel_loader_patch_applied = False

_VEOMNI_FLASH_ATTN_IMPL_MAPPING = {
    "veomni_flash_attention_2_with_sp": "flash_attention_2",
    "veomni_flash_attention_3_with_sp": "flash_attention_3",
}


def _is_veomni_custom_flash_attention(implementation: str | None) -> bool:
    return implementation in _VEOMNI_FLASH_ATTN_IMPL_MAPPING


def _load_veomni_local_flash_kernel(implementation: str) -> SimpleNamespace:
    """
    Build a local kernel-like object for VeOmni custom attention names.

    This object mimics the minimal interface expected by Transformers `_lazy_imports`,
    i.e. it exposes `flash_attn_func` and `flash_attn_varlen_func`.
    """
    if implementation == "veomni_flash_attention_2_with_sp":
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except ImportError as e:
            raise ImportError(
                "VeOmni attention implementation `veomni_flash_attention_2_with_sp` requires "
                "`flash_attn` (FA2) to be importable."
            ) from e
    elif implementation == "veomni_flash_attention_3_with_sp":
        try:
            from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        except ImportError as e:
            raise ImportError(
                "VeOmni attention implementation `veomni_flash_attention_3_with_sp` requires "
                "`flash_attn_interface` (FA3) to be importable."
            ) from e
    else:
        raise ValueError(f"Unknown VeOmni flash attention implementation: {implementation}")

    return SimpleNamespace(
        flash_attn_func=flash_attn_func,
        flash_attn_varlen_func=flash_attn_varlen_func,
    )


def _patch_transformers_hub_kernel_loader_for_veomni():
    """
    Patch Transformers hub-kernel loader to support VeOmni custom attention names.

    See docs/transformers_v5/veomni_flash_attention_kernel_adapter.md for
    background, failure mode, and design details.

    Transformers>=5 may route custom flash names through
    `transformers.integrations.hub_kernels.load_and_register_attn_kernel`.
    VeOmni names are not hub kernel identifiers, so we intercept and provide
    local FA2/FA3 functions directly.
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
        attn_implementation: str, attention_wrapper: Callable | None = None
    ) -> SimpleNamespace | object:
        if _is_veomni_custom_flash_attention(attn_implementation):
            return _load_veomni_local_flash_kernel(attn_implementation)
        return _original_load_and_register_attn_kernel(attn_implementation, attention_wrapper)

    hub_kernels.load_and_register_attn_kernel = _veomni_load_and_register_attn_kernel
    _veomni_hub_kernel_loader_patch_applied = True


def transformers_flash_attention_forward(
    query,
    key,
    value,
    attention_mask,
    **kwargs,
):
    attn_implementation = kwargs.pop("attn_implementation")
    return _transformers_flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        implementation=attn_implementation,  # TODO(szl): bug in 4.57.3, update to 5.0 to remove this patch
        **kwargs,
    )


# patch transformers.integrations.flash_attention.py
# 1. set use_top_left_mask always False
# 2. optional ulysses sp patch
# 3. external flash attention backends (for internal use)
def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    skip_ulysses: bool = False,  # Skip ulysses for some ViT cases like internvl3.5
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # This is before the transpose
    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )
    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = module.is_causal

    # Ulysses patch
    ulysses_enabled = get_parallel_state().ulysses_enabled
    if ulysses_enabled and not skip_ulysses:
        ulysses_group = get_parallel_state().ulysses_group
        # Sanity Check & Repeat Key & Value
        ulysses_size = get_parallel_state().ulysses_size
        q_head_num = query.shape[2]
        kv_head_num = key.shape[2]
        unpadded_seq_len = None

        assert q_head_num % ulysses_size == 0, (
            f"num_query_heads ({q_head_num}) must be divisible by ulysses_size ({ulysses_size})"
        )
        if ulysses_size > kv_head_num:
            assert ulysses_size % kv_head_num == 0, (
                f"ulysses_size ({ulysses_size}) must be divisible by num_key_value_heads ({kv_head_num})"
            )
            n_repeat = ulysses_size // kv_head_num
            # Shape before: (batch_size, seq_len, kv_head_num, head_dim)
            # This repeats the K/V heads (dim 2) to match the ulysses_size (SP world size)
            # Shape after: (batch_size, seq_len, kv_head_num * n_repeat, head_dim)
            # where (kv_head_num * n_repeat) == ulysses_size
            key = torch.repeat_interleave(key, dim=2, repeats=n_repeat)
            value = torch.repeat_interleave(value, dim=2, repeats=n_repeat)

        if query.ndim == 4 and query.size(0) == 1:
            query, key, value = query.squeeze(0), key.squeeze(0), value.squeeze(0)
            query = gather_seq_scatter_heads(
                query, seq_dim=0, head_dim=1, group=ulysses_group, unpadded_dim_size=unpadded_seq_len
            )
            key = gather_seq_scatter_heads(
                key, seq_dim=0, head_dim=1, group=ulysses_group, unpadded_dim_size=unpadded_seq_len
            )
            value = gather_seq_scatter_heads(
                value, seq_dim=0, head_dim=1, group=ulysses_group, unpadded_dim_size=unpadded_seq_len
            )
            query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)
        else:
            query = gather_seq_scatter_heads(
                query, seq_dim=1, head_dim=2, group=ulysses_group, unpadded_dim_size=unpadded_seq_len
            )
            key = gather_seq_scatter_heads(
                key, seq_dim=1, head_dim=2, group=ulysses_group, unpadded_dim_size=unpadded_seq_len
            )
            value = gather_seq_scatter_heads(
                value, seq_dim=1, head_dim=2, group=ulysses_group, unpadded_dim_size=unpadded_seq_len
            )

        # Only after all_to_all we got the full seq_len
        seq_len = query.shape[1]

    if module.config._attn_implementation == "veomni_flash_attention_2_with_sp":
        fa_kernel_implementation = "flash_attention_2"
    elif module.config._attn_implementation == "veomni_flash_attention_3_with_sp":
        fa_kernel_implementation = "flash_attention_3"
    else:
        raise ValueError(
            f"unknown attn_implementation for veomni flash_attention with SP support: {module.config._attn_implementation}"
        )

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=False,
        target_dtype=target_dtype,
        attn_implementation=fa_kernel_implementation,
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        **kwargs,
    )

    # Ulysses patch
    if ulysses_enabled and not skip_ulysses:
        ulysses_group = get_parallel_state().ulysses_group
        if attn_output.ndim == 4 and attn_output.size(0) == 1:
            attn_output = attn_output.squeeze(0)
            attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1, group=ulysses_group)
            attn_output = attn_output.unsqueeze(0)
        else:
            attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2, group=ulysses_group)

    return attn_output, None


def apply_veomni_attention_patch():
    if is_transformers_version_greater_or_equal_to("5.0.0"):
        _patch_transformers_hub_kernel_loader_for_veomni()
    ALL_ATTENTION_FUNCTIONS.register("veomni_flash_attention_2_with_sp", flash_attention_forward)
    ALL_ATTENTION_FUNCTIONS.register("veomni_flash_attention_3_with_sp", flash_attention_forward)
    global _flash_attention_forward
    _flash_attention_forward = transformers_flash_attention_forward
