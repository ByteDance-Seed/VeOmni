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

"""FlashAttention backend loading and SP-aware adapter implementation."""

from typing import Optional

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from .....distributed.parallel_state import get_parallel_state
from .....utils import logging
from ..ulysses import (
    prepare_ulysses_qkv,
    restore_ulysses_output,
    should_apply_ulysses,
    slice_ulysses_head_auxiliary,
)


logger = logging.get_logger(__name__)


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
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    VeOmni unified flash-attention forward, registered in Transformers'
    ``ALL_ATTENTION_FUNCTIONS`` for all three ``veomni_flash_attention_*``
    implementation names.

    Differences from the stock Transformers flash-attention forward:

    1. ``use_top_left_mask`` is always ``False`` — VeOmni models handle masking
       via ``cu_seqlens`` (varlen path) and do not need the top-left causal mask
       workaround required by some older Transformers models.

    2. **Ulysses sequence-parallelism** — when Ulysses SP is on and async is
       off, the full Q/K/V sequence is gathered across SP ranks before the
       kernel call and the output is scattered back afterwards. Async SP and
       ``ulysses_size == 1`` leave the layout unchanged.

    3. **FA backend selection** — the implementation name stored in
       ``module.config._attn_implementation`` is mapped to the token that
       Transformers' ``lazy_import_flash_attention`` expects:

       * FA2/FA3 → plain name (``"flash_attention_2"`` / ``"flash_attention_3"``)
         because ``_lazy_imports`` has an explicit branch for each and resolves
         them without touching the hub-kernel path.
       * FA4 → kept as ``"veomni_flash_attention_4"`` so that
         Transformers v5's hub-kernel fallback is intercepted by VeOmni's
         monkey-patch of ``load_and_register_attn_kernel``, which loads
         ``flash_attn.cute`` locally instead of fetching from the hub.
    """
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
    parallel_state = get_parallel_state()
    ulysses_enabled = should_apply_ulysses()
    if ulysses_enabled:
        query, key, value, query_head_count = prepare_ulysses_qkv(
            query,
            key,
            value,
            group=parallel_state.ulysses_group,
            ulysses_size=parallel_state.ulysses_size,
        )

        # Only after all_to_all we got the full seq_len
        seq_len = query.shape[1]
        if "s_aux" in kwargs:
            kwargs["s_aux"] = slice_ulysses_head_auxiliary(
                kwargs["s_aux"],
                query_head_count=query_head_count,
                local_query_head_count=query.shape[2],
                group=parallel_state.ulysses_group,
            )

    # Resolve the token that will be passed to Transformers' lazy_import_flash_attention.
    #
    # FA2 and FA3 have dedicated branches in transformers' _lazy_imports, so we
    # use the plain transformers names and they are resolved without hitting the
    # hub-kernel path.
    #
    # FA4 has no such branch; unrecognised names fall through to the hub-kernel
    # loader. By keeping the VeOmni name here, our monkey-patch of
    # ``load_and_register_attn_kernel`` intercepts it and loads
    # ``flash_attn.cute`` locally.
    impl = module.config._attn_implementation
    if "flash_attention_2" in impl:
        fa_kernel_implementation = "flash_attention_2"
    elif "flash_attention_3" in impl:
        fa_kernel_implementation = "flash_attention_3"
    elif "flash_attention_4" in impl:
        fa_kernel_implementation = impl
    else:
        raise ValueError(f"unknown attn_implementation for veomni_flash_attention: {impl}")

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
    if ulysses_enabled:
        attn_output = restore_ulysses_output(attn_output, group=parallel_state.ulysses_group)

    return attn_output, None
