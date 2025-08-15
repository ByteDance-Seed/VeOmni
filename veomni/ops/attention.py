from typing import Optional, Tuple

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from ..distributed.parallel_state import get_parallel_state
from ..distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
)
from ..utils import logging


logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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
) -> Tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask", None) is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    # This is for Qwen2VL's mrope
    position_ids = kwargs.pop("position_ids", None)
    if position_ids is not None and position_ids.dim() == 3:
        position_ids = position_ids[0]

    # Ulysses patch
    ulysses_enabled = get_parallel_state().ulysses_enabled
    if ulysses_enabled:
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
            key = repeat_kv(key, n_repeat)
            value = repeat_kv(value, n_repeat)

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

    attn_output: torch.Tensor = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=module.is_causal,
        dropout=dropout,
        position_ids=position_ids,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=False,
        **kwargs,
    )

    # Ulysses patch
    if ulysses_enabled:
        ulysses_group = get_parallel_state().ulysses_group
        if attn_output.ndim == 4 and attn_output.size(0) == 1:
            attn_output = attn_output.squeeze(0)
            attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1, group=ulysses_group)
            attn_output = attn_output.unsqueeze(0)
        else:
            attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2, group=ulysses_group)

    return attn_output, None


def flash_attention_3_forward():
    # TODO: add flash attention 3
    pass
