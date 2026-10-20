"""Dense native attention for HunyuanVideo 1.5; no sequence-parallel collectives."""

import torch
import torch.nn.functional as F


def validate_attention_mode(mode):
    """Normalize supported dense-attention labels to the native SDPA path."""
    # "flash" is the official checkpoint's dense-attention label, not a
    # requirement to import the CUDA-only flash_attn package.
    if mode not in (None, "torch", "flash", "sdpa"):
        raise ValueError(f"HunyuanVideo 1.5 currently supports dense SDPA only, got {mode!r}")
    return "torch"


def attention(q, k, v, drop_rate=0.0, attn_mask=None, causal=False, attn_mode="torch"):
    """Attend to valid keys and zero invalid queries with linear mask storage."""
    validate_attention_mode(attn_mode)
    mask = None
    if attn_mask is not None:
        if attn_mask.dtype != torch.bool:
            raise TypeError("The attention boundary requires a boolean validity mask")
        mask = attn_mask[:, None, None, :]
        if q.device.type == "npu":
            # CANN requires the query dimension, even for a key-only mask.
            # expand is a zero-stride view; do not materialize a pairwise mask here.
            mask = mask.expand(-1, 1, q.shape[1], -1)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
        dropout_p=drop_rate,
        is_causal=causal,
    )
    if attn_mask is not None:
        out = out.masked_fill(~attn_mask[:, None, :, None], 0)
    return out.transpose(1, 2).flatten(2)


def parallel_attention(
    q, k, v, img_q_len, img_kv_len, attn_mode=None, text_mask=None, attn_param=None, block_idx=None
):
    """Join image/text streams for dense self attention without SP collectives."""
    validate_attention_mode(attn_mode)
    if img_q_len != img_kv_len:
        raise ValueError("Only self attention is supported")
    q, k, v = (torch.cat(parts, dim=1) for parts in (q, k, v))
    mask = None
    if text_mask is not None:
        mask = F.pad(text_mask.bool(), (img_kv_len, 0), value=True)
    return attention(q, k, v, attn_mask=mask)
