"""Dense native attention for HunyuanVideo 1.5; no sequence-parallel collectives."""

import torch
import torch.nn.functional as F


def validate_attention_mode(mode):
    # "flash" is the official checkpoint's dense-attention label, not a
    # requirement to import the CUDA-only flash_attn package.
    if mode not in (None, "torch", "flash", "sdpa"):
        raise ValueError(f"HunyuanVideo 1.5 currently supports dense SDPA only, got {mode!r}")
    return "torch"


def attention(q, k, v, drop_rate=0.0, attn_mask=None, causal=False, attn_mode="torch"):
    validate_attention_mode(attn_mode)
    mask = None
    if attn_mask is not None:
        if attn_mask.dtype != torch.bool:
            raise TypeError("The attention boundary requires a boolean validity mask")
        mask = attn_mask[:, None, :, None] & attn_mask[:, None, None, :]
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
        dropout_p=drop_rate,
        is_causal=causal,
    )
    return out.transpose(1, 2).flatten(2)


def parallel_attention(
    q, k, v, img_q_len, img_kv_len, attn_mode=None, text_mask=None, attn_param=None, block_idx=None
):
    validate_attention_mode(attn_mode)
    if img_q_len != img_kv_len:
        raise ValueError("Only self attention is supported")
    q, k, v = (torch.cat(parts, dim=1) for parts in (q, k, v))
    mask = None
    if text_mask is not None:
        mask = F.pad(text_mask.bool(), (img_kv_len, 0), value=True)
    return attention(q, k, v, attn_mask=mask)
