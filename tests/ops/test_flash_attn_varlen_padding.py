import pytest
import torch

from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


def _skip_if_no_flash_attn():
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA is required for flash-attn.")
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except Exception as exc:
        pytest.skip(f"flash-attn is not available: {exc}")


def test_varlen_flash_attn_padded_input_matches_unpadded():
    """Varlen FA should tolerate padded tails when cu_seqlens reflect real tokens."""
    _skip_if_no_flash_attn()
    from flash_attn import flash_attn_varlen_func

    torch.manual_seed(0)
    device = torch.device(get_device_type())
    dtype = torch.float16

    seqlens = torch.tensor([5, 7], dtype=torch.int32, device=device)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0), value=0)
    max_seqlen = int(seqlens.max().item())

    total_tokens = int(cu_seqlens[-1].item())
    padded_tokens = total_tokens + 4

    nheads = 4
    head_dim = 8

    q = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    q_padded = torch.cat(
        [q, torch.zeros(padded_tokens - total_tokens, nheads, head_dim, device=device, dtype=dtype)],
        dim=0,
    )
    k_padded = torch.cat(
        [k, torch.zeros(padded_tokens - total_tokens, nheads, head_dim, device=device, dtype=dtype)],
        dim=0,
    )
    v_padded = torch.cat(
        [v, torch.zeros(padded_tokens - total_tokens, nheads, head_dim, device=device, dtype=dtype)],
        dim=0,
    )

    out_unpadded = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=0.0)
    out_padded = flash_attn_varlen_func(
        q_padded, k_padded, v_padded, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=0.0
    )

    assert out_padded.shape[0] == padded_tokens
    torch.testing.assert_close(out_padded[:total_tokens], out_unpadded, rtol=0.0, atol=0.0)
