"""CPU-only correctness test for the Wan self-attention padding mask.

Unlike `test_wan_ulysses_padding.py` (which needs NCCL/multi-process to exercise
the actual Ulysses slice/gather), the mask itself is a single-process concern:
does masking out the padded key positions with `eager_attention_forward`
produce EXACTLY the same result as never having padded at all? This runs on
CPU with no distributed setup.
"""

import torch

from veomni.models.transformers.wan.modeling_wan import _trim_kv_tail_padding, eager_attention_forward


class _Module:
    training = False


def test_padding_mask_matches_unpadded_reference():
    torch.manual_seed(0)
    batch, heads, real_len, head_dim = 2, 4, 37, 16
    pad_size = 3  # e.g. local_len=40, real_len=37 on the last SP rank

    query = torch.randn(batch, heads, real_len, head_dim)
    key_real = torch.randn(batch, heads, real_len, head_dim)
    value_real = torch.randn(batch, heads, real_len, head_dim)

    # Reference: no padding ever existed.
    ref_output, _ = eager_attention_forward(_Module(), query, key_real, value_real, attention_mask=None)

    # Padded: append zero keys/values (as `padding_tensor_for_seqeunce_parallel`
    # does), mask them out exactly as WanModel.forward now does.
    zeros = torch.zeros(batch, heads, pad_size, head_dim)
    key_padded = torch.cat([key_real, zeros], dim=2)
    value_padded = torch.cat([value_real, zeros], dim=2)

    local_len = real_len + pad_size
    mask = torch.zeros(1, 1, 1, local_len)
    mask[..., local_len - pad_size :] = torch.finfo(torch.float32).min

    padded_output, _ = eager_attention_forward(_Module(), query, key_padded, value_padded, attention_mask=mask)

    torch.testing.assert_close(padded_output, ref_output, atol=1e-6, rtol=1e-5)


def test_unmasked_padding_would_have_diverged():
    """Sanity check that the test above is actually exercising something --
    without the mask, padded zero-keys DO perturb the output, confirming the
    fix is load-bearing rather than a no-op."""
    torch.manual_seed(0)
    batch, heads, real_len, head_dim = 2, 4, 37, 16
    pad_size = 3

    query = torch.randn(batch, heads, real_len, head_dim)
    key_real = torch.randn(batch, heads, real_len, head_dim)
    value_real = torch.randn(batch, heads, real_len, head_dim)

    ref_output, _ = eager_attention_forward(_Module(), query, key_real, value_real, attention_mask=None)

    zeros = torch.zeros(batch, heads, pad_size, head_dim)
    key_padded = torch.cat([key_real, zeros], dim=2)
    value_padded = torch.cat([value_real, zeros], dim=2)

    unmasked_output, _ = eager_attention_forward(_Module(), query, key_padded, value_padded, attention_mask=None)

    assert not torch.allclose(unmasked_output, ref_output, atol=1e-6, rtol=1e-5), (
        "expected unmasked padding to perturb the output -- if this now passes, "
        "the mask test above may not be exercising real behavior"
    )


def test_trim_kv_tail_padding_matches_unpadded_reference():
    """`_trim_kv_tail_padding` backs `wrapped_flash_attention_3`/`wrapped_sageattention`,
    neither of which take a mask argument. It must drop exactly the padded tail the
    global mask marks, and re-report that count as `pad_size` so the caller can re-pad
    the kernel's output back to the original (padded) length."""
    torch.manual_seed(0)
    batch, heads, real_len, head_dim = 2, 4, 37, 16
    pad_size = 3
    local_len = real_len + pad_size

    query = torch.randn(batch, heads, local_len, head_dim)
    key = torch.randn(batch, heads, local_len, head_dim)
    value = torch.randn(batch, heads, local_len, head_dim)

    mask = torch.zeros(1, 1, 1, local_len)
    mask[..., local_len - pad_size :] = torch.finfo(torch.float32).min

    trimmed_q, trimmed_k, trimmed_v, reported_pad = _trim_kv_tail_padding(query, key, value, mask)

    assert reported_pad == pad_size
    torch.testing.assert_close(trimmed_q, query[..., :real_len, :])
    torch.testing.assert_close(trimmed_k, key[..., :real_len, :])
    torch.testing.assert_close(trimmed_v, value[..., :real_len, :])


def test_trim_kv_tail_padding_is_noop_without_mask():
    query = torch.randn(2, 4, 37, 16)
    key = torch.randn(2, 4, 37, 16)
    value = torch.randn(2, 4, 37, 16)

    trimmed_q, trimmed_k, trimmed_v, pad_size = _trim_kv_tail_padding(query, key, value, None)

    assert pad_size == 0
    assert trimmed_q is query
    assert trimmed_k is key
    assert trimmed_v is value


if __name__ == "__main__":
    test_padding_mask_matches_unpadded_reference()
    test_unmasked_padding_would_have_diverged()
    test_trim_kv_tail_padding_matches_unpadded_reference()
    test_trim_kv_tail_padding_is_noop_without_mask()
    print("OK")
