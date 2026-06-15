from types import SimpleNamespace

import pytest
import torch

import veomni.ops.kernels.attention as attention
from veomni.models.diffusers.wan_t2v.wan_transformer.modeling_wan_transformer import _get_wan_packed_varlen_kwargs


def _attention_module():
    return SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="veomni_flash_attention_2_with_sp"),
        is_causal=False,
        layer_idx=None,
    )


def test_flash_attention_forward_passes_explicit_varlen_metadata(monkeypatch):
    captured = {}

    def fake_flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        is_causal,
        **kwargs,
    ):
        captured.update(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            query_length=query_length,
            is_causal=is_causal,
            cu_seq_lens_q=kwargs["cu_seq_lens_q"],
            cu_seq_lens_k=kwargs["cu_seq_lens_k"],
            max_length_q=kwargs["max_length_q"],
            max_length_k=kwargs["max_length_k"],
        )
        return query_states

    monkeypatch.setattr(attention, "_flash_attention_forward", fake_flash_attention_forward)

    query = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
    key = torch.randn(1, 2, 7, 4, dtype=torch.bfloat16)
    value = torch.randn(1, 2, 7, 4, dtype=torch.bfloat16)

    output, _ = attention.flash_attention_forward(
        _attention_module(),
        query,
        key,
        value,
        attention_mask=None,
        is_causal=False,
        skip_ulysses=True,
        cu_seq_lens_q=torch.tensor([0, 5], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, 7], dtype=torch.int32),
        max_length_q=5,
        max_length_k=7,
    )

    assert output.shape == (1, 5, 2, 4)
    assert captured["attention_mask"] is None
    assert captured["query_length"] == 5
    assert captured["is_causal"] is False
    torch.testing.assert_close(captured["cu_seq_lens_q"], torch.tensor([0, 5], dtype=torch.int32))
    torch.testing.assert_close(captured["cu_seq_lens_k"], torch.tensor([0, 7], dtype=torch.int32))
    assert captured["max_length_q"] == 5
    assert captured["max_length_k"] == 7


def test_wan_packed_varlen_kwargs_rejects_batched_dense_input():
    query = torch.randn(2, 5, 2, 4, dtype=torch.bfloat16)
    key = torch.randn(2, 5, 2, 4, dtype=torch.bfloat16)
    value = torch.randn(2, 5, 2, 4, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="batch-size-1"):
        _get_wan_packed_varlen_kwargs(query, key, value, None)


def test_wan_packed_varlen_kwargs_rejects_dense_attention_mask():
    query = torch.randn(1, 5, 2, 4, dtype=torch.bfloat16)
    key = torch.randn(1, 5, 2, 4, dtype=torch.bfloat16)
    value = torch.randn(1, 5, 2, 4, dtype=torch.bfloat16)
    attention_mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)

    with pytest.raises(ValueError, match="does not accept a dense attention mask"):
        _get_wan_packed_varlen_kwargs(query, key, value, attention_mask)


def test_wan_packed_varlen_kwargs_supplies_single_sequence_metadata():
    query = torch.randn(1, 5, 2, 4, dtype=torch.bfloat16)
    key = torch.randn(1, 7, 2, 4, dtype=torch.bfloat16)
    value = torch.randn(1, 7, 2, 4, dtype=torch.bfloat16)

    kwargs = _get_wan_packed_varlen_kwargs(query, key, value, None)

    torch.testing.assert_close(kwargs["cu_seq_lens_q"], torch.tensor([0, 5], dtype=torch.int32))
    torch.testing.assert_close(kwargs["cu_seq_lens_k"], torch.tensor([0, 7], dtype=torch.int32))
    assert kwargs["max_length_q"] == 5
    assert kwargs["max_length_k"] == 7
