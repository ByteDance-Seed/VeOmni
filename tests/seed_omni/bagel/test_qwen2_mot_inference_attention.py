"""Inference coverage for BAGEL's unified FlashAttention dispatch."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from flash_attn import flash_attn_varlen_func

from tests.seed_omni.bagel.contracts.helpers import config_cls, tiny_bagel_qwen2_cfg
from veomni.models.seed_omni.modules.bagel.qwen2_mot import modeling
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoTAttention, NaiveCache
from veomni.utils.device import get_device_type, get_torch_device


@pytest.mark.skipif(get_torch_device().device_count() < 1, reason="device_count should be >= 1")
def test_inference_flash_attention_wrapper_matches_varlen_kernel_with_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device(f"{get_device_type()}:0")
    config_type = config_cls("bagel_qwen2_mot")
    config = config_type(
        **{
            **tiny_bagel_qwen2_cfg(),
            "num_key_value_heads": 2,
            "attn_implementation": "veomni_flash_attention_2_with_sp",
        }
    )
    attention = BagelQwen2MoTAttention(config, layer_idx=0).to(device=device, dtype=torch.bfloat16).eval()
    original_wrapper = modeling.flash_attention_forward
    calls: list[dict[str, Any]] = []

    def checking_wrapper(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        assert attention_mask is None
        assert kwargs["skip_ulysses"] is True
        wrapped_output, _ = original_wrapper(
            module,
            query,
            key,
            value,
            attention_mask,
            **kwargs,
        )
        expected_output = flash_attn_varlen_func(
            q=query.squeeze(0).transpose(0, 1),
            k=key.squeeze(0).transpose(0, 1),
            v=value.squeeze(0).transpose(0, 1),
            cu_seqlens_q=kwargs["cu_seq_lens_q"],
            cu_seqlens_k=kwargs["cu_seq_lens_k"],
            max_seqlen_q=kwargs["max_length_q"],
            max_seqlen_k=kwargs["max_length_k"],
            causal=kwargs["is_causal"],
        ).unsqueeze(0)
        torch.testing.assert_close(wrapped_output, expected_output, rtol=0.0, atol=0.0)
        calls.append(kwargs)
        return wrapped_output, None

    monkeypatch.setattr(modeling, "flash_attention_forward", checking_wrapper)
    generator = torch.Generator(device=device).manual_seed(8123)
    head_dim = config.hidden_size // config.num_attention_heads
    cache = NaiveCache(num_layers=1)

    prefill_lens = torch.tensor([2, 3], device=device, dtype=torch.int32)
    prefill = torch.randn(
        int(prefill_lens.sum().item()),
        config.hidden_size,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    prefill_output, _ = attention._forward_packed_inference(
        packed_query_sequence=prefill,
        query_lens=prefill_lens,
        packed_query_position_embeddings=(
            torch.ones(prefill.shape[0], head_dim, device=device, dtype=torch.bfloat16),
            torch.zeros(prefill.shape[0], head_dim, device=device, dtype=torch.bfloat16),
        ),
        packed_query_indexes=torch.arange(prefill.shape[0], device=device),
        past_key_values=cache,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
    )
    assert prefill_output.shape == prefill.shape

    decode_lens = torch.tensor([1, 1], device=device, dtype=torch.int32)
    decode = torch.randn(
        int(decode_lens.sum().item()),
        config.hidden_size,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    decode_output, _ = attention._forward_packed_inference(
        packed_query_sequence=decode,
        query_lens=decode_lens,
        packed_query_position_embeddings=(
            torch.ones(decode.shape[0], head_dim, device=device, dtype=torch.bfloat16),
            torch.zeros(decode.shape[0], head_dim, device=device, dtype=torch.bfloat16),
        ),
        packed_query_indexes=torch.tensor([2, 6], device=device),
        past_key_values=cache,
        key_values_lens=prefill_lens,
        packed_key_value_indexes=torch.tensor([0, 1, 3, 4, 5], device=device),
        update_past_key_values=True,
        is_causal=True,
        mode="und",
    )
    assert decode_output.shape == decode.shape
    assert len(calls) == 2
    assert calls[0]["max_length_q"] == 3
    assert calls[0]["max_length_k"] == 3
    assert calls[1]["max_length_q"] == 1
    assert calls[1]["max_length_k"] == 4
