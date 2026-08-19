"""Inference coverage for BAGEL's unified attention facade."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from flash_attn import flash_attn_varlen_func
from torch.nn.attention.flex_attention import BlockMask

from tests.seed_omni.bagel.contracts.helpers import config_cls, tiny_bagel_qwen2_cfg
from veomni.models.seed_omni.modules.bagel.qwen2_mot import accelerated
from veomni.models.seed_omni.modules.bagel.qwen2_mot.masking import (
    build_mot_attention_metadata,
    build_mot_block_mask,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import (
    BaseNavitOutputWithPast,
    NaiveCache,
)
from veomni.utils.device import get_device_type, get_torch_device


def _packed_attention_metadata(
    query_lens: torch.Tensor,
    key_value_lens: torch.Tensor | None = None,
) -> dict[str, Any]:
    key_value_lens = query_lens if key_value_lens is None else key_value_lens
    return {
        "cu_seq_lens_q": torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0)).to(torch.int32),
        "cu_seq_lens_k": torch.nn.functional.pad(torch.cumsum(key_value_lens, dim=0), (1, 0)).to(torch.int32),
        "max_length_q": int(query_lens.max().item()),
        "max_length_k": int(key_value_lens.max().item()),
        "total_key_value_tokens": int(key_value_lens.sum().item()),
    }


def test_forward_inference_temporarily_overrides_and_restores_attention_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_type = config_cls("bagel_qwen2_mot")
    config = config_type(
        **{
            **tiny_bagel_qwen2_cfg(),
            "attn_implementation": "veomni_flex_attention_with_sp",
        }
    )
    model = accelerated.BagelQwen2MoTAccelerated(config).eval()
    packed_query = torch.randn(2, config.hidden_size)
    observed_implementations: list[str] = []

    def fake_forward(**kwargs: Any) -> BaseNavitOutputWithPast:
        observed_implementations.append(model.config._attn_implementation)
        return BaseNavitOutputWithPast(
            packed_query_sequence=kwargs["packed_query_sequence"],
            past_key_values=kwargs["past_key_values"],
        )

    monkeypatch.setattr(model.model, "forward_packed_inference", fake_forward)
    output = model.forward_inference(
        packed_query_sequence=packed_query,
        query_lens=torch.tensor([2], dtype=torch.int32),
        packed_query_position_ids=torch.arange(2),
        packed_query_indexes=torch.arange(2),
        past_key_values=NaiveCache(num_layers=1),
        attention_implementation="veomni_flash_attention_2_with_sp",
    )

    assert output["hidden_states"] is packed_query
    assert observed_implementations == ["veomni_flash_attention_2_with_sp"]
    assert config._attn_implementation == "veomni_flex_attention_with_sp"

    def failing_forward(**kwargs: Any) -> BaseNavitOutputWithPast:
        del kwargs
        assert model.config._attn_implementation == "veomni_flash_attention_2_with_sp"
        raise RuntimeError("inference failure")

    monkeypatch.setattr(model.model, "forward_packed_inference", failing_forward)
    with pytest.raises(RuntimeError, match="inference failure"):
        model.forward_inference(
            packed_query_sequence=packed_query,
            query_lens=torch.tensor([2], dtype=torch.int32),
            packed_query_position_ids=torch.arange(2),
            packed_query_indexes=torch.arange(2),
            attention_implementation="veomni_flash_attention_2_with_sp",
        )
    assert config._attn_implementation == "veomni_flex_attention_with_sp"


def test_inference_attention_facade_dispatches_flex_for_packed_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_type = config_cls("bagel_qwen2_mot")
    config = config_type(
        **{
            **tiny_bagel_qwen2_cfg(),
            "attn_implementation": "veomni_flex_attention_with_sp",
        }
    )
    attention = accelerated.BagelQwen2MoTAttentionAccelerated(config, layer_idx=0).to(dtype=torch.bfloat16).eval()
    query_lens = torch.tensor([2, 3], dtype=torch.int32)
    packed_query = torch.randn(5, config.hidden_size, dtype=torch.bfloat16)
    attention_metadata = build_mot_attention_metadata(
        [[2], [3]],
        [["full"], ["causal"]],
        device=torch.device("cpu"),
    )
    block_mask = build_mot_block_mask(attention_metadata)
    calls: list[dict[str, Any]] = []

    def fake_flex_facade(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        assert module.config._attn_implementation == "veomni_flex_attention_with_sp"
        assert isinstance(attention_mask, BlockMask)
        assert kwargs["skip_ulysses"] is True
        assert "cu_seq_lens_q" not in kwargs
        calls.append(kwargs)
        return query.transpose(1, 2), None

    monkeypatch.setattr(accelerated, "fused_attention_forward", fake_flex_facade)
    cache = NaiveCache(num_layers=1)
    output, output_cache = attention.forward_packed_inference(
        packed_query_sequence=packed_query,
        query_lens=query_lens,
        packed_query_position_embeddings=(
            torch.ones(5, config.hidden_size // config.num_attention_heads, dtype=torch.bfloat16),
            torch.zeros(5, config.hidden_size // config.num_attention_heads, dtype=torch.bfloat16),
        ),
        packed_query_indexes=torch.arange(5),
        past_key_values=cache,
        update_past_key_values=True,
        is_causal=False,
        mode="und",
        attention_mask=block_mask,
        **_packed_attention_metadata(query_lens),
    )

    assert output.shape == packed_query.shape
    assert output_cache is cache
    assert cache.key_cache[0].shape[0] == 5
    assert cache.value_cache[0].shape[0] == 5
    assert len(calls) == 1


@pytest.mark.skipif(
    get_device_type() != "cuda" or get_torch_device().device_count() < 1,
    reason="Flex/FlashAttention prefill parity requires a CUDA device",
)
def test_flex_prefill_matches_spanwise_flash_attention() -> None:
    device = torch.device(f"{get_device_type()}:0")
    config_type = config_cls("bagel_qwen2_mot")
    config = config_type(
        **{
            **tiny_bagel_qwen2_cfg(),
            "attn_implementation": "veomni_flex_attention_with_sp",
        }
    )
    model = accelerated.BagelQwen2MoTAccelerated(config).to(device=device, dtype=torch.bfloat16).eval()
    generator = torch.Generator(device=device).manual_seed(1234)
    image = torch.randn(2, config.hidden_size, generator=generator, device=device, dtype=torch.bfloat16)
    text = torch.randn(3, config.hidden_size, generator=generator, device=device, dtype=torch.bfloat16)
    packed_query = torch.cat([image, text], dim=0)
    position_ids = torch.tensor([0, 0, 1, 2, 3], device=device)

    flex_cache = NaiveCache(config.num_hidden_layers)
    attention_metadata = build_mot_attention_metadata(
        [[2, 3]],
        [["full", "causal"]],
        device=device,
    )
    with torch.no_grad():
        flex_output = model.forward_inference(
            packed_query_sequence=packed_query,
            query_lens=torch.tensor([5], device=device, dtype=torch.int32),
            packed_query_position_ids=position_ids,
            packed_query_indexes=torch.arange(5, device=device),
            past_key_values=flex_cache,
            key_values_lens=torch.zeros(1, device=device, dtype=torch.int32),
            packed_key_value_indexes=torch.empty(0, device=device, dtype=torch.long),
            update_past_key_values=True,
            is_causal=False,
            mode="gen",
            packed_attention_metadata=attention_metadata,
            packed_vae_token_indexes=torch.tensor([0, 1], device=device),
            packed_text_indexes=torch.tensor([2, 3, 4], device=device),
        )

    flash_cache = NaiveCache(config.num_hidden_layers)
    with torch.no_grad():
        image_output = model.forward_inference(
            packed_query_sequence=image,
            query_lens=torch.tensor([2], device=device, dtype=torch.int32),
            packed_query_position_ids=position_ids[:2],
            packed_query_indexes=torch.arange(2, device=device),
            past_key_values=flash_cache,
            key_values_lens=torch.zeros(1, device=device, dtype=torch.int32),
            packed_key_value_indexes=torch.empty(0, device=device, dtype=torch.long),
            update_past_key_values=True,
            is_causal=False,
            mode="gen",
            attention_implementation="veomni_flash_attention_2_with_sp",
            packed_vae_token_indexes=torch.tensor([0, 1], device=device),
            packed_text_indexes=torch.empty(0, device=device, dtype=torch.long),
        )
        text_output = model.forward_inference(
            packed_query_sequence=text,
            query_lens=torch.tensor([3], device=device, dtype=torch.int32),
            packed_query_position_ids=position_ids[2:],
            packed_query_indexes=torch.tensor([2, 3, 4], device=device),
            past_key_values=flash_cache,
            key_values_lens=torch.tensor([2], device=device, dtype=torch.int32),
            packed_key_value_indexes=torch.tensor([0, 1], device=device),
            update_past_key_values=True,
            is_causal=True,
            mode="und",
            attention_implementation="veomni_flash_attention_2_with_sp",
        )

    flash_hidden_states = torch.cat(
        [image_output["hidden_states"], text_output["hidden_states"]],
        dim=0,
    )
    torch.testing.assert_close(flex_output["hidden_states"], flash_hidden_states, rtol=2e-2, atol=2e-2)
    for layer_idx in range(config.num_hidden_layers):
        torch.testing.assert_close(
            flex_cache.key_cache[layer_idx],
            flash_cache.key_cache[layer_idx],
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            flex_cache.value_cache[layer_idx],
            flash_cache.value_cache[layer_idx],
            rtol=2e-2,
            atol=2e-2,
        )


@pytest.mark.skipif(
    get_device_type() != "cuda" or get_torch_device().device_count() < 1,
    reason="FlashAttention inference parity requires a CUDA device",
)
def test_inference_attention_facade_dispatches_flash_with_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device(f"{get_device_type()}:0")
    config_type = config_cls("bagel_qwen2_mot")
    config = config_type(
        **{
            **tiny_bagel_qwen2_cfg(),
            "num_key_value_heads": 2,
            "attn_implementation": "veomni_flash_attention_2_with_sp",
        }
    )
    attention = (
        accelerated.BagelQwen2MoTAttentionAccelerated(config, layer_idx=0)
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    original_facade = accelerated.fused_attention_forward
    calls: list[dict[str, Any]] = []

    def checking_facade(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        assert attention_mask is None
        assert kwargs["skip_ulysses"] is True
        wrapped_output, _ = original_facade(
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

    monkeypatch.setattr(accelerated, "fused_attention_forward", checking_facade)
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
    prefill_output, _ = attention.forward_packed_inference(
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
        **_packed_attention_metadata(prefill_lens),
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
    decode_output, _ = attention.forward_packed_inference(
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
        **_packed_attention_metadata(decode_lens, prefill_lens + decode_lens),
    )
    assert decode_output.shape == decode.shape
    assert len(calls) == 2
    assert calls[0]["max_length_q"] == 3
    assert calls[0]["max_length_k"] == 3
    assert calls[1]["max_length_q"] == 1
    assert calls[1]["max_length_k"] == 4
