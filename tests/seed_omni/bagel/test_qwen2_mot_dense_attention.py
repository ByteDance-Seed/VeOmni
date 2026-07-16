"""Dense masked-SDPA parity tests for BAGEL Qwen2-MoT training."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls, tiny_bagel_qwen2_cfg
from veomni.models.seed_omni.modules.bagel.qwen2_mot import modeling
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoTAttention
from veomni.models.seed_omni.modules.bagel.qwen2_mot.processing import build_mot_attention_mask
from veomni.models.seed_omni.utils.conversation import ConversationItem


def _spanwise_math_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    sample_splits: list[list[int]],
    sample_attn_modes: list[list[str]],
) -> torch.Tensor:
    """Reproduce the legacy span semantics with explicit per-span math SDPA."""
    sample_lens = [sum(split_lens) for split_lens in sample_splits]
    query_samples = torch.split(query, sample_lens, dim=0)
    key_samples = torch.split(key, sample_lens, dim=0)
    value_samples = torch.split(value, sample_lens, dim=0)
    sample_outputs: list[torch.Tensor] = []

    for query_sample, key_sample, value_sample, split_lens, attn_modes in zip(
        query_samples,
        key_samples,
        value_samples,
        sample_splits,
        sample_attn_modes,
        strict=True,
    ):
        query_spans = torch.split(query_sample, split_lens, dim=0)
        key_spans = torch.split(key_sample, split_lens, dim=0)
        value_spans = torch.split(value_sample, split_lens, dim=0)
        clean_keys: list[torch.Tensor] = []
        clean_values: list[torch.Tensor] = []
        span_outputs: list[torch.Tensor] = []

        for query_span, key_span, value_span, mode in zip(
            query_spans,
            key_spans,
            value_spans,
            attn_modes,
            strict=True,
        ):
            keys = torch.cat((*clean_keys, key_span), dim=0)
            values = torch.cat((*clean_values, value_span), dim=0)
            context_length = keys.shape[0] - key_span.shape[0]
            mask = torch.ones((query_span.shape[0], keys.shape[0]), device=query.device, dtype=torch.bool)
            if mode == "causal":
                mask[:, context_length:] = torch.ones(
                    (query_span.shape[0], key_span.shape[0]),
                    device=query.device,
                    dtype=torch.bool,
                ).tril()

            num_groups = query_span.shape[1] // key_span.shape[1]
            expanded_keys = torch.repeat_interleave(keys, repeats=num_groups, dim=1)
            expanded_values = torch.repeat_interleave(values, repeats=num_groups, dim=1)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                output = scaled_dot_product_attention(
                    query_span.transpose(0, 1).unsqueeze(0),
                    expanded_keys.transpose(0, 1).unsqueeze(0),
                    expanded_values.transpose(0, 1).unsqueeze(0),
                    attn_mask=mask.unsqueeze(0).unsqueeze(0),
                    dropout_p=0.0,
                    is_causal=False,
                )
            span_outputs.append(output.squeeze(0).transpose(0, 1))
            if mode != "noise":
                clean_keys.append(key_span)
                clean_values.append(value_span)

        sample_outputs.append(torch.cat(span_outputs, dim=0))

    return torch.cat(sample_outputs, dim=0).contiguous()


@pytest.mark.parametrize(
    ("sample_splits", "sample_attn_modes", "num_heads", "num_key_value_heads"),
    [
        ([[2, 3, 2]], [["causal", "full", "causal"]], 4, 2),
        ([[2, 2, 2, 2]], [["full", "noise", "noise", "causal"]], 4, 2),
        ([[1, 4, 1, 3]], [["full", "full", "full", "noise"]], 28, 4),
        ([[2, 3], [1, 2, 1]], [["causal", "noise"], ["full", "causal", "full"]], 4, 2),
        # SP4 local head layout for BAGEL 7B: global 28Q/4KV -> local 7Q/1KV.
        ([[2, 1, 3]], [["causal", "full", "noise"]], 7, 1),
    ],
)
def test_dense_masked_attention_matches_spanwise_forward_and_qkv_gradients(
    sample_splits: list[list[int]],
    sample_attn_modes: list[list[str]],
    num_heads: int,
    num_key_value_heads: int,
) -> None:
    config_type = config_cls("bagel_qwen2_mot")
    hidden_size = num_heads * 8
    config = config_type(
        **{
            **tiny_bagel_qwen2_cfg(),
            "hidden_size": hidden_size,
            "intermediate_size": hidden_size * 2,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_key_value_heads,
        }
    )
    attention = BagelQwen2MoTAttention(config, layer_idx=0)
    sequence_length = sum(sum(split_lens) for split_lens in sample_splits)
    generator = torch.Generator().manual_seed(2917)
    tensor_shapes = [
        (sequence_length, num_heads, attention.head_dim),
        (sequence_length, num_key_value_heads, attention.head_dim),
        (sequence_length, num_key_value_heads, attention.head_dim),
    ]
    tensors = [torch.randn(shape, generator=generator) for shape in tensor_shapes]
    dense_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
    oracle_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
    attention_mask = build_mot_attention_mask(sample_splits, sample_attn_modes, device=torch.device("cpu"))

    dense_output = attention._masked_dense_attention(*dense_inputs, attention_mask=attention_mask)
    oracle_output = _spanwise_math_oracle(
        *oracle_inputs,
        sample_splits=sample_splits,
        sample_attn_modes=sample_attn_modes,
    )
    torch.testing.assert_close(dense_output, oracle_output, rtol=1e-5, atol=1e-5)

    output_gradient = torch.randn(dense_output.shape, generator=generator)
    dense_gradients = torch.autograd.grad(dense_output, dense_inputs, output_gradient)
    oracle_gradients = torch.autograd.grad(oracle_output, oracle_inputs, output_gradient)
    for dense_gradient, oracle_gradient in zip(dense_gradients, oracle_gradients, strict=True):
        torch.testing.assert_close(dense_gradient, oracle_gradient, rtol=1e-5, atol=1e-5)


def test_training_attention_calls_sdpa_once(monkeypatch: pytest.MonkeyPatch) -> None:
    config_type = config_cls("bagel_qwen2_mot")
    config = config_type(**tiny_bagel_qwen2_cfg())
    attention = BagelQwen2MoTAttention(config, layer_idx=0)
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))

    call_count = 0
    original_sdpa = modeling.scaled_dot_product_attention

    def counted_sdpa(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(modeling, "scaled_dot_product_attention", counted_sdpa)
    sequence_length = 6
    packed_sequence = torch.randn(sequence_length, config.hidden_size)
    attention_mask = build_mot_attention_mask(
        [[2, 2, 2]],
        [["causal", "full", "noise"]],
        device=torch.device("cpu"),
    )
    packed_position_cos = torch.ones(sequence_length, attention.head_dim)
    packed_position_sin = torch.zeros(sequence_length, attention.head_dim)

    output = attention._forward_packed_train(
        packed_sequence=packed_sequence,
        attention_mask=attention_mask,
        packed_position_cos=packed_position_cos,
        packed_position_sin=packed_position_sin,
        packed_und_token_indexes=torch.tensor([0, 1, 2, 3]),
        packed_gen_token_indexes=torch.tensor([4, 5]),
    )

    assert output.shape == packed_sequence.shape
    assert torch.isfinite(output).all()
    assert call_count == 1


def test_training_forward_gradient_checkpointing_uses_tensor_mask_metadata() -> None:
    model_type = model_cls("bagel_qwen2_mot")
    config_type = config_cls("bagel_qwen2_mot")
    model = model_type(config_type(**tiny_bagel_qwen2_cfg())).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    hidden_size = int(model.config.hidden_size)
    text = torch.randn(2, hidden_size, requires_grad=True)
    output = torch.randn(3, hidden_size, requires_grad=True)
    inputs = model.forward_pre(
        conversation_list=[
            [
                ConversationItem(type="text", value=text, role="user"),
                ConversationItem(type="output", value=output, role="assistant"),
            ]
        ]
    )

    hidden_states = model(**inputs)["hidden_states"]
    hidden_states.float().square().mean().backward()

    assert text.grad is not None and torch.isfinite(text.grad).all()
    assert output.grad is not None and torch.isfinite(output.grad).all()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None
    assert model.model.layers[0].self_attn.q_proj_moe_gen.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CuDNN SDPA validation requires CUDA")
def test_cudnn_sdpa_supports_dense_bool_mask_and_native_local_gqa() -> None:
    device = torch.device("cuda")
    query = torch.randn(1, 7, 8, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(1, 1, 8, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(1, 1, 8, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    attention_mask = build_mot_attention_mask(
        [[2, 3, 3]],
        [["causal", "full", "noise"]],
        device=device,
    )

    with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )
    output.float().square().mean().backward()

    assert torch.isfinite(output).all()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_native_gqa_rejects_invalid_global_head_ratio_at_config_init() -> None:
    config_type = config_cls("bagel_qwen2_mot")
    with pytest.raises(ValueError, match="query heads must be divisible"):
        config_type(
            **{
                **tiny_bagel_qwen2_cfg(),
                "num_attention_heads": 6,
                "num_key_value_heads": 4,
            }
        )


def test_native_gqa_rejects_kv_heads_that_cannot_be_sharded_in_sp_pre(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from veomni.models.seed_omni.modules.bagel.qwen2_mot import modulemixin

    model_type = model_cls("bagel_qwen2_mot")
    config_type = config_cls("bagel_qwen2_mot")
    model = model_type(
        config_type(
            **{
                **tiny_bagel_qwen2_cfg(),
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
            }
        )
    )
    monkeypatch.setattr(
        modulemixin,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_size=1, ulysses_size=4),
    )

    with pytest.raises(ValueError, match="KV heads must be divisible"):
        model.forward_sp_pre(
            packed_sequence=torch.randn(2, model.config.hidden_size),
            packed_position_ids=torch.arange(2),
            packed_token_type_ids=torch.zeros(2, dtype=torch.long),
            attention_mask=torch.ones(1, 1, 2, 2, dtype=torch.bool),
        )
