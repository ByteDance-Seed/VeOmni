"""BAGEL Qwen2-MoT training and inference attention coverage."""

from __future__ import annotations

import importlib
import random
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from flash_attn import flash_attn_varlen_func
from torch.nn.attention.flex_attention import BlockMask, create_mask
from torch.nn.functional import scaled_dot_product_attention

from tests.seed_omni.bagel.helpers import config_cls, model_cls, tiny_bagel_qwen2_cfg
from veomni.models.seed_omni.mixins.base_mixin import BaseMixin
from veomni.models.seed_omni.modules.bagel.qwen2_mot import accelerated
from veomni.models.seed_omni.modules.bagel.qwen2_mot.masking import (
    build_mot_attention_metadata,
    build_mot_block_mask,
    build_mot_sdpa_mask,
    pad_mot_attention_metadata,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import (
    BagelQwen2MoT,
    BagelQwen2MoTAttention,
    BagelQwen2MoTCore,
    BaseNavitOutputWithPast,
    InferenceMixin,
    NaiveCache,
    _sdpa_packed_attention,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.ops.kernels.attention import fused_attention_forward
from veomni.utils.device import get_device_type, get_torch_device


def _flex_config(**overrides):
    config_type = config_cls("bagel_qwen2_mot")
    return config_type(
        **{
            **tiny_bagel_qwen2_cfg(),
            "attn_implementation": "veomni_flex_attention_with_sp",
            **overrides,
        }
    )


def _build_dense_attention_mask_oracle(
    sample_splits: list[list[int]],
    sample_attn_modes: list[list[str]],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Materialize BAGEL visibility for tests without entering production code."""
    total_length = sum(sum(split_lens) for split_lens in sample_splits)
    visible = torch.zeros((total_length, total_length), device=device, dtype=torch.bool)
    sample_start = 0

    for split_lens, attn_modes in zip(sample_splits, sample_attn_modes, strict=True):
        clean_spans: list[tuple[int, int]] = []
        span_start = sample_start
        for length, mode in zip(split_lens, attn_modes, strict=True):
            span_end = span_start + length
            for clean_start, clean_end in clean_spans:
                visible[span_start:span_end, clean_start:clean_end] = True

            if mode == "causal":
                visible[span_start:span_end, span_start:span_end].fill_(True).tril_()
            else:
                visible[span_start:span_end, span_start:span_end] = True

            if mode != "noise":
                clean_spans.append((span_start, span_end))
            span_start = span_end

        sample_start = span_start

    return visible.unsqueeze(0).unsqueeze(0).contiguous()


def test_dense_attention_oracle_preserves_causal_full_and_noise_semantics() -> None:
    mask = _build_dense_attention_mask_oracle(
        [[2, 2, 2, 1]],
        [["causal", "full", "noise", "causal"]],
        device=torch.device("cpu"),
    )[0, 0]

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_dense_attention_oracle_isolates_packed_samples() -> None:
    mask = _build_dense_attention_mask_oracle(
        [[2], [1, 2]],
        [["full"], ["causal", "noise"]],
        device=torch.device("cpu"),
    )

    assert mask.shape == (1, 1, 5, 5)
    assert mask.dtype == torch.bool
    assert mask.is_contiguous()
    assert not mask[0, 0, :2, 2:].any()
    assert not mask[0, 0, 2:, :2].any()


def _dense_mask_to_block_presence(mask: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    sequence_length = int(mask.shape[-1])
    padded_length = (sequence_length + block_size - 1) // block_size * block_size
    padded = torch.nn.functional.pad(mask, (0, padded_length - sequence_length, 0, padded_length - sequence_length))
    block_count = padded_length // block_size
    return (
        padded.view(1, 1, block_count, block_size, block_count, block_size).permute(0, 1, 2, 4, 3, 5).any(dim=(-2, -1))
    )


def _ordered_rows_to_dense(num_blocks: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    dense = torch.zeros(indices.shape, dtype=torch.bool, device=indices.device)
    for row in range(indices.shape[-2]):
        count = int(num_blocks[0, 0, row])
        if count:
            dense[0, 0, row, indices[0, 0, row, :count].long()] = True
    return dense


@pytest.mark.parametrize(
    ("sample_splits", "sample_attn_modes"),
    [
        ([[2, 3, 2]], [["causal", "full", "causal"]]),
        ([[2, 2, 2, 2]], [["full", "noise", "noise", "causal"]]),
        ([[1, 4, 1, 3]], [["full", "full", "full", "noise"]]),
        ([[2, 3], [1, 2, 1]], [["causal", "noise"], ["full", "causal", "full"]]),
    ],
)
def test_block_mask_metadata_matches_dense_attention_oracle(
    sample_splits: list[list[int]],
    sample_attn_modes: list[list[str]],
) -> None:
    metadata = build_mot_attention_metadata(sample_splits, sample_attn_modes, device=torch.device("cpu"))
    block_mask = build_mot_block_mask(metadata)
    sequence_length = metadata.shape[1]
    materialized = create_mask(
        block_mask.mask_mod,
        B=1,
        H=1,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device="cpu",
    )
    dense_oracle = _build_dense_attention_mask_oracle(
        sample_splits,
        sample_attn_modes,
        device=torch.device("cpu"),
    )

    assert metadata.shape == (3, sequence_length)
    assert metadata.dtype == torch.int32
    assert metadata.is_contiguous()
    assert metadata.numel() == 3 * sequence_length
    sdpa_mask = build_mot_sdpa_mask(metadata)
    assert materialized.any(dim=-1).all()
    assert torch.equal(materialized, dense_oracle)
    assert sdpa_mask.dtype == torch.bool
    assert torch.equal(sdpa_mask, materialized)


def test_randomized_block_mask_metadata_matches_dense_attention_oracle() -> None:
    generator = random.Random(3197)
    modes = ("causal", "full", "noise")
    for _ in range(40):
        sample_splits: list[list[int]] = []
        sample_attn_modes: list[list[str]] = []
        for _sample in range(generator.randint(1, 4)):
            span_count = generator.randint(1, 6)
            sample_splits.append([generator.randint(1, 5) for _span in range(span_count)])
            sample_attn_modes.append([generator.choice(modes) for _span in range(span_count)])

        metadata = build_mot_attention_metadata(sample_splits, sample_attn_modes, device=torch.device("cpu"))
        block_mask = build_mot_block_mask(metadata)
        sequence_length = metadata.shape[1]
        materialized = create_mask(block_mask.mask_mod, 1, 1, sequence_length, sequence_length, device="cpu")
        dense_oracle = _build_dense_attention_mask_oracle(
            sample_splits,
            sample_attn_modes,
            device=torch.device("cpu"),
        )
        sdpa_mask = build_mot_sdpa_mask(metadata)
        assert sdpa_mask.dtype == torch.bool
        torch.testing.assert_close(materialized, dense_oracle)
        torch.testing.assert_close(sdpa_mask, dense_oracle)


def test_block_mask_sparse_layout_covers_dense_attention_oracle() -> None:
    sample_splits = [[137, 259, 131, 73], [191, 149, 263]]
    sample_attn_modes = [["causal", "full", "noise", "causal"], ["full", "noise", "causal"]]
    metadata = build_mot_attention_metadata(sample_splits, sample_attn_modes, device=torch.device("cpu"))
    block_mask = build_mot_block_mask(metadata)
    dense_oracle = _build_dense_attention_mask_oracle(
        sample_splits,
        sample_attn_modes,
        device=torch.device("cpu"),
    )
    required_blocks = _dense_mask_to_block_presence(dense_oracle)

    assert torch.all(~required_blocks | block_mask.to_dense())
    full_blocks = _ordered_rows_to_dense(block_mask.full_kv_num_blocks, block_mask.full_kv_indices)
    padded_length = full_blocks.shape[-1] * block_mask.BLOCK_SIZE[1]
    padded_oracle = torch.nn.functional.pad(
        dense_oracle,
        (0, padded_length - dense_oracle.shape[-1], 0, padded_length - dense_oracle.shape[-2]),
    )
    block_is_fully_visible = (
        padded_oracle.view(
            1,
            1,
            full_blocks.shape[-2],
            block_mask.BLOCK_SIZE[0],
            full_blocks.shape[-1],
            block_mask.BLOCK_SIZE[1],
        )
        .permute(0, 1, 2, 4, 3, 5)
        .all(dim=(-2, -1))
    )
    assert torch.all(~full_blocks | block_is_fully_visible)


def test_block_mask_construction_does_not_materialize_token_dense_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flex_attention_module = importlib.import_module("torch.nn.attention.flex_attention")

    def forbidden_create_mask(*args, **kwargs):
        raise AssertionError("BlockMask construction must not materialize an O(sequence^2) token mask")

    monkeypatch.setattr(flex_attention_module, "create_mask", forbidden_create_mask)
    metadata = build_mot_attention_metadata(
        [[1024, 1024]],
        [["causal", "full"]],
        device=torch.device("cpu"),
    )

    block_mask = build_mot_block_mask(metadata)

    assert block_mask.shape == (1, 1, 2048, 2048)


def test_sp_padding_metadata_is_isolated_and_has_visible_rows() -> None:
    metadata = build_mot_attention_metadata(
        [[2, 3]],
        [["causal", "noise"]],
        device=torch.device("cpu"),
    )
    padded = pad_mot_attention_metadata(metadata, padded_length=8)
    block_mask = build_mot_block_mask(padded)
    materialized = create_mask(block_mask.mask_mod, 1, 1, 8, 8, device="cpu")
    dense_oracle = _build_dense_attention_mask_oracle(
        [[2, 3]],
        [["causal", "noise"]],
        device=torch.device("cpu"),
    )

    assert torch.equal(materialized[..., :5, :5], dense_oracle)
    assert not materialized[..., :5, 5:].any()
    assert not materialized[..., 5:, :5].any()
    assert materialized[..., 5:, 5:].all()
    assert materialized.any(dim=-1).all()


def test_training_attention_dispatches_unified_flex_once(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _flex_config()
    attention = accelerated.BagelQwen2MoTAttentionAccelerated(config, layer_idx=0)
    calls = []

    def fake_attention(module, query, key, value, attention_mask, **kwargs):
        calls.append((module, query.shape, key.shape, value.shape, attention_mask, kwargs))
        return query.transpose(1, 2).contiguous(), None

    monkeypatch.setattr(accelerated, "fused_attention_forward", fake_attention)
    sequence_length = 6
    packed_sequence = torch.randn(sequence_length, config.hidden_size)
    metadata = build_mot_attention_metadata(
        [[2, 2, 2]],
        [["causal", "full", "noise"]],
        device=torch.device("cpu"),
    )
    packed_position_cos = torch.ones(sequence_length, attention.head_dim)
    packed_position_sin = torch.zeros(sequence_length, attention.head_dim)

    output = attention.forward_packed_train(
        packed_sequence=packed_sequence,
        attention_mask=build_mot_block_mask(metadata),
        packed_position_cos=packed_position_cos,
        packed_position_sin=packed_position_sin,
        packed_und_token_indexes=torch.tensor([0, 1, 2, 3]),
        packed_gen_token_indexes=torch.tensor([4, 5]),
    )

    assert output.shape == packed_sequence.shape
    assert torch.isfinite(output).all()
    assert len(calls) == 1
    assert calls[0][1] == (1, config.num_attention_heads, sequence_length, attention.head_dim)
    assert calls[0][4].shape == (1, 1, sequence_length, sequence_length)
    assert calls[0][5]["scaling"] == attention.head_dim**-0.5


def test_eager_training_attention_uses_sdpa() -> None:
    config_type = config_cls("bagel_qwen2_mot")
    config = config_type(**tiny_bagel_qwen2_cfg(), attn_implementation="sdpa")
    attention = BagelQwen2MoTAttention(config, layer_idx=0)
    sequence_length = 2
    metadata = build_mot_attention_metadata([[2]], [["causal"]], device=torch.device("cpu"))

    output = attention.forward_packed_train(
        packed_sequence=torch.randn(sequence_length, config.hidden_size),
        attention_mask=build_mot_sdpa_mask(metadata),
        packed_position_cos=torch.ones(sequence_length, attention.head_dim),
        packed_position_sin=torch.zeros(sequence_length, attention.head_dim),
        packed_und_token_indexes=torch.arange(sequence_length),
        packed_gen_token_indexes=torch.empty(0, dtype=torch.long),
    )

    assert output.shape == (sequence_length, config.hidden_size)
    assert torch.isfinite(output).all()


def test_eager_sdpa_mask_is_boolean() -> None:
    metadata = build_mot_attention_metadata([[2]], [["causal"]], device=torch.device("cpu"))
    mask = build_mot_sdpa_mask(metadata)
    packed = torch.randn(2, 4, 8)

    assert mask.dtype == torch.bool
    output = _sdpa_packed_attention(
        packed,
        packed,
        packed,
        attention_mask=mask,
        scale=1.0,
        enable_gqa=False,
    )
    assert output.shape == packed.shape
    assert torch.isfinite(output).all()


def test_eager_sdpa_masked_gqa_repeats_kv_heads() -> None:
    metadata = build_mot_attention_metadata([[3]], [["causal"]], device=torch.device("cpu"))
    mask = build_mot_sdpa_mask(metadata)
    query = torch.randn(3, 4, 8)
    key = torch.randn(3, 2, 8)
    value = torch.randn(3, 2, 8)

    output = _sdpa_packed_attention(
        query,
        key,
        value,
        attention_mask=mask,
        scale=1.0,
        enable_gqa=True,
    )
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_eager_model_advertises_sdpa_only() -> None:
    assert BagelQwen2MoT._supports_sdpa is True
    assert BagelQwen2MoT._supports_flex_attn is False


def test_accelerated_model_advertises_flex_attention() -> None:
    assert accelerated.BagelQwen2MoTAccelerated._supports_flex_attn is True
    assert accelerated.BagelQwen2MoTAccelerated._supports_sdpa is True


def test_eager_and_accelerated_share_core_not_each_other() -> None:
    assert issubclass(BagelQwen2MoT, InferenceMixin)
    assert issubclass(BagelQwen2MoT, BagelQwen2MoTCore)
    assert not issubclass(BagelQwen2MoT, BaseMixin)
    assert not issubclass(BagelQwen2MoT, accelerated.InferenceMixinAccelerated)

    accelerated_cls = accelerated.BagelQwen2MoTAccelerated
    assert issubclass(accelerated_cls, accelerated.InferenceMixinAccelerated)
    assert issubclass(accelerated_cls, InferenceMixin)
    assert issubclass(accelerated_cls, BagelQwen2MoTCore)
    assert issubclass(accelerated_cls, BaseMixin)
    assert not issubclass(accelerated_cls, BagelQwen2MoT)
    assert BagelQwen2MoT.denoise_branch is InferenceMixin.denoise_branch
    assert accelerated_cls.denoise_branch is accelerated.InferenceMixinAccelerated.denoise_branch


def test_accelerated_training_attention_rejects_non_flex_backend() -> None:
    config_type = config_cls("bagel_qwen2_mot")
    config = config_type(**tiny_bagel_qwen2_cfg(), attn_implementation="sdpa")
    attention = accelerated.BagelQwen2MoTAttentionAccelerated(config, layer_idx=0)
    sequence_length = 2
    metadata = build_mot_attention_metadata([[2]], [["causal"]], device=torch.device("cpu"))

    with pytest.raises(ValueError, match="requires packed fused attention"):
        attention.forward_packed_train(
            packed_sequence=torch.randn(sequence_length, config.hidden_size),
            attention_mask=build_mot_block_mask(metadata),
            packed_position_cos=torch.ones(sequence_length, attention.head_dim),
            packed_position_sin=torch.zeros(sequence_length, attention.head_dim),
            packed_und_token_indexes=torch.arange(sequence_length),
            packed_gen_token_indexes=torch.empty(0, dtype=torch.long),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward requires CUDA")
def test_flex_attention_matches_dense_sdpa_forward_and_qkv_gradients() -> None:
    config = _flex_config(num_key_value_heads=2)
    attention = BagelQwen2MoTAttention(config, layer_idx=0).cuda().train()
    sequence_length = 64
    metadata = build_mot_attention_metadata(
        [[17, 23, 24]],
        [["causal", "full", "noise"]],
        device=torch.device("cuda"),
    )
    block_mask = build_mot_block_mask(metadata)
    dense_mask = _build_dense_attention_mask_oracle(
        [[17, 23, 24]],
        [["causal", "full", "noise"]],
        device=torch.device("cuda"),
    )
    generator = torch.Generator(device="cuda").manual_seed(9231)
    base_inputs = [
        torch.randn(
            (1, heads, sequence_length, attention.head_dim),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for heads in (4, 2, 2)
    ]
    flex_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in base_inputs]
    dense_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in base_inputs]

    flex_output, _ = fused_attention_forward(attention, *flex_inputs, block_mask)
    dense_output = scaled_dot_product_attention(
        *dense_inputs,
        attn_mask=dense_mask,
        enable_gqa=True,
    ).transpose(1, 2)
    torch.testing.assert_close(flex_output, dense_output, rtol=2e-2, atol=2e-2)

    output_gradient = torch.randn(
        flex_output.shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    flex_gradients = torch.autograd.grad(flex_output, flex_inputs, output_gradient)
    dense_gradients = torch.autograd.grad(dense_output, dense_inputs, output_gradient)
    for flex_gradient, dense_gradient in zip(flex_gradients, dense_gradients, strict=True):
        torch.testing.assert_close(flex_gradient, dense_gradient, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward requires CUDA")
def test_training_forward_gradient_checkpointing_reuses_one_block_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    model_type = model_cls("bagel_qwen2_mot")
    model = model_type(_flex_config()).to(device="cuda", dtype=torch.bfloat16).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    original_builder = accelerated.BagelQwen2MoTAttentionAccelerated.build_attention_mask
    build_count = 0

    def counted_builder(metadata):
        nonlocal build_count
        build_count += 1
        return original_builder(metadata)

    monkeypatch.setattr(accelerated.BagelQwen2MoTAttentionAccelerated, "build_attention_mask", counted_builder)
    hidden_size = int(model.config.hidden_size)
    text = torch.randn(2, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = torch.randn(3, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
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

    assert build_count == 1
    assert text.grad is not None and torch.isfinite(text.grad).all()
    assert output.grad is not None and torch.isfinite(output.grad).all()
    assert model.model.layers[0].self_attn.qkv_proj_und.weight.grad is not None
    assert model.model.layers[0].self_attn.qkv_proj_gen.weight.grad is not None


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


def test_native_gqa_rejects_kv_heads_that_cannot_be_sharded_in_forward_pre(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from veomni.models.seed_omni.modules.bagel.qwen2_mot import accelerated

    model_type = model_cls("bagel_qwen2_mot")
    model = model_type(
        _flex_config(
            num_attention_heads=8,
            num_key_value_heads=2,
        )
    )
    monkeypatch.setattr(
        accelerated,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_size=4, cp_size=1, ulysses_size=4),
    )
    conversation = [
        [
            ConversationItem(
                type="text",
                value=torch.randn(2, model.config.hidden_size),
                role="user",
            )
        ]
    ]

    with pytest.raises(ValueError, match="KV heads must be divisible"):
        model.forward_pre(conversation_list=conversation)


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
