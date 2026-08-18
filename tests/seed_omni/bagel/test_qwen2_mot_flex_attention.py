"""FlexAttention tests with a dense-mask oracle for BAGEL Qwen2-MoT training."""

from __future__ import annotations

import importlib
import random
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask
from torch.nn.functional import scaled_dot_product_attention

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls, tiny_bagel_qwen2_cfg
from veomni.models.seed_omni.modules.bagel.qwen2_mot import accelerated
from veomni.models.seed_omni.modules.bagel.qwen2_mot.masking import (
    build_mot_attention_metadata,
    build_mot_block_mask,
    build_mot_sdpa_mask,
    pad_mot_attention_metadata,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoTAttention
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.ops.kernels.attention import fused_attention_forward


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
    assert materialized.any(dim=-1).all()
    assert torch.equal(materialized, dense_oracle)
    assert torch.equal(build_mot_sdpa_mask(metadata), materialized)


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
        torch.testing.assert_close(materialized, dense_oracle)


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

    output = attention._forward_packed_train(
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

    output = attention._forward_packed_train(
        packed_sequence=torch.randn(sequence_length, config.hidden_size),
        attention_mask=build_mot_sdpa_mask(metadata),
        packed_position_cos=torch.ones(sequence_length, attention.head_dim),
        packed_position_sin=torch.zeros(sequence_length, attention.head_dim),
        packed_und_token_indexes=torch.arange(sequence_length),
        packed_gen_token_indexes=torch.empty(0, dtype=torch.long),
    )

    assert output.shape == (sequence_length, config.hidden_size)
    assert torch.isfinite(output).all()


def test_eager_model_advertises_sdpa_only() -> None:
    from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoT

    assert BagelQwen2MoT._supports_sdpa is True
    assert BagelQwen2MoT._supports_flex_attn is False


def test_accelerated_model_advertises_flex_attention() -> None:
    assert accelerated.BagelQwen2MoTAccelerated._supports_flex_attn is True
    assert accelerated.BagelQwen2MoTAccelerated._supports_sdpa is True


def test_accelerated_training_attention_rejects_non_flex_backend() -> None:
    config_type = config_cls("bagel_qwen2_mot")
    config = config_type(**tiny_bagel_qwen2_cfg(), attn_implementation="sdpa")
    attention = accelerated.BagelQwen2MoTAttentionAccelerated(config, layer_idx=0)
    sequence_length = 2
    metadata = build_mot_attention_metadata([[2]], [["causal"]], device=torch.device("cpu"))

    with pytest.raises(ValueError, match="requires packed fused attention"):
        attention._forward_packed_train(
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
