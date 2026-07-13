import copy
import types
from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer

from veomni.arguments import ChunkMBSConfig, MixedPrecisionConfig
from veomni.distributed.chunk_mbs import (
    PackedSequenceRange,
    apply_chunk_mbs,
    build_chunk_mbs_ranges,
    chunk_mbs_context,
)


def _config(chunk_mbs=2):
    return ChunkMBSConfig(enable=True, chunk_mbs=chunk_mbs)


def test_build_chunk_mbs_ranges_from_dynamic_packed_batch():
    batch = {"cu_seq_lens_q": torch.tensor([0, 3, 5, 9, 10], dtype=torch.int32)}

    ranges = build_chunk_mbs_ranges(batch, _config(chunk_mbs=2))

    assert ranges is not None
    assert [
        (range_.segment_start, range_.segment_end, range_.token_start, range_.token_end, range_.max_length)
        for range_ in ranges
    ] == [(0, 2, 0, 5, 3), (2, 4, 5, 10, 4)]


def test_build_chunk_mbs_ranges_from_fixed_sample_packed_batch(monkeypatch):
    import veomni.data.data_collator as data_collator

    monkeypatch.setattr(
        data_collator,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, sp_size=1, sp_rank=0),
    )
    collator = data_collator.MainCollator()
    batch = collator(
        [
            {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([1, 2, 3], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([4, 5], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1], dtype=torch.long),
                "labels": torch.tensor([4, 5], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([6, 7, 8, 9], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([6, 7, 8, 9], dtype=torch.long),
            },
        ]
    )

    ranges = build_chunk_mbs_ranges(batch, _config(chunk_mbs=2))

    assert ranges is not None
    assert batch["cu_seq_lens_q"].tolist() == [0, 3, 5, 9]
    assert [
        (range_.segment_start, range_.segment_end, range_.token_start, range_.token_end, range_.max_length)
        for range_ in ranges
    ] == [(0, 2, 0, 5, 3), (2, 3, 5, 9, 4)]


def test_build_chunk_mbs_ranges_noops_when_micro_batch_is_small():
    batch = {"cu_seq_lens_q": torch.tensor([0, 3, 5], dtype=torch.int32)}

    assert build_chunk_mbs_ranges(batch, _config(chunk_mbs=2)) is None


@pytest.mark.parametrize(
    "cu_seq_lens_q",
    [
        torch.tensor([0, 3, 2, 5], dtype=torch.int32),
        torch.tensor([0, 3, 3, 5], dtype=torch.int32),
    ],
)
def test_build_chunk_mbs_ranges_rejects_non_increasing_segments(cu_seq_lens_q):
    with pytest.raises(ValueError, match="strictly increasing cu_seq_lens_q"):
        build_chunk_mbs_ranges({"cu_seq_lens_q": cu_seq_lens_q}, _config(chunk_mbs=4))


def test_build_chunk_mbs_ranges_rejects_nonzero_start():
    with pytest.raises(ValueError, match="start from 0"):
        build_chunk_mbs_ranges({"cu_seq_lens_q": torch.tensor([1, 3, 5], dtype=torch.int32)}, _config(chunk_mbs=1))


@pytest.mark.parametrize(
    ("linear_attn_cu_seq_lens_q", "match"),
    [
        (torch.tensor([1, 3, 5, 10], dtype=torch.int32), "start from 0"),
        (torch.tensor([0, 5, 3, 10], dtype=torch.int32), "strictly increasing linear_attn_cu_seq_lens_q"),
        (torch.tensor([0, 3, 5, 9], dtype=torch.int32), "end at cu_seq_lens_q"),
    ],
)
def test_build_chunk_mbs_ranges_rejects_invalid_linear_attn_cu_seq_lens(linear_attn_cu_seq_lens_q, match):
    batch = {
        "cu_seq_lens_q": torch.tensor([0, 3, 5, 10], dtype=torch.int32),
        "linear_attn_cu_seq_lens_q": linear_attn_cu_seq_lens_q,
    }
    with pytest.raises(ValueError, match=match):
        build_chunk_mbs_ranges(batch, _config(chunk_mbs=2))


def test_build_chunk_mbs_ranges_rejects_misaligned_linear_attn_boundary():
    batch = {
        "cu_seq_lens_q": torch.tensor([0, 3, 5, 9, 10], dtype=torch.int32),
        "linear_attn_cu_seq_lens_q": torch.tensor([0, 3, 5, 10], dtype=torch.int32),
    }

    with pytest.raises(ValueError, match="must align with linear_attn_cu_seq_lens_q"):
        build_chunk_mbs_ranges(batch, _config(chunk_mbs=3))


def test_build_chunk_mbs_ranges_rejects_non_packed_batch_dim():
    batch = {"hidden_states": torch.ones(4, 8, 2)}

    with pytest.raises(ValueError, match="packed-sequence FlashAttention kwargs"):
        build_chunk_mbs_ranges(batch, _config(chunk_mbs=2))


class _ToyDecoderLayer(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.calls = []

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        cu_seq_lens_q=None,
        cu_seq_lens_k=None,
        max_length_q=None,
        max_length_k=None,
        use_cache=None,
        past_key_value=None,
        past_key_values=None,
        layer_past=None,
        **kwargs,
    ):
        linear_attn_cu_seq_lens_q = kwargs.get("linear_attn_cu_seq_lens_q")
        self.calls.append(
            {
                "hidden_shape": tuple(hidden_states.shape),
                "position_ids": position_ids.clone(),
                "cu_seq_lens_q": cu_seq_lens_q.clone(),
                "cu_seq_lens_k": cu_seq_lens_k.clone(),
                "max_length_q": max_length_q,
                "max_length_k": max_length_k,
                "attention_shape": tuple(attention_mask.shape),
                "has_linear_attn_cu": "linear_attn_cu_seq_lens_q" in kwargs,
                "linear_attn_cu_seq_lens_q": (
                    linear_attn_cu_seq_lens_q.clone() if linear_attn_cu_seq_lens_q is not None else None
                ),
                "use_cache": use_cache,
                "past_key_value": past_key_value,
                "past_key_values": past_key_values,
                "layer_past": layer_past,
            }
        )
        cos, sin = position_embeddings
        return hidden_states + position_ids.unsqueeze(-1).to(hidden_states.dtype) + cos + sin


class _ToyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ToyDecoderLayer(), _ToyDecoderLayer()])


class _ToyVisionBlock(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class _AuxDecoderLayer(GradientCheckpointingLayer):
    def forward(self, hidden_states):
        return hidden_states


class _PlainDecoderLayer(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class _ToyRoot(nn.Module):
    _no_split_modules = ["_ToyDecoderLayer", "_ToyVisionBlock"]

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _ToyLanguageModel()
        self.model.visual = _ToyVisionBlock()

    def gradient_checkpointing_enable(self, checkpoint_func=None, gradient_checkpointing_kwargs=None):
        if checkpoint_func is None:
            checkpoint_func = partial(checkpoint, **(gradient_checkpointing_kwargs or {}))
        for layer in self.model.language_model.layers:
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = checkpoint_func

    def gradient_checkpointing_disable(self):
        for layer in self.model.language_model.layers:
            layer.gradient_checkpointing = False


class _Qwen3VLRoot(nn.Module):
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, layer):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([layer])


def _packed_causal_attention_mask(cu_seq_lens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    cu_values = [int(v) for v in cu_seq_lens.tolist()]
    total_seq_len = cu_values[-1]
    mask = torch.full((1, 1, total_seq_len, total_seq_len), -1e4, dtype=dtype)
    for start, end in zip(cu_values, cu_values[1:]):
        length = end - start
        mask[:, :, start:end, start:end] = torch.triu(torch.full((length, length), -1e4, dtype=dtype), diagonal=1)
    return mask


def _run_qwen3_vl_decoder_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    max_length: int,
    ranges,
):
    layer.train()
    for param in layer.parameters():
        param.grad = None

    hidden_states = hidden_states.detach().clone().requires_grad_(True)
    with chunk_mbs_context(ranges):
        output = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=max_length,
            max_length_k=max_length,
            use_cache=False,
        )
        loss = output.float().square().mean()
        loss.backward()

    param_grads = {name: param.grad.detach().clone() for name, param in layer.named_parameters()}
    return output.detach(), hidden_states.grad.detach().clone(), param_grads


def test_apply_chunk_mbs_slices_packed_samples(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    model = _ToyRoot()
    layer = model.model.language_model.layers[0]
    apply_chunk_mbs(model, _config(chunk_mbs=2))

    assert getattr(layer, "_chunk_mbs_wrapped", False)
    assert not getattr(layer.proj, "_chunk_mbs_wrapped", False)
    assert not getattr(model.model.visual, "_chunk_mbs_wrapped", False)

    batch = {
        "cu_seq_lens_q": torch.tensor([0, 3, 5, 9, 10], dtype=torch.int32),
        "linear_attn_cu_seq_lens_q": torch.tensor([0, 3, 5, 10], dtype=torch.int32),
    }
    ranges = build_chunk_mbs_ranges(batch, _config(chunk_mbs=2))
    hidden_states = torch.arange(20, dtype=torch.float32).view(1, 10, 2).requires_grad_()
    position_ids = torch.arange(10).view(1, 10)
    position_embeddings = (torch.ones(1, 10, 2), torch.full((1, 10, 2), 2.0))
    attention_mask = torch.ones(1, 1, 10, 10)

    with chunk_mbs_context(ranges):
        output = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seq_lens_q=batch["cu_seq_lens_q"],
            cu_seq_lens_k=batch["cu_seq_lens_q"],
            max_length_q=4,
            max_length_k=4,
            linear_attn_cu_seq_lens_q=batch["linear_attn_cu_seq_lens_q"],
        )

    expected = hidden_states + position_ids.unsqueeze(-1).to(hidden_states.dtype) + 3.0
    assert torch.equal(output, expected)
    output.sum().backward()
    assert torch.equal(hidden_states.grad, torch.ones_like(hidden_states))
    assert [call["hidden_shape"] for call in layer.calls] == [(1, 5, 2), (1, 5, 2)]
    assert [call["position_ids"].tolist() for call in layer.calls] == [[[0, 1, 2, 3, 4]], [[5, 6, 7, 8, 9]]]
    assert [call["cu_seq_lens_q"].tolist() for call in layer.calls] == [[0, 3, 5], [0, 4, 5]]
    assert [call["cu_seq_lens_k"].tolist() for call in layer.calls] == [[0, 3, 5], [0, 4, 5]]
    assert [call["max_length_q"] for call in layer.calls] == [3, 4]
    assert [call["max_length_k"] for call in layer.calls] == [3, 4]
    assert [call["attention_shape"] for call in layer.calls] == [(1, 1, 5, 5), (1, 1, 5, 5)]
    assert [call["has_linear_attn_cu"] for call in layer.calls] == [True, True]
    assert [call["linear_attn_cu_seq_lens_q"].tolist() for call in layer.calls] == [[0, 3, 5], [0, 5]]


def test_apply_chunk_mbs_preserves_gradient_checkpointing_without_ranges(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    model = _ToyRoot()
    layer = model.model.language_model.layers[0]
    checkpoint_calls = []

    def checkpoint_func(function, *args):
        checkpoint_calls.append(None)
        return function(*args)

    layer.gradient_checkpointing = True
    layer._gradient_checkpointing_func = checkpoint_func
    apply_chunk_mbs(model, _config(chunk_mbs=2))

    hidden_states = torch.zeros(1, 2, 2)
    position_ids = torch.arange(2).view(1, 2)
    forward_kwargs = dict(
        position_embeddings=(torch.ones_like(hidden_states), torch.ones_like(hidden_states)),
        attention_mask=torch.ones(1, 1, 2, 2),
        position_ids=position_ids,
        cu_seq_lens_q=torch.tensor([0, 2], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, 2], dtype=torch.int32),
        max_length_q=2,
        max_length_k=2,
    )
    output = layer(hidden_states, **forward_kwargs)

    assert output.shape == hidden_states.shape
    assert checkpoint_calls == [None]
    assert layer.gradient_checkpointing

    model.gradient_checkpointing_disable()
    layer(hidden_states, **forward_kwargs)
    assert checkpoint_calls == [None]

    model.gradient_checkpointing_enable(checkpoint_func)
    layer.eval()
    layer(hidden_states, **forward_kwargs)
    assert checkpoint_calls == [None]

    layer.train()
    cache = object()
    layer(
        hidden_states,
        **forward_kwargs,
        use_cache=True,
        past_key_value=cache,
        past_key_values=cache,
        layer_past=cache,
    )
    assert checkpoint_calls == [None, None]
    assert layer.calls[-1]["use_cache"] is False
    assert layer.calls[-1]["past_key_value"] is None
    assert layer.calls[-1]["past_key_values"] is None
    assert layer.calls[-1]["layer_past"] is None
    assert layer.gradient_checkpointing


def test_slice_kwargs_slices_multimodal_position_ids():
    import veomni.distributed.chunk_mbs as chunk_mbs

    position_ids = torch.arange(16).view(4, 1, 4)
    seq_range = PackedSequenceRange(segment_start=0, segment_end=1, token_start=0, token_end=2, max_length=2)

    chunk_kwargs = chunk_mbs._slice_kwargs({"position_ids": position_ids}, seq_range, full_seq_len=4)

    assert torch.equal(chunk_kwargs["position_ids"], position_ids[:, :, :2])


def test_slice_kwargs_slices_one_dimensional_cache_position():
    import veomni.distributed.chunk_mbs as chunk_mbs

    cache_position = torch.arange(4)
    seq_range = PackedSequenceRange(segment_start=1, segment_end=2, token_start=2, token_end=4, max_length=2)

    chunk_kwargs = chunk_mbs._slice_kwargs({"cache_position": cache_position}, seq_range, full_seq_len=4)

    assert torch.equal(chunk_kwargs["cache_position"], cache_position[2:4])


def test_slice_kwargs_does_not_slice_attention_head_dimension():
    import veomni.distributed.chunk_mbs as chunk_mbs

    attention_mask = torch.arange(64).view(1, 4, 4, 4)
    seq_range = PackedSequenceRange(segment_start=0, segment_end=1, token_start=1, token_end=3, max_length=2)

    chunk_kwargs = chunk_mbs._slice_kwargs({"attention_mask": attention_mask}, seq_range, full_seq_len=4)

    assert torch.equal(chunk_kwargs["attention_mask"], attention_mask[:, :, 1:3, 1:3])


def test_slice_kwargs_slices_two_dimensional_square_attention_mask():
    import veomni.distributed.chunk_mbs as chunk_mbs

    attention_mask = torch.arange(16).view(4, 4)
    seq_range = PackedSequenceRange(segment_start=0, segment_end=1, token_start=1, token_end=3, max_length=2)

    chunk_kwargs = chunk_mbs._slice_kwargs({"attention_mask": attention_mask}, seq_range, full_seq_len=4)

    assert torch.equal(chunk_kwargs["attention_mask"], attention_mask[1:3, 1:3])


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_apply_chunk_mbs_matches_qwen3_vl_decoder_layer(monkeypatch, use_checkpoint):
    import veomni.distributed.chunk_mbs as chunk_mbs
    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
        Qwen3VLTextConfig,
        Qwen3VLTextDecoderLayer,
        Qwen3VLTextRotaryEmbedding,
    )

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    torch.manual_seed(0)
    config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000,
    )
    config._attn_implementation = "eager"

    baseline_layer = Qwen3VLTextDecoderLayer(config, layer_idx=0)
    chunked_root = _Qwen3VLRoot(copy.deepcopy(baseline_layer))
    chunked_layer = chunked_root.model.language_model.layers[0]
    checkpoint_calls = []
    baseline_hook_calls = []
    chunked_hook_calls = []

    baseline_layer.register_forward_pre_hook(lambda *_: baseline_hook_calls.append("pre"))
    baseline_layer.register_forward_hook(lambda *_: baseline_hook_calls.append("post"))
    chunked_layer.register_forward_pre_hook(lambda *_: chunked_hook_calls.append("pre"))
    chunked_layer.register_forward_hook(lambda *_: chunked_hook_calls.append("post"))

    if use_checkpoint:
        baseline_layer.gradient_checkpointing = True
        baseline_layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

        def checkpoint_func(function, *args):
            checkpoint_calls.append(None)
            return checkpoint(function, *args, use_reentrant=False)

        chunked_layer.gradient_checkpointing = True
        chunked_layer._gradient_checkpointing_func = checkpoint_func

    apply_chunk_mbs(chunked_root, _config(chunk_mbs=2))

    assert getattr(chunked_layer, "_chunk_mbs_wrapped", False)

    cu_seq_lens = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    ranges = build_chunk_mbs_ranges({"cu_seq_lens_q": cu_seq_lens}, _config(chunk_mbs=2))
    hidden_states = torch.randn(1, 9, config.hidden_size)
    text_position_ids = torch.arange(9, dtype=torch.long).view(1, 9)
    position_ids = torch.stack(
        (
            text_position_ids,
            text_position_ids,
            text_position_ids // 2,
            text_position_ids % 3,
        ),
        dim=0,
    )
    rotary_emb = Qwen3VLTextRotaryEmbedding(config)
    position_embeddings = rotary_emb(hidden_states, position_ids[1:])
    assert position_embeddings[0].shape == (1, 9, config.head_dim)
    attention_mask = _packed_causal_attention_mask(cu_seq_lens, hidden_states.dtype)
    max_length = int((cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item())

    baseline_output, baseline_hidden_grad, baseline_param_grads = _run_qwen3_vl_decoder_layer(
        baseline_layer,
        hidden_states,
        position_embeddings,
        attention_mask,
        position_ids[0],
        cu_seq_lens,
        max_length,
        ranges=None,
    )
    chunked_output, chunked_hidden_grad, chunked_param_grads = _run_qwen3_vl_decoder_layer(
        chunked_layer,
        hidden_states,
        position_embeddings,
        attention_mask,
        position_ids[0],
        cu_seq_lens,
        max_length,
        ranges=ranges,
    )

    if use_checkpoint:
        assert checkpoint_calls == [None]
        assert chunked_layer.gradient_checkpointing

    assert chunked_hook_calls == baseline_hook_calls

    torch.testing.assert_close(chunked_output, baseline_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(chunked_hidden_grad, baseline_hidden_grad, rtol=1e-5, atol=1e-5)
    assert chunked_param_grads.keys() == baseline_param_grads.keys()
    for name, baseline_grad in baseline_param_grads.items():
        torch.testing.assert_close(chunked_param_grads[name], baseline_grad, rtol=1e-5, atol=1e-5)


def test_replace_hidden_states_does_not_mutate_kwargs():
    import veomni.distributed.chunk_mbs as chunk_mbs

    hidden_states = torch.zeros(1, 4, 2)
    replacement = torch.ones(1, 2, 2)
    kwargs = {"hidden_states": hidden_states, "use_cache": False}

    args, chunk_kwargs = chunk_mbs._replace_hidden_states((), kwargs, replacement)

    assert args == ()
    assert chunk_kwargs["hidden_states"] is replacement
    assert kwargs["hidden_states"] is hidden_states


def test_apply_chunk_mbs_rejects_sequence_parallel(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=True, any_extra_parallel_enabled=False),
    )

    with pytest.raises(RuntimeError, match="sequence parallelism"):
        apply_chunk_mbs(_ToyRoot(), _config(chunk_mbs=2))


def test_apply_chunk_mbs_requires_decoder_layer_in_no_split_modules(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    model = nn.Module()
    model._no_split_modules = ["_ToyVisionBlock"]

    with pytest.raises(ValueError, match="model._no_split_modules"):
        apply_chunk_mbs(model, _config(chunk_mbs=2))


def test_apply_chunk_mbs_rejects_ambiguous_decoder_layer_classes(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    model = nn.Module()
    model._no_split_modules = ["_ToyDecoderLayer", "_AuxDecoderLayer"]
    model.decoder = _ToyDecoderLayer()
    model.aux_decoder = _AuxDecoderLayer()

    with pytest.raises(ValueError, match="exactly one decoder layer class"):
        apply_chunk_mbs(model, _config(chunk_mbs=2))
    assert not getattr(model.decoder, "_chunk_mbs_wrapped", False)
    assert not getattr(model.aux_decoder, "_chunk_mbs_wrapped", False)


def test_apply_chunk_mbs_rejects_ambiguous_decoder_stacks(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    model = nn.Module()
    model._no_split_modules = ["_ToyDecoderLayer"]
    model.decoder = nn.ModuleList([_ToyDecoderLayer()])
    model.aux_decoder = nn.ModuleList([_ToyDecoderLayer()])

    with pytest.raises(ValueError, match="exactly one decoder stack"):
        apply_chunk_mbs(model, _config(chunk_mbs=2))
    assert not getattr(model.decoder[0], "_chunk_mbs_wrapped", False)
    assert not getattr(model.aux_decoder[0], "_chunk_mbs_wrapped", False)


def test_apply_chunk_mbs_rejects_incompatible_decoder_layer(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )

    model = nn.Module()
    model._no_split_modules = ["_PlainDecoderLayer"]
    model.decoder = _PlainDecoderLayer()

    with pytest.raises(TypeError, match="GradientCheckpointingLayer"):
        apply_chunk_mbs(model, _config(chunk_mbs=2))
    assert not getattr(model.decoder, "_chunk_mbs_wrapped", False)


def test_build_parallelize_model_applies_chunk_mbs_before_fsdp2(monkeypatch):
    import veomni.distributed.chunk_mbs as chunk_mbs
    import veomni.distributed.torch_parallelize as torch_parallelize

    monkeypatch.setattr(
        chunk_mbs,
        "get_parallel_state",
        lambda: types.SimpleNamespace(sp_enabled=False, any_extra_parallel_enabled=False),
    )
    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: types.SimpleNamespace(fsdp_enabled=True, tp_enabled=False, dp_mode="fsdp2"),
    )

    model = _ToyRoot()

    def fake_parallelize_model_fsdp2(model, **kwargs):
        layer = model.model.language_model.layers[0]
        assert getattr(layer, "_chunk_mbs_wrapped", False)
        assert layer.gradient_checkpointing
        assert getattr(layer._gradient_checkpointing_func, "_chunk_mbs_wrapped", False)
        return model

    monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", fake_parallelize_model_fsdp2)

    result = torch_parallelize.build_parallelize_model(
        model,
        mixed_precision=MixedPrecisionConfig(enable=False),
        chunk_mbs_config=_config(chunk_mbs=2),
    )

    assert result is model

    cu_seq_lens = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    ranges = build_chunk_mbs_ranges({"cu_seq_lens_q": cu_seq_lens}, _config(chunk_mbs=2))
    hidden_states = torch.zeros(1, 4, 2, requires_grad=True)
    with chunk_mbs_context(ranges):
        output = result.model.language_model.layers[0](
            hidden_states,
            position_embeddings=(torch.ones_like(hidden_states), torch.ones_like(hidden_states)),
            attention_mask=torch.ones(1, 1, 4, 4),
            position_ids=torch.arange(4).view(1, 4),
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=1,
            max_length_k=1,
        )
        output.sum().backward()

    assert torch.equal(hidden_states.grad, torch.ones_like(hidden_states))
