# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.training_utils import make_eager_ops_config
from veomni.models.configuration_base import as_veomni_config
from veomni.models.transformers.minimax_m3_vl import modeling_minimax_m3_vl as modeling


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _upstream_text_model(model):
    """Unpatched Transformers text model with the same weights, run with eager attention as the parity reference."""
    upstream = importlib.import_module("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    config = copy.deepcopy(model.config)
    config._attn_implementation = "eager"
    reference = upstream.MiniMaxM3VLTextModel(config).eval()
    reference.load_state_dict(model.state_dict())
    return reference


class _RecordingVisionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def forward(self, hidden_states, attention_mask, **kwargs):
        self.kwargs = {"attention_mask": attention_mask, **kwargs}
        return hidden_states


def _text_config(modeling, **overrides):
    kwargs = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=6,
        dense_intermediate_size=12,
        shared_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        index_n_heads=1,
        head_dim=4,
        rotary_dim=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["minimax_m3_sparse"],
        mlp_layer_types=["dense"],
        bos_token_id=1,
        eos_token_id=2,
    )
    kwargs.update(overrides)
    config = modeling.MiniMaxM3VLTextConfig(**kwargs)
    as_veomni_config(config, validate=False)._set_ops_implementation(make_eager_ops_config())
    return config


def _single_layer_sparse_text_model(modeling):
    config = _text_config(modeling, index_block_size=2, index_topk_blocks=2)
    return modeling.MiniMaxM3VLTextModel(config).eval()


class _VarlenReferenceAttention:
    """CPU stand-in for varlen FlashAttention: causal SDPA per packed segment, ``[1, H, S, D]`` in, ``[1, S, H, D]`` out."""

    def __init__(self):
        self.calls = 0

    def __call__(self, module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        assert attention_mask is None
        self.calls += 1
        bounds = kwargs["cu_seq_lens_q"].tolist()
        outputs = [
            F.scaled_dot_product_attention(
                query[:, :, start:end],
                key[:, :, start:end],
                value[:, :, start:end],
                dropout_p=dropout,
                is_causal=True,
                scale=scaling,
                enable_gqa=True,
            )
            for start, end in zip(bounds[:-1], bounds[1:])
        ]
        return torch.cat(outputs, dim=2).transpose(1, 2), None


@pytest.mark.parametrize("layer_type", ["minimax_m3_sparse", "full_attention"], ids=["sparse", "full"])
def test_minimax_m3_vl_packed_attention_matches_separate_samples(layer_type):
    """Packed samples match the upstream model run per sample, in forward and backward."""
    model = modeling.MiniMaxM3VLTextModel(_text_config(modeling, layer_types=[layer_type])).eval()
    reference = _upstream_text_model(model)
    attention = model.layers[0].self_attn
    if layer_type == "full_attention":
        # Dense layers dispatch the configured varlen backend; FlashAttention needs
        # an accelerator, so a per-segment SDPA stands in for it here.
        assert attention.veomni_attn.op == "attention"
        attention.veomni_attn = _VarlenReferenceAttention()
    model.config._attn_implementation = "veomni_flash_attention_2"
    hidden_size = model.config.hidden_size
    first = torch.randn(1, 2, hidden_size, requires_grad=True)
    second = torch.randn(1, 3, hidden_size, requires_grad=True)
    packed = torch.cat((first, second), dim=1).detach().requires_grad_(True)

    actual = model(
        inputs_embeds=packed,
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 0, 1, 2]]),
        cu_seq_lens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, 2, 5], dtype=torch.int32),
        max_length_q=3,
        max_length_k=3,
        use_cache=False,
    ).last_hidden_state
    expected = torch.cat(
        [
            reference(
                inputs_embeds=sample,
                attention_mask=torch.ones(1, sample.shape[1], dtype=torch.long),
                position_ids=torch.arange(sample.shape[1]).unsqueeze(0),
                use_cache=False,
            ).last_hidden_state
            for sample in (first, second)
        ],
        dim=1,
    )
    torch.testing.assert_close(actual, expected)
    if layer_type == "full_attention":
        assert attention.veomni_attn.calls == 1

    (packed_grad,) = torch.autograd.grad(actual.square().sum(), packed)
    first_grad, second_grad = torch.autograd.grad(expected.square().sum(), (first, second))
    torch.testing.assert_close(packed_grad, torch.cat((first_grad, second_grad), dim=1))


def test_minimax_m3_vl_dense_eager_packed_matches_separate_samples():
    """veomni_eager causal-masks and splits packed cu_seqlens, so it matches per-sample eager."""
    model = modeling.MiniMaxM3VLTextModel(_text_config(modeling, layer_types=["full_attention"])).eval()
    model.config._attn_implementation = "veomni_eager"
    reference = _upstream_text_model(model)
    hidden_size = model.config.hidden_size
    first = torch.randn(1, 2, hidden_size)
    second = torch.randn(1, 3, hidden_size)
    packed = torch.cat((first, second), dim=1)

    with torch.no_grad():
        actual = model(
            inputs_embeds=packed,
            position_ids=torch.tensor([[0, 1, 0, 1, 2]]),
            cu_seq_lens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            cu_seq_lens_k=torch.tensor([0, 2, 5], dtype=torch.int32),
            max_length_q=3,
            max_length_k=3,
            use_cache=False,
        ).last_hidden_state
        expected = torch.cat(
            [
                reference(
                    inputs_embeds=sample,
                    attention_mask=torch.ones(1, sample.shape[1], dtype=torch.long),
                    position_ids=torch.arange(sample.shape[1]).unsqueeze(0),
                    use_cache=False,
                ).last_hidden_state
                for sample in (first, second)
            ],
            dim=1,
        )

    torch.testing.assert_close(actual, expected)


def test_minimax_m3_vl_sparse_layers_never_bind_the_standard_attention():
    """Each layer binds only the operators its layer type calls."""
    model = modeling.MiniMaxM3VLTextModel(
        _text_config(
            modeling,
            num_hidden_layers=2,
            layer_types=["minimax_m3_sparse", "full_attention"],
            mlp_layer_types=["dense", "dense"],
        )
    )
    sparse, dense = (layer.self_attn for layer in model.layers)
    assert (sparse.veomni_msa_indexer.op, sparse.veomni_msa_attention.op) == (
        "minimax_sparse_indexer",
        "minimax_sparse_attention",
    )
    assert sparse.veomni_msa_attention.impl == "eager"
    assert not hasattr(sparse, "veomni_attn")
    assert dense.veomni_attn.op == "attention"
    assert not hasattr(dense, "veomni_msa_attention")


@pytest.mark.parametrize(
    ("attn_implementation", "cu_seq_lens", "ulysses_enabled", "rejected"),
    [
        ("eager", [0, 2, 5], False, False),
        ("veomni_eager", [0, 2, 5], True, False),
        ("eager", [0, 5], True, True),
        ("flash_attention_2", [0, 5], True, True),
        ("eager", [0, 5], False, False),
        ("sdpa", [0, 5], False, True),
        ("flash_attention_2", [0, 5], False, False),
        ("veomni_flash_attention_2", [0, 2, 5], True, False),
    ],
    ids=[
        "eager_packed",
        "veomni_eager_ulysses",
        "eager_ulysses",
        "non_sp_varlen_ulysses",
        "eager_single_sequence",
        "sdpa_single_sequence",
        "varlen_single_sequence",
        "sp_varlen_ulysses",
    ],
)
def test_minimax_m3_vl_dense_language_attention_requires_varlen_backend(
    attn_implementation, cu_seq_lens, ulysses_enabled, rejected, monkeypatch
):
    """Without a mask, non-varlen dense attention leaks future tokens or mixes packed samples; non-VeOmni backends skip the Ulysses exchange."""
    model = modeling.MiniMaxM3VLTextModel(_text_config(modeling, layer_types=["full_attention"])).eval()
    model.config._attn_implementation = attn_implementation
    model.layers[0].self_attn.veomni_attn = _VarlenReferenceAttention()
    # SP is off so the dense op runs locally; only the backend check sees Ulysses.
    monkeypatch.setattr(
        modeling,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_enabled=False, ulysses_enabled=ulysses_enabled),
    )
    cu_seq_lens = torch.tensor(cu_seq_lens, dtype=torch.int32)
    forward_kwargs = dict(
        inputs_embeds=torch.randn(1, 5, model.config.hidden_size),
        cu_seq_lens_q=cu_seq_lens,
        cu_seq_lens_k=cu_seq_lens,
        max_length_q=5,
        max_length_k=5,
    )

    if rejected:
        with pytest.raises(ValueError, match="dense language attention needs a varlen FlashAttention backend"):
            model(**forward_kwargs)
    else:
        model(**forward_kwargs)


@pytest.mark.parametrize("with_attention_mask", [False, True], ids=["no_mask", "all_ones_mask"])
def test_minimax_m3_vl_text_model_runs_unpacked_single_sequence_as_one_segment(with_attention_mask):
    """Without varlen kwargs a lone unpadded sequence runs as a one-segment pack and matches upstream."""
    model = _single_layer_sparse_text_model(modeling)
    hidden_states = torch.randn(1, 7, model.config.hidden_size)
    position_ids = torch.arange(7).unsqueeze(0)

    with torch.no_grad():
        attention_mask = torch.ones(1, 7, dtype=torch.long) if with_attention_mask else None
        output = model(inputs_embeds=hidden_states, attention_mask=attention_mask)
        packed = model(
            inputs_embeds=hidden_states,
            position_ids=position_ids,
            cu_seq_lens_q=torch.tensor([0, 7], dtype=torch.int32),
            cu_seq_lens_k=torch.tensor([0, 7], dtype=torch.int32),
            max_length_q=7,
            max_length_k=7,
        )
        expected = _upstream_text_model(model)(
            inputs_embeds=hidden_states,
            attention_mask=torch.ones(1, 7, dtype=torch.long),
            position_ids=position_ids,
            use_cache=False,
        )

    assert output.past_key_values is None
    torch.testing.assert_close(output.last_hidden_state, packed.last_hidden_state, rtol=0, atol=0)
    torch.testing.assert_close(output.last_hidden_state, expected.last_hidden_state)


def test_minimax_m3_vl_indexer_is_frozen_and_stays_out_of_autograd():
    """Top-k block selection is not differentiable: the indexer is frozen and builds no dead autograd graph."""
    model = _single_layer_sparse_text_model(modeling)
    attention = model.layers[0].self_attn
    assert not any(param.requires_grad for param in attention.indexer.parameters())
    assert all(param.requires_grad for name, param in model.named_parameters() if ".indexer." not in name)

    hidden_states = torch.randn(1, 5, model.config.hidden_size, requires_grad=True)
    position_embeddings = model.rotary_emb(hidden_states, torch.arange(5).unsqueeze(0))

    index_query, index_key = modeling._minimax_m3_indexer_project(
        attention.indexer, hidden_states, position_embeddings
    )
    assert not index_query.requires_grad
    assert not index_key.requires_grad

    output = model(
        inputs_embeds=hidden_states,
        position_ids=torch.arange(5).unsqueeze(0),
        cu_seq_lens_q=torch.tensor([0, 5], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, 5], dtype=torch.int32),
        max_length_q=5,
        max_length_k=5,
    ).last_hidden_state
    output.square().sum().backward()
    assert all(param.grad is None for param in attention.indexer.parameters())
    assert attention.q_proj.weight.grad is not None


@pytest.mark.parametrize(
    ("forward_kwargs", "message"),
    [
        ({"inputs_embeds": torch.randn(2, 4, 8)}, "padded batches are not supported"),
        ({"inputs_embeds": torch.randn(1, 4, 8), "past_key_values": object()}, "does not support KV cache"),
        (
            {
                "inputs_embeds": torch.randn(2, 4, 8),
                "cu_seq_lens_q": torch.tensor([0, 4, 8], dtype=torch.int32),
                "cu_seq_lens_k": torch.tensor([0, 4, 8], dtype=torch.int32),
                "max_length_q": 4,
                "max_length_k": 4,
            },
            "must have batch size 1",
        ),
        (
            {"inputs_embeds": torch.randn(1, 4, 8), "attention_mask": torch.tensor([[0, 0, 1, 1]])},
            "only accepts an unpadded sequence",
        ),
    ],
    ids=["padded_batch", "kv_cache", "packed_batch_gt_1", "padded_single_sequence"],
)
def test_minimax_m3_vl_text_model_rejects_inputs_it_cannot_pack(forward_kwargs, message):
    model = _single_layer_sparse_text_model(modeling)

    with pytest.raises(ValueError, match=message):
        model(**forward_kwargs)


def test_minimax_m3_vl_text_model_rejects_unpacked_input_under_sequence_parallelism(monkeypatch):
    """A local SP shard cannot describe the global sequence, so the collator's metadata is mandatory."""
    model = _single_layer_sparse_text_model(modeling)
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=True))

    with pytest.raises(ValueError, match="requires packed inputs"):
        model(inputs_embeds=torch.randn(1, 4, model.config.hidden_size))


def test_minimax_m3_vl_metadata_preserves_visual_batch_boundaries_and_isolates_sp_padding():
    batch = {
        "image_grid_thw": torch.tensor([[1, 2, 2], [2, 1, 2]], dtype=torch.long),
        "video_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
    }

    modeling.collate_multimodal_metadata(
        batch,
        {"pixel_values": 4, "pixel_values_videos": 8},
    )

    metadata = batch["multimodal_metadata"]
    assert metadata["vit_image_cu_seqlens"].tolist() == [0, 4, 8, 12]
    assert metadata["vit_image_max_seqlen"] == 4
    assert metadata["vit_video_cu_seqlens"].tolist() == [0, 4, 12]
    assert metadata["vit_video_max_seqlen"] == 8


@pytest.mark.parametrize("merge_size", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_minimax_m3_vl_vision_rope_grid_list_matches_upstream(merge_size, dtype):
    """Preserve temporal order and spatial merge blocks across mixed image/video grids."""
    upstream = importlib.import_module("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    grid_list = [[1, 2 * merge_size, 3 * merge_size], [3, 3 * merge_size, 2 * merge_size]]
    grid = torch.tensor(grid_list, dtype=torch.long)
    reference = upstream.MiniMaxM3VL3DRotaryEmbedding(80, spatial_merge_size=merge_size)
    rope = modeling.MiniMaxM3VL3DRotaryEmbedding(80, spatial_merge_size=merge_size)
    expected = reference(grid, device="cpu", dtype=dtype)
    actual = rope(grid, device="cpu", dtype=dtype, grid_thw_list=grid_list)
    fallback = rope(grid, device="cpu", dtype=dtype)
    for fast, slow, ref in zip(actual, fallback, expected):
        torch.testing.assert_close(fast, ref, rtol=0, atol=0)
        torch.testing.assert_close(slow, ref, rtol=0, atol=0)


def test_minimax_m3_vl_vision_rope_preserves_precomputed_positions():
    upstream = importlib.import_module("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    grid_list = [[2, 4, 6]]
    grid = torch.tensor(grid_list, dtype=torch.long)
    positions = upstream.get_vision_position_ids(grid, 2, include_temporal=True).flip(0)
    reference = upstream.MiniMaxM3VL3DRotaryEmbedding(80, spatial_merge_size=2)
    rope = modeling.MiniMaxM3VL3DRotaryEmbedding(80, spatial_merge_size=2)
    kwargs = {"position_ids": positions}
    expected = reference(grid, "cpu", torch.float32, kwargs={"position_ids": positions})
    actual = rope(grid, "cpu", torch.float32, kwargs=kwargs, grid_thw_list=grid_list)
    assert "position_ids" not in kwargs
    for result, ref in zip(actual, expected):
        torch.testing.assert_close(result, ref, rtol=0, atol=0)


def _tiny_vision_model(modeling, attn_implementation):
    config = modeling.MiniMaxM3VLVisionConfig(
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
    )
    as_veomni_config(config, validate=False)._set_ops_implementation(make_eager_ops_config())
    vision_model = modeling.MiniMaxM3VLVisionModel(config)
    vision_model.config._attn_implementation = attn_implementation
    vision_model.layers = torch.nn.ModuleList([_RecordingVisionLayer()])
    return vision_model


_TWO_MEDIA = [[1, 2, 2], [1, 2, 2]]
_ONE_MEDIUM = [[1, 2, 2]]


@pytest.mark.parametrize(
    ("attn_implementation", "grid_thw_list", "ulysses_enabled", "rejected"),
    [
        ("eager", _TWO_MEDIA, False, False),
        ("veomni_eager", _TWO_MEDIA, True, False),
        ("eager", _ONE_MEDIUM, True, True),
        ("flash_attention_2", _ONE_MEDIUM, True, True),
        ("eager", _ONE_MEDIUM, False, False),
        ("veomni_flash_attention_2", _TWO_MEDIA, False, False),
        ("veomni_flash_attention_2", _TWO_MEDIA, True, False),
    ],
    ids=[
        "eager_multiple_media",
        "veomni_eager_ulysses",
        "eager_ulysses",
        "non_sp_varlen_ulysses",
        "eager_single_medium",
        "sp_varlen_multiple_media",
        "sp_varlen_ulysses",
    ],
)
def test_minimax_m3_vl_vision_requires_varlen_attention_to_isolate_media(
    attn_implementation, grid_thw_list, ulysses_enabled, rejected, monkeypatch
):
    """Without a mask, non-varlen vision attention would mix media samples; non-VeOmni backends skip the Ulysses exchange."""
    vision_model = _tiny_vision_model(modeling, attn_implementation)
    monkeypatch.setattr(
        modeling,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_enabled=False, ulysses_enabled=ulysses_enabled),
    )
    num_patches = sum(t * h * w for t, h, w in grid_thw_list)
    forward_kwargs = dict(
        pixel_values=torch.randn(num_patches, 3 * 2 * 2),
        grid_thw=torch.tensor(grid_thw_list, dtype=torch.long),
        vit_metadata={"grid_thw_list": grid_thw_list},
    )

    if rejected:
        with pytest.raises(ValueError, match="needs a varlen FlashAttention backend"):
            vision_model(**forward_kwargs)
    else:
        vision_model(**forward_kwargs)


def test_minimax_m3_vl_lists_every_veomni_varlen_backend_as_flash_compatible():
    """HF replaces a flash backend missing from this list with its first entry, MiniMax's own Hub kernel."""
    compatible = modeling.MiniMaxM3VLPreTrainedModel._compatible_flash_implementations
    sp_backends = [
        "veomni_flash_attention_2",
        "veomni_flash_attention_2_hub",
        "veomni_flash_attention_3",
        "veomni_flash_attention_3_hub",
        "veomni_flash_attention_4",
    ]
    assert set(sp_backends) <= set(compatible)
    for backend in sp_backends:
        modeling._require_varlen_attention(
            backend, num_segments=2, ulysses_enabled=True, attention_name="test", causal=True
        )
        modeling._require_varlen_attention(
            backend.removeprefix("veomni_"),
            num_segments=2,
            ulysses_enabled=False,
            attention_name="test",
            causal=True,
        )


@pytest.mark.parametrize("precompute_attention_metadata", [False, True])
def test_minimax_m3_vl_vision_forward_threads_global_sp_metadata(
    precompute_attention_metadata,
    monkeypatch,
):
    config = modeling.MiniMaxM3VLVisionConfig(
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
    )
    as_veomni_config(config, validate=False)._set_ops_implementation(make_eager_ops_config())
    vision_model = modeling.MiniMaxM3VLVisionModel(config)
    vision_model.config._attn_implementation = "veomni_flash_attention_2"
    recording_layer = _RecordingVisionLayer()
    vision_model.layers = torch.nn.ModuleList([recording_layer])

    parallel_state = SimpleNamespace(
        sp_enabled=True,
        sp_size=2,
        ulysses_enabled=True,
    )
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: parallel_state)

    def fake_sp_pad_and_slice(tensor, dim, pad_value, pad_scale):
        assert dim == 0
        assert pad_value == 0
        assert pad_scale == config.spatial_merge_size**2
        alignment = parallel_state.sp_size * pad_scale
        pad_size = (-tensor.shape[dim]) % alignment
        if pad_size > 0:
            pad_shape = list(tensor.shape)
            pad_shape[dim] = pad_size
            pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat((tensor, pad), dim=dim)
        return tensor.narrow(dim, 0, tensor.shape[dim] // parallel_state.sp_size)

    monkeypatch.setattr(modeling, "sp_pad_and_slice", fake_sp_pad_and_slice)

    grid_thw = torch.tensor([[1, 2, 2], [2, 2, 2]], dtype=torch.long)
    vit_metadata = {"grid_thw_list": [[1, 2, 2], [2, 2, 2]]}
    if precompute_attention_metadata:
        vit_metadata.update(
            {
                "cu_seqlens": torch.tensor([0, 4, 12, 16], dtype=torch.int32),
                "max_seqlen": 8,
            }
        )

    pixel_values = torch.randn(8, config.num_channels * config.temporal_patch_size * config.patch_size**2)
    output = vision_model(
        pixel_values=pixel_values,
        grid_thw=grid_thw,
        vit_metadata=vit_metadata,
    )

    assert output.last_hidden_state.shape == (1, 8, config.hidden_size)
    assert recording_layer.kwargs["attention_mask"] is None
    assert recording_layer.kwargs["cu_seq_lens_q"].tolist() == [0, 4, 12, 16]
    assert recording_layer.kwargs["cu_seq_lens_k"].tolist() == [0, 4, 12, 16]
    assert recording_layer.kwargs["max_length_q"] == 8
    assert recording_layer.kwargs["max_length_k"] == 8
    cos, sin = recording_layer.kwargs["position_embeddings"]
    assert cos.shape[0] == sin.shape[0] == 8


def test_minimax_m3_vl_placeholder_mask_fallback_gathers_input_ids_under_sp(monkeypatch):
    """Without precomputed masks, SP must derive them from the full sequence, not this rank's slice."""
    from veomni.models.registry import get_model_config

    config = get_model_config("./tests/toy_config/minimax_m3_vl_toy", ops_implementation=make_eager_ops_config())
    model = modeling.MiniMaxM3SparseForConditionalGeneration(config).model.eval()
    sp_size, local_len = 2, 4
    full_ids = torch.randint(0, 100, (1, sp_size * local_len))
    full_ids[0, [1, 6]] = config.image_token_id
    local_ids = full_ids[:, :local_len]

    def fake_all_gather(tensor_list, tensor, group):
        for rank, chunk in enumerate(full_ids.split(local_len, dim=1)):
            tensor_list[rank].copy_(chunk)

    monkeypatch.setattr(
        modeling,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_enabled=True, sp_size=sp_size, sp_group=None, fsdp_enabled=False),
    )
    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(
        modeling, "gather_outputs", lambda x, gather_dim, group: model.get_input_embeddings()(full_ids)
    )
    monkeypatch.setattr(modeling, "slice_input_tensor", lambda x, dim, group: x.narrow(dim, 0, local_len))
    seen = {}
    original_mask = model.get_placeholder_mask

    def spy_mask(input_ids, inputs_embeds, **kwargs):
        seen["input_ids"] = input_ids
        return original_mask(input_ids, inputs_embeds, **kwargs)

    monkeypatch.setattr(model, "get_placeholder_mask", spy_mask)

    embed_tokens = model.get_input_embeddings()

    class _LanguageModelStub(torch.nn.Module):
        def get_input_embeddings(self):
            return embed_tokens

        def forward(self, inputs_embeds, **kwargs):
            seen["inputs_embeds"] = inputs_embeds
            return SimpleNamespace(last_hidden_state=inputs_embeds, past_key_values=None)

    model.language_model = _LanguageModelStub()
    with torch.no_grad():
        model(input_ids=local_ids)

    assert torch.equal(seen["input_ids"], full_ids)
    assert seen["inputs_embeds"].shape[1] == local_len


@pytest.mark.parametrize("with_metadata", [False, True], ids=["grid_tensor", "grid_list"])
@pytest.mark.parametrize("modality", ["image", "video"])
def test_minimax_m3_vl_sp_drops_vision_padding_features(modality, with_metadata, monkeypatch):
    """SP=2 with a 1x2x2 grid gathers two features (one from the padding) for a single placeholder."""
    from veomni.models.registry import get_model_config

    config = get_model_config("./tests/toy_config/minimax_m3_vl_toy", ops_implementation=make_eager_ops_config())
    model = modeling.MiniMaxM3SparseForConditionalGeneration(config).model.eval()
    assert model.vision_tower.config.spatial_merge_size == 2
    sp_size, local_len = 2, 4
    token_id = config.image_token_id if modality == "image" else config.video_token_id
    full_ids = torch.randint(0, 100, (1, sp_size * local_len))
    full_ids[0, 5] = token_id
    embed_tokens = model.get_input_embeddings()
    hidden_size = embed_tokens.weight.shape[1]
    real_feature, padding_feature = torch.randn(1, hidden_size), torch.randn(1, hidden_size)

    def fake_all_gather(tensor_list, tensor, group):
        for rank, chunk in enumerate(full_ids.split(local_len, dim=1)):
            tensor_list[rank].copy_(chunk)

    def fake_gather_outputs(x, gather_dim, group):
        if gather_dim == 1:
            return embed_tokens(full_ids)
        # Rank 0 holds the four real patch rows, rank 1 the SP-padding rows.
        return torch.cat([real_feature, padding_feature])

    monkeypatch.setattr(
        modeling,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_enabled=True, sp_size=sp_size, sp_group=None, fsdp_enabled=False),
    )
    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(modeling, "gather_outputs", fake_gather_outputs)
    monkeypatch.setattr(modeling, "slice_input_tensor", lambda x, dim, group: x)
    local_features = SimpleNamespace(pooler_output=real_feature)
    monkeypatch.setattr(model, f"get_{modality}_features", lambda *args, **kwargs: local_features)

    seen = {}

    class _LanguageModelStub(torch.nn.Module):
        def get_input_embeddings(self):
            return embed_tokens

        def forward(self, inputs_embeds, **kwargs):
            seen["inputs_embeds"] = inputs_embeds
            return SimpleNamespace(last_hidden_state=inputs_embeds, past_key_values=None)

    model.language_model = _LanguageModelStub()
    grid_thw = torch.tensor([[1, 2, 2]])
    pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
    metadata = {f"{modality}_grid_thw_list": [[1, 2, 2]]} if with_metadata else None
    with torch.no_grad():
        output = model(
            input_ids=full_ids[:, :local_len],
            **{pixel_key: torch.zeros(2, 1), f"{modality}_grid_thw": grid_thw},
            multimodal_metadata=metadata,
        )

    torch.testing.assert_close(seen["inputs_embeds"][0, 5], real_feature[0])
    torch.testing.assert_close(getattr(output, f"{modality}_hidden_states"), real_feature)


def test_minimax_m3_vl_parallel_plan_targets_the_expert_parameters():
    """Every EP plan pattern must match the stacked expert weights of the multimodal model."""
    import fnmatch

    from veomni.models.registry import get_model_config

    config = get_model_config("./tests/toy_config/minimax_m3_vl_toy", ops_implementation=make_eager_ops_config())
    with torch.device("meta"):
        model = modeling.MiniMaxM3SparseForConditionalGeneration(config)
    parameter_names = [name for name, _ in model.named_parameters()]

    plan = modeling.MiniMaxM3SparseForConditionalGeneration.get_parallel_plan(model)
    patterns = set(plan.extra_parallel_plan["ep"])
    assert patterns == {
        "model.language_model.layers.*.mlp.experts.gate_up_proj",
        "model.language_model.layers.*.mlp.experts.down_proj",
    }
    for pattern in patterns:
        assert fnmatch.filter(parameter_names, pattern), pattern


def test_minimax_m3_vl_experts_pass_swiglu_config_to_op(monkeypatch):
    config = SimpleNamespace(
        num_local_experts=2,
        hidden_size=4,
        intermediate_size=3,
        swiglu_limit=0.7,
        swiglu_alpha=1.25,
        ops_config=make_eager_ops_config(),
    )
    experts = modeling.MiniMaxM3VLExperts(config)
    op = Mock()
    monkeypatch.setattr(experts, "veomni_moe", op)
    experts(torch.zeros(1, 4), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1))

    op.assert_called_once()
    assert op.call_args.kwargs["swiglu_alpha"] == config.swiglu_alpha
    assert op.call_args.kwargs["swiglu_limit"] == config.swiglu_limit


def test_minimax_m3_vl_rms_norm_passes_fp32_weight_to_op(monkeypatch):
    norm = modeling.MiniMaxM3VLRMSNorm(8, eps=1e-6).to(torch.bfloat16)
    op = Mock()
    monkeypatch.setattr(norm, "veomni_rms_norm", op)
    norm(torch.ones(2, 3, 8, dtype=torch.bfloat16))

    op.assert_called_once()
    assert op.call_args.args[1].dtype == torch.float32
