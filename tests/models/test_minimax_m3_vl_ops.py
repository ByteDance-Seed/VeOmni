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

import pytest
import torch


_MODELING_MODULES = (
    "veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_gpu",
    "veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_npu",
)


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
    return modeling.MiniMaxM3VLTextConfig(**kwargs)


def _single_layer_sparse_text_model(modeling):
    config = _text_config(modeling, index_block_size=2, index_topk_blocks=2)
    return modeling.MiniMaxM3VLTextModel(config).eval()


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
@pytest.mark.parametrize("layer_type", ["minimax_m3_sparse", "full_attention"], ids=["sparse", "full"])
def test_minimax_m3_vl_packed_attention_matches_separate_samples(module_name, layer_type):
    """Packed samples match the upstream model run per sample, in forward and backward."""
    modeling = importlib.import_module(module_name)
    model = modeling.MiniMaxM3VLTextModel(_text_config(modeling, layer_types=[layer_type])).eval()
    reference = _upstream_text_model(model)
    # The shared config selects the ViT backend only; language attention must ignore it.
    model.config._attn_implementation = "veomni_flash_attention_2_with_sp"
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

    (packed_grad,) = torch.autograd.grad(actual.square().sum(), packed)
    first_grad, second_grad = torch.autograd.grad(expected.square().sum(), (first, second))
    torch.testing.assert_close(packed_grad, torch.cat((first_grad, second_grad), dim=1))


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_text_model_runs_unpacked_single_sequence_as_one_segment(module_name):
    """Without varlen kwargs a lone unpadded sequence runs as a one-segment pack and matches upstream."""
    modeling = importlib.import_module(module_name)
    model = _single_layer_sparse_text_model(modeling)
    hidden_states = torch.randn(1, 7, model.config.hidden_size)
    position_ids = torch.arange(7).unsqueeze(0)

    with torch.no_grad():
        output = model(inputs_embeds=hidden_states, attention_mask=torch.ones(1, 7, dtype=torch.long))
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


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_indexer_projection_stays_out_of_autograd(module_name):
    """Top-k block selection is not differentiable, so the indexer must not build a dead autograd graph."""
    modeling = importlib.import_module(module_name)
    model = _single_layer_sparse_text_model(modeling)
    attention = model.layers[0].self_attn
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


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
@pytest.mark.parametrize(
    ("forward_kwargs", "message"),
    [
        ({"inputs_embeds": torch.randn(2, 4, 8)}, "padded batches are not supported"),
        ({"inputs_embeds": torch.randn(1, 4, 8), "past_key_values": object()}, "does not support KV cache"),
    ],
    ids=["padded_batch", "kv_cache"],
)
def test_minimax_m3_vl_text_model_rejects_inputs_it_cannot_pack(module_name, forward_kwargs, message):
    modeling = importlib.import_module(module_name)
    model = _single_layer_sparse_text_model(modeling)

    with pytest.raises(ValueError, match=message):
        model(**forward_kwargs)


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_text_model_rejects_unpacked_input_under_sequence_parallelism(module_name, monkeypatch):
    """A local SP shard cannot describe the global sequence, so the collator's metadata is mandatory."""
    modeling = importlib.import_module(module_name)
    model = _single_layer_sparse_text_model(modeling)
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=False, sp_enabled=True))

    with pytest.raises(ValueError, match="requires packed inputs"):
        model(inputs_embeds=torch.randn(1, 4, model.config.hidden_size))


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_metadata_preserves_visual_batch_boundaries_and_isolates_sp_padding(module_name):
    modeling = importlib.import_module(module_name)
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


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
@pytest.mark.parametrize("precompute_attention_metadata", [False, True])
def test_minimax_m3_vl_vision_forward_threads_global_sp_metadata(
    module_name,
    precompute_attention_metadata,
    monkeypatch,
):
    modeling = importlib.import_module(module_name)
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
    vision_model = modeling.MiniMaxM3VLVisionModel(config)
    vision_model.config._attn_implementation = "veomni_flash_attention_2_with_sp"
    recording_layer = _RecordingVisionLayer()
    vision_model.layers = torch.nn.ModuleList([recording_layer])

    parallel_state = SimpleNamespace(
        sp_enabled=True,
        sp_size=2,
        ulysses_enabled=True,
        cp_enabled=False,
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


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_placeholder_mask_fallback_gathers_input_ids_under_sp(module_name, monkeypatch):
    """Without precomputed masks, SP must derive them from the full sequence, not this rank's slice."""
    from veomni.models.loader import get_model_config

    modeling = importlib.import_module(module_name)
    config = get_model_config("./tests/toy_config/minimax_m3_vl_toy")
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


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_parallel_plan_targets_the_expert_parameters(module_name):
    """Every EP plan pattern must match the stacked expert weights of the multimodal model."""
    import fnmatch

    from veomni.models.loader import get_model_config

    modeling = importlib.import_module(module_name)
    config = get_model_config("./tests/toy_config/minimax_m3_vl_toy")
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


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_experts_preserve_swiglu_oai_eager_semantics(module_name, monkeypatch):
    modeling = importlib.import_module(module_name)
    config = SimpleNamespace(
        num_local_experts=2,
        hidden_size=4,
        intermediate_size=3,
        swiglu_limit=0.7,
        swiglu_alpha=1.702,
    )
    experts = modeling.MiniMaxM3VLExperts(config)
    torch.manual_seed(0)
    experts.gate_up_proj.data.normal_(mean=0.0, std=0.1)
    experts.down_proj.data.normal_(mean=0.0, std=0.1)
    hidden_states = torch.randn(5, 4)
    top_k_index = torch.tensor([[0], [1], [0], [1], [0]])
    top_k_weights = torch.ones(5, 1)
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False))

    output = experts(hidden_states, top_k_index, top_k_weights)
    expected = torch.zeros_like(hidden_states)
    for expert_idx in range(config.num_local_experts):
        token_idx = torch.where(top_k_index[:, 0] == expert_idx)[0]
        gate_up = torch.nn.functional.linear(hidden_states[token_idx], experts.gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=config.swiglu_limit)
        up = up.clamp(min=-config.swiglu_limit, max=config.swiglu_limit)
        activated = (up + 1.0) * gate * torch.sigmoid(config.swiglu_alpha * gate)
        expected[token_idx] = torch.nn.functional.linear(activated, experts.down_proj[expert_idx])

    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("module_name", _MODELING_MODULES)
def test_minimax_m3_vl_rms_norm_preserves_eager_fp32_semantics(module_name):
    modeling = importlib.import_module(module_name)
    norm = modeling.MiniMaxM3VLRMSNorm(8, eps=1e-6)
    norm.weight.data.copy_(torch.linspace(-0.1, 0.1, 8))
    hidden_states = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)

    output = norm(hidden_states)
    expected = hidden_states.float()
    expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + norm.eps)
    expected = expected * (1.0 + norm.weight.float())

    assert output.dtype == hidden_states.dtype
    assert torch.equal(output, expected.to(hidden_states.dtype))
