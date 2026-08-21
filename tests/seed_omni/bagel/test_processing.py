from __future__ import annotations

from inspect import Parameter, signature
from types import SimpleNamespace

import pytest
import torch

from tests.seed_omni.bagel.helpers import (
    config_cls,
    model_cls,
    native_model_cls,
    tiny_bagel_qwen2_cfg,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated import TrainingMixin
from veomni.models.seed_omni.modules.bagel.qwen2_mot.processing import preprocess_mot_inputs
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.modules.bagel.vae.processing import BagelVAEProcessor
from veomni.models.seed_omni.utils.conversation import _IMG_TAG_KEY, ConversationItem


@pytest.mark.parametrize("conversation_list", [None, []])
def test_bagel_mot_packing_rejects_missing_conversation_list(conversation_list):
    with pytest.raises(ValueError, match="requires a non-empty conversation_list"):
        preprocess_mot_inputs(
            conversation_list,
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_size=4,
        )


def test_bagel_mot_packing_rejects_incompatible_vae_img_tag():
    item = ConversationItem(
        type="image",
        value=torch.ones(2, 4),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={_IMG_TAG_KEY: "und"},
    )

    with pytest.raises(ValueError, match="_img_tag"):
        preprocess_mot_inputs([[item]], device=torch.device("cpu"), dtype=torch.float32, hidden_size=4)


def test_bagel_mot_packing_routes_tagged_edit_vae_through_generation_expert():
    edit_context = ConversationItem(
        type="image",
        value=torch.ones(2, 4),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={_IMG_TAG_KEY: "edit"},
    )
    gen_target = ConversationItem(
        type="image",
        value=torch.full((2, 4), 2),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={_IMG_TAG_KEY: "gen"},
    )
    siglip_context = ConversationItem(
        type="image",
        value=torch.full((1, 4), 3),
        role="user",
        source=BAGEL_SIGLIP_CONTEXT,
        meta={_IMG_TAG_KEY: "edit"},
    )

    packed = preprocess_mot_inputs(
        [[edit_context, gen_target, siglip_context]],
        device=torch.device("cpu"),
        dtype=torch.float32,
        hidden_size=4,
    )

    assert packed is not None
    assert torch.equal(packed.packed_token_type_ids, torch.tensor([1, 1, 1, 1, 0]))
    assert packed.sample_splits == [[2, 2, 1]]


def test_bagel_mot_forward_pre_returns_sample_local_tensor_contract():
    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg())).train()
    hidden_size = int(model.config.hidden_size)
    text = ConversationItem(type="text", value=torch.ones(2, hidden_size), role="user")
    gen_target = ConversationItem(
        type="image",
        value=torch.ones(3, hidden_size),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={_IMG_TAG_KEY: "gen"},
    )

    inputs = model.forward_pre(conversation_list=[[text, gen_target]])

    # The accelerated class inherits the graph's generic ``forward(**kwargs)``
    # trampoline, so the explicit packed-tensor contract lives on the native one.
    forward_parameters = signature(native_model_cls("bagel_qwen2_mot").forward).parameters
    assert "packed_token_type_ids" in forward_parameters
    assert "packed_und_token_indexes" not in forward_parameters
    assert "packed_gen_token_indexes" not in forward_parameters
    assert all(parameter.kind is not Parameter.VAR_KEYWORD for parameter in forward_parameters.values())
    assert set(inputs) == {
        "packed_sequence",
        "packed_position_ids",
        "packed_token_type_ids",
        "packed_attention_metadata",
    }
    assert torch.equal(inputs["packed_token_type_ids"], torch.tensor([0, 0, 1, 1, 1]))
    assert inputs["packed_attention_metadata"].shape == (3, 5)
    assert inputs["packed_attention_metadata"].dtype == torch.int32
    assert inputs["packed_attention_metadata"].is_contiguous()


def test_bagel_mot_forward_pre_keeps_metadata_full_and_marks_sequence_padding(monkeypatch):
    from veomni.models.seed_omni.modules.bagel.qwen2_mot import accelerated

    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg())).train()
    monkeypatch.setattr(
        accelerated,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_size=4, cp_size=1, ulysses_size=4, sp_group=object()),
    )

    def fake_sp_pad(tensor, dim, pad_value):
        pad_shape = list(tensor.shape)
        pad_shape[dim] = 3
        padding = torch.full(pad_shape, pad_value, device=tensor.device, dtype=tensor.dtype)
        return torch.cat((tensor, padding), dim=dim)

    monkeypatch.setattr(accelerated, "sp_pad", fake_sp_pad)
    monkeypatch.setattr(accelerated, "slice_input_tensor", lambda tensor, **kwargs: tensor)
    hidden_size = int(model.config.hidden_size)
    text = ConversationItem(type="text", value=torch.ones(2, hidden_size), role="user")
    image = ConversationItem(
        type="image",
        value=torch.ones(3, hidden_size),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={_IMG_TAG_KEY: "gen"},
    )

    inputs = model.forward_pre(conversation_list=[[text, image]])

    assert inputs["packed_attention_metadata"].shape == (3, 8)
    assert inputs["packed_sequence"].shape[0] == 8
    assert set(inputs) == {
        "packed_sequence",
        "packed_position_ids",
        "packed_token_type_ids",
        "packed_attention_metadata",
    }
    assert torch.equal(inputs["packed_token_type_ids"], torch.tensor([0, 0, 1, 1, 1, -1, -1, -1]))
    assert torch.equal(inputs["packed_attention_metadata"][0, 5:], torch.tensor([5, 5, 5], dtype=torch.int32))
    assert torch.equal(inputs["packed_attention_metadata"][1, 5:], torch.tensor([5, 5, 5], dtype=torch.int32))
    assert torch.equal(inputs["packed_attention_metadata"][2, 5:], torch.tensor([-1, -1, -1], dtype=torch.int32))


def test_bagel_flow_training_embed_treats_edit_context_as_clean_and_gen_as_target():
    BagelFlowConnector = model_cls("bagel_flow_connector")
    BagelFlowConnectorConfig = config_cls("bagel_flow_connector")
    model = BagelFlowConnector(
        BagelFlowConnectorConfig(
            hidden_size=4,
            z_channels=1,
            latent_patch_size=1,
            patch_latent_dim=1,
            max_latent_size=4,
            timestep_frequency_embedding_size=4,
            timestep_shift=1.0,
        )
    )
    edit_context = ConversationItem(
        type="image",
        value=torch.ones(1, 1, 2),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={_IMG_TAG_KEY: "edit"},
    )
    gen_target = ConversationItem(
        type="image",
        value=torch.full((1, 1, 2), 2.0),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={_IMG_TAG_KEY: "gen"},
    )

    inputs = model.embed_latent_pre(conversation_list=[[edit_context, gen_target]])

    assert model._embed_items == [edit_context, gen_target]
    assert model._embed_lengths == [2, 2]
    assert torch.equal(inputs["latents"][:2], torch.ones(2, 1))
    assert torch.equal(inputs["timesteps"][:2], torch.zeros(2))
    assert "flow_velocity_target" not in edit_context.meta
    assert "timestep" not in edit_context.meta
    assert gen_target.meta["flow_velocity_target"].shape == (2, 1)
    assert gen_target.meta["timestep"].shape == (2,)


def test_mot_forward_post_scatters_virtual_marker_triplet_hidden_states() -> None:
    hidden_size = 4
    markers = torch.tensor([[101.0, 102.0, 103.0, 104.0], [201.0, 202.0, 203.0, 204.0]])

    def marker(row: int, source: str) -> ConversationItem:
        return ConversationItem(
            type="text",
            value=markers[row : row + 1],
            role="user",
            source=source,
            meta={"labels": torch.full((1,), -100)},
        )

    latent_item = ConversationItem(
        type="image",
        value=torch.arange(8, dtype=torch.float32).reshape(2, hidden_size) + 20,
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
    )
    siglip_item = ConversationItem(
        type="image",
        value=torch.arange(12, dtype=torch.float32).reshape(3, hidden_size),
        role="user",
        source=BAGEL_SIGLIP_CONTEXT,
    )
    text_item = ConversationItem(
        type="text",
        value=torch.arange(8, dtype=torch.float32).reshape(2, hidden_size) + 40,
        role="user",
    )
    sample = [
        marker(0, BAGEL_VAE_CONTEXT),
        latent_item,
        marker(1, BAGEL_VAE_CONTEXT),
        marker(0, BAGEL_SIGLIP_CONTEXT),
        siglip_item,
        marker(1, BAGEL_SIGLIP_CONTEXT),
        text_item,
    ]
    packed = preprocess_mot_inputs(
        [sample],
        device=torch.device("cpu"),
        dtype=torch.float32,
        hidden_size=hidden_size,
    )
    assert packed is not None

    model = SimpleNamespace(
        device=torch.device("cpu"),
        _conversation_carrier=[sample],
        _packed_training=packed,
    )
    hidden_states = torch.arange(11 * hidden_size, dtype=torch.float32).reshape(11, hidden_size)

    out = TrainingMixin.forward_post(model, hidden_states)

    assert out["conversation_list"] == [sample]
    assert torch.equal(sample[0].value, hidden_states[0:1])
    assert torch.equal(latent_item.value, hidden_states[1:3])
    assert torch.equal(sample[2].value, hidden_states[3:4])
    assert torch.equal(sample[3].value, hidden_states[4:5])
    assert torch.equal(siglip_item.value, hidden_states[5:8])
    assert torch.equal(sample[5].value, hidden_states[8:9])
    assert torch.equal(text_item.value, hidden_states[9:11])


def test_bagel_vae_process_only_skips_codec_modules() -> None:
    encode_model = _tiny_vae()
    process_model = _tiny_vae(support_cache=True, train_type="train_with_cache")
    encoded_cache = encode_model.offline_encode(pixel_values=torch.zeros(1, 3, 8, 8))["encoded_cache"]
    item_cache = encoded_cache[0].reshape(2, process_model.config.z_channels, *encoded_cache.shape[-2:])

    assert not hasattr(process_model, "encoder")
    assert not hasattr(process_model, "decoder")
    latents = process_model.online_process(encoded_cache=item_cache)["latents"]
    assert isinstance(latents, list)
    assert latents[0].shape == item_cache.shape[1:]

    with pytest.raises(RuntimeError, match="VAE encoder"):
        process_model.encode(pixel_values=torch.zeros(1, 3, 8, 8))
    with pytest.raises(RuntimeError, match="VAE decoder"):
        process_model.decode(latents=torch.zeros(1, 2, 2, 2))


def test_bagel_vae_encode_only_skips_decoder_module() -> None:
    model = _tiny_vae(support_cache=True, train_type="offline_cache")

    assert hasattr(model, "encoder")
    assert not hasattr(model, "decoder")
    assert model.offline_encode(pixel_values=torch.zeros(1, 3, 8, 8))["encoded_cache"].shape[:2] == (
        1,
        2 * model.config.z_channels,
    )

    with pytest.raises(RuntimeError, match="VAE decoder"):
        model.decode(latents=torch.zeros(1, 2, 2, 2))


def test_bagel_vae_online_process_consumes_variable_size_cache_items_without_padding() -> None:
    model = _tiny_vae(support_cache=True, train_type="train_with_cache")
    first = ConversationItem(
        type="image",
        value=torch.zeros(2, 2, 2, 1),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
    )
    second = ConversationItem(
        type="image",
        value=torch.zeros(2, 2, 2, 2),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
    )
    conversation = [[first], [second]]

    pre = model.online_process_pre(conversation_list=conversation)
    out = model.online_process(encoded_cache=pre["encoded_cache"])
    post = model.post_forward("online_process", latents=out["latents"])

    assert isinstance(pre["encoded_cache"], list)
    assert [tuple(cache.shape) for cache in pre["encoded_cache"]] == [(2, 2, 2, 1), (2, 2, 2, 2)]
    assert post["conversation_list"] is conversation
    assert first.value.shape == (2, 2, 1)
    assert second.value.shape == (2, 2, 2)


def _tiny_vae(**config_overrides):
    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    config_kwargs = dict(
        resolution=8,
        ch=32,
        ch_mult=[1],
        num_res_blocks=1,
        z_channels=2,
        max_image_size=8,
        min_image_size=4,
        image_stride=4,
        max_pixels=64,
        downsample=1,
    )
    config_kwargs.update(config_overrides)
    model = BagelVAE(BagelVAEConfig(**config_kwargs))
    model._image_processor = BagelVAEProcessor.from_config(model.config)
    return model
