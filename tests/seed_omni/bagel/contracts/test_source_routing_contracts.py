from __future__ import annotations

import pytest
import torch

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls, tiny_bagel_qwen2_cfg
from veomni.models.seed_omni.graphs.generation_graph import FSM_SIGNAL_KEY
from veomni.models.seed_omni.modules.bagel.flow_connector.modulemixin import SIGNAL_IMAGE_COMPLETE
from veomni.models.seed_omni.modules.bagel.sources import (
    BAGEL_FLOW_HIDDEN,
    BAGEL_FLOW_QUERY,
    BAGEL_FLOW_VELOCITY,
    BAGEL_GENERATED_LATENT,
    BAGEL_SIGLIP_CONTEXT,
    BAGEL_VAE_CONTEXT,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem


def test_bagel_denoise_item_source_lifecycle() -> None:
    model = _tiny_flow_connector()
    model.embed_latent = lambda **kwargs: {  # type: ignore[method-assign]
        "latent_embeds": torch.ones(kwargs["latents"].shape[0], 4, device=model.device, dtype=model.dtype)
    }
    model.decode_velocity = lambda hidden_states, **kwargs: {  # type: ignore[method-assign]
        "velocity": torch.ones(hidden_states.shape[0], 1, device=model.device, dtype=model.dtype)
    }
    conversation = [ConversationItem(type="text", value="prompt", role="user")]

    model.prepare_denoise_query(
        conversation_list=conversation,
        generation_kwargs={"image_height": 16, "image_width": 16, "latent_downsample": 16, "num_timesteps": 2},
    )
    item = conversation[-1]
    assert item.source == BAGEL_FLOW_QUERY
    assert item.meta.keys() == {"timestep"}

    item.source = BAGEL_FLOW_HIDDEN
    item.value = torch.ones(3, 4)
    model.decode_velocity_from_hidden(conversation_list=conversation)
    assert item.source == BAGEL_FLOW_VELOCITY
    assert item.value.shape == (1, 1)
    assert item.meta.keys() == {"timestep"}

    out = model.advance_denoise(conversation_list=conversation)
    assert out[FSM_SIGNAL_KEY] == SIGNAL_IMAGE_COMPLETE
    assert item.source == BAGEL_GENERATED_LATENT
    assert item.value.shape == (1, 1, 1)
    assert item.meta == {}


def test_bagel_qwen2_mot_generate_does_not_validate_denoise_cfg() -> None:
    model = _tiny_qwen2_mot()
    conversation = [
        ConversationItem(
            type="text",
            value=torch.zeros(1, int(model.config.hidden_size)),
            role="user",
        )
    ]

    def fake_prefill(
        conversation_list: list[ConversationItem],
        generation_kwargs: dict[str, object],
    ) -> torch.Tensor:
        del conversation_list
        assert generation_kwargs["cfg_img_scale"] == 2.0
        return torch.zeros(1, int(model.config.hidden_size))

    model._prefill_prompt = fake_prefill  # type: ignore[method-assign]

    out = model.generate(
        conversation_list=conversation,
        generation_kwargs={"infer_mode": "gen", "cfg_img_scale": 2.0, "cfg_text_scale": 1.0},
    )

    assert out["conversation_list"] is conversation


def test_bagel_qwen2_mot_prefill_builds_cfg_img_context_only_when_requested() -> None:
    model = _tiny_qwen2_mot()
    hidden_size = int(model.config.hidden_size)
    calls: list[dict[str, object]] = []

    def fake_forward_inference(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "past_key_values": object(),
            "hidden_states": torch.zeros(1, hidden_size),
        }

    model.forward_inference = fake_forward_inference  # type: ignore[method-assign]
    model.generate(
        conversation_list=[
            ConversationItem(type="text", value=torch.zeros(1, hidden_size), role="user"),
        ],
        generation_kwargs={"infer_mode": "gen"},
    )

    assert len(calls) == 1
    assert model._generation_state.cfg_img.cache is None

    model = _tiny_qwen2_mot()
    calls = []
    model.forward_inference = fake_forward_inference  # type: ignore[method-assign]
    model.generate(
        conversation_list=[
            ConversationItem(type="text", value=torch.zeros(1, hidden_size), role="user"),
        ],
        generation_kwargs={"infer_mode": "gen", "cfg_text_scale": 4.0, "cfg_img_scale": 1.5},
    )

    assert len(calls) == 2
    assert model._generation_state.cfg_img.cache is not None


def test_bagel_qwen2_mot_keeps_denoise_shape_validation_after_source_selection() -> None:
    model = _tiny_qwen2_mot()
    model._generation_state.main.snapshot(
        cache=object(),
        key_values_lens=torch.zeros(1, dtype=torch.int32),
        packed_key_value_indexes=torch.empty(0, dtype=torch.long),
        next_position_id=torch.zeros(1, dtype=torch.long),
        empty_cache_factory=model._new_empty_cache,
        device=model.device,
    )
    wrong_hidden = ConversationItem(
        type="output",
        value=torch.zeros(3, int(model.config.hidden_size) + 1),
        role="assistant",
        source=BAGEL_FLOW_QUERY,
        meta={},
    )

    with pytest.raises(ValueError, match="hidden-size mismatch"):
        model.denoise_branch(conversation_list=[wrong_hidden], generation_kwargs={"infer_mode": "gen"})


def test_bagel_qwen2_mot_siglip_alignment_uses_source_before_shape() -> None:
    model = _tiny_qwen2_mot()
    hidden_size = int(model.config.hidden_size)
    packed_sequence = torch.ones(1, hidden_size)
    siglip_dummy = ConversationItem(
        type="image",
        value=torch.ones(1, hidden_size, requires_grad=True),
        role="dummy",
        meta={"source": "bagel_siglip_navit"},
    )
    raw_hidden_shaped_image = ConversationItem(
        type="image",
        value=torch.zeros(2, hidden_size),
        role="user",
        meta={},
    )

    out = model._fold_dummy_anchors(
        packed_sequence,
        [[raw_hidden_shaped_image, siglip_dummy]],
    )
    assert out is packed_sequence

    source_tagged_siglip = ConversationItem(
        type="image",
        value=torch.zeros(2, hidden_size),
        role="user",
        source=BAGEL_SIGLIP_CONTEXT,
        meta={},
    )
    out = model._fold_dummy_anchors(
        packed_sequence,
        [[source_tagged_siglip, siglip_dummy]],
    )
    assert out is not packed_sequence
    assert out.requires_grad

    wrong_size_siglip = ConversationItem(
        type="image",
        value=torch.zeros(2, hidden_size + 1),
        role="user",
        source=BAGEL_SIGLIP_CONTEXT,
        meta={},
    )
    with pytest.raises(ValueError, match="SigLIP alignment hidden-size mismatch"):
        model._fold_dummy_anchors(packed_sequence, [[wrong_size_siglip, siglip_dummy]])


def test_bagel_qwen2_mot_flow_alignment_requires_real_flow_item() -> None:
    model = _tiny_qwen2_mot()
    hidden_size = int(model.config.hidden_size)
    packed_sequence = torch.ones(1, hidden_size)
    flow_dummy = ConversationItem(
        type="output",
        value=torch.ones(1, hidden_size, requires_grad=True),
        role="dummy",
        meta={"source": "bagel_flow_connector"},
    )
    unrelated_output = ConversationItem(
        type="output",
        value=torch.zeros(2, hidden_size),
        role="assistant",
        meta={},
    )

    out = model._fold_dummy_anchors(packed_sequence, [[unrelated_output, flow_dummy]])
    assert out is packed_sequence

    flow_output = ConversationItem(
        type="output",
        value=torch.zeros(2, hidden_size),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={"flow_velocity_target": torch.zeros(2, 1)},
    )
    out = model._fold_dummy_anchors(packed_sequence, [[flow_output, flow_dummy]])
    assert out is not packed_sequence
    assert out.requires_grad


def test_bagel_flow_decode_and_advance_require_source_routed_products() -> None:
    model = _tiny_flow_connector()
    model.decode_velocity = lambda hidden_states, **kwargs: {  # type: ignore[method-assign]
        "velocity": torch.ones(hidden_states.shape[0], 1, device=model.device, dtype=model.dtype)
    }

    wrong_source_hidden = ConversationItem(
        type="output",
        value=torch.zeros(3, int(model.config.hidden_size)),
        role="assistant",
        source=BAGEL_FLOW_QUERY,
        meta={"timestep": torch.tensor(0.5)},
    )
    with pytest.raises(ValueError, match="bagel_flow_hidden"):
        model.decode_velocity_from_hidden(conversation_list=[wrong_source_hidden])

    wrong_size_hidden = ConversationItem(
        type="output",
        value=torch.zeros(3, int(model.config.hidden_size) + 1),
        role="assistant",
        source=BAGEL_FLOW_HIDDEN,
        meta={"timestep": torch.tensor(0.5)},
    )
    with pytest.raises(ValueError, match="hidden-size mismatch"):
        model.decode_velocity_from_hidden(conversation_list=[wrong_size_hidden])

    model._generation_state.initialize(
        {"image_height": 16, "image_width": 16, "latent_downsample": 16, "num_timesteps": 2},
        resolution=16,
        patch_latent_dim=int(model.config.patch_latent_dim),
        device=model.device,
    )
    wrong_source_velocity = ConversationItem(type="output", value=torch.zeros(1, 1), role="assistant", meta={})
    with pytest.raises(ValueError, match="bagel_flow_velocity"):
        model.advance_denoise(conversation_list=[wrong_source_velocity])

    wrong_shape_velocity = ConversationItem(
        type="output",
        value=torch.zeros(2, 1),
        role="assistant",
        source=BAGEL_FLOW_VELOCITY,
        meta={},
    )
    with pytest.raises(ValueError, match="velocity shape mismatch"):
        model.advance_denoise(conversation_list=[wrong_shape_velocity])


def test_bagel_flow_reset_clears_denoise_state() -> None:
    model = _tiny_flow_connector()
    model.embed_latent = lambda **kwargs: {  # type: ignore[method-assign]
        "latent_embeds": torch.ones(kwargs["latents"].shape[0], 4, device=model.device, dtype=model.dtype)
    }
    conversation = [ConversationItem(type="text", value="prompt", role="user")]

    model.prepare_denoise_query(conversation_list=conversation)
    assert model._generation_state.initialized

    model.reset_local_inference_state()
    assert not model._generation_state.initialized


def test_bagel_flow_context_embed_consumes_only_vae_context_latents() -> None:
    model = _tiny_flow_connector()
    raw_latent = ConversationItem(type="output", value=torch.zeros(1, 2, 2), role="assistant", meta={})
    vae_context = ConversationItem(
        type="output",
        value=torch.ones(1, 2, 2),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={},
    )
    conversation = [raw_latent, vae_context]
    model.embed_latent = lambda **kwargs: {  # type: ignore[method-assign]
        "latent_embeds": torch.full((kwargs["latents"].shape[0], 4), 7.0, device=model.device, dtype=model.dtype)
    }

    model.embed_context_latents(conversation_list=conversation)

    assert raw_latent.value.shape == (1, 2, 2)
    assert vae_context.value.shape == (4, 4)


def test_bagel_flow_training_embed_consumes_only_vae_context_latents() -> None:
    model = _tiny_flow_connector()
    raw_latent = ConversationItem(type="output", value=torch.zeros(1, 2, 2), role="assistant", meta={})
    vae_context = ConversationItem(
        type="output",
        value=torch.ones(1, 2, 2),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={},
    )

    out = model.embed_latent_pre(conversation_list=[[raw_latent, vae_context]], timestep_shift=1.0)

    assert model._embed_items == [vae_context]
    assert out["latents"].shape == (4, 1)
    assert "flow_velocity_target" not in raw_latent.meta
    assert "flow_velocity_target" in vae_context.meta


def test_bagel_flow_training_decode_consumes_velocity_target_items() -> None:
    model = _tiny_flow_connector()
    hidden = torch.zeros(2, int(model.config.hidden_size))
    unrelated = ConversationItem(type="output", value=hidden.clone(), role="assistant", meta={})
    target = ConversationItem(
        type="output",
        value=hidden.clone(),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={"flow_velocity_target": torch.ones(2, int(model.config.patch_latent_dim))},
    )

    out = model.decode_velocity_pre(conversation_list=[[unrelated, target]])

    assert model._decode_items == [target]
    assert out["hidden_states"].shape == (2, int(model.config.hidden_size))


def test_bagel_qwen2_mot_velocity_collect_requires_flow_velocity_source() -> None:
    model = _tiny_qwen2_mot()
    item = ConversationItem(
        type="output",
        value=torch.zeros(2, 4),
        role="assistant",
        meta={"timestep": torch.tensor(0.5)},
    )

    with pytest.raises(ValueError, match="bagel_flow_velocity"):
        model.collect_velocity(conversation_list=[item], generation_kwargs={})


def test_bagel_marker_wrapping_skips_velocity_and_generated_latent_sources() -> None:
    siglip = ConversationItem(
        type="image",
        value=torch.zeros(2, 4),
        role="user",
        source=BAGEL_SIGLIP_CONTEXT,
        meta={},
    )
    raw_image = ConversationItem(
        type="image",
        value=torch.zeros(2, 4),
        role="user",
        meta={},
    )
    context = ConversationItem(
        type="output",
        value=torch.zeros(2, 4),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={},
    )
    query = ConversationItem(
        type="output",
        value=torch.zeros(2, 4),
        role="assistant",
        source=BAGEL_FLOW_QUERY,
        meta={},
    )
    velocity = ConversationItem(
        type="output",
        value=torch.zeros(2, 4),
        role="assistant",
        source=BAGEL_FLOW_VELOCITY,
        meta={},
    )
    generated = ConversationItem(
        type="output",
        value=torch.zeros(2, 4),
        role="assistant",
        source=BAGEL_GENERATED_LATENT,
        meta={},
    )

    marker_sources = {BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT, BAGEL_FLOW_QUERY}
    assert [
        item
        for item in (siglip, raw_image, context, query, velocity, generated)
        if item.type in {"image", "output"} and item.source in marker_sources
    ] == [
        siglip,
        context,
        query,
    ]


def test_bagel_siglip_selector_accepts_raw_and_context_sources() -> None:
    model = _tiny_siglip()
    raw_image = ConversationItem(type="image", value=torch.zeros(3, 4, 4), role="user")
    context_image = ConversationItem(
        type="image",
        value=torch.zeros(3, 4, 4),
        role="user",
        source=BAGEL_SIGLIP_CONTEXT,
    )

    assert model._select_siglip_image_items([[raw_image]]) == [raw_image]
    assert model._select_siglip_image_items([[context_image]]) == [context_image]


def _tiny_siglip():
    BagelSiglip = model_cls("bagel_siglip_navit")
    BagelSiglipConfig = config_cls("bagel_siglip_navit")
    return BagelSiglip(
        BagelSiglipConfig(
            hidden_size=8,
            output_size=8,
            image_size=28,
            min_image_size=14,
            max_pixels=28 * 14,
            intermediate_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            patch_size=14,
            vit_max_num_patch_per_side=2,
        )
    )


def _tiny_flow_connector():
    BagelFlowConnector = model_cls("bagel_flow_connector")
    BagelFlowConnectorConfig = config_cls("bagel_flow_connector")
    return BagelFlowConnector(
        BagelFlowConnectorConfig(
            hidden_size=4,
            z_channels=1,
            latent_patch_size=1,
            patch_latent_dim=1,
            max_latent_size=4,
            timestep_frequency_embedding_size=4,
            resolution=16,
        )
    )


def _tiny_qwen2_mot():
    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    return BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg()))
