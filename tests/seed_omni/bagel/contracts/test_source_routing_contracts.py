from __future__ import annotations

import pytest
import torch

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls, tiny_bagel_qwen2_cfg
from veomni.models.seed_omni.conversation import ConversationItem
from veomni.models.seed_omni.generation_graph import FSM_SIGNAL_KEY
from veomni.models.seed_omni.modules.bagel.flow_connector.modulemixin import SIGNAL_IMAGE_COMPLETE
from veomni.models.seed_omni.modules.bagel.sources import (
    BAGEL_FLOW_HIDDEN,
    BAGEL_FLOW_QUERY,
    BAGEL_FLOW_VELOCITY,
    BAGEL_GENERATED_LATENT,
    BAGEL_VAE_CONTEXT,
)


def test_bagel_denoise_item_source_lifecycle() -> None:
    model = _tiny_flow_connector()
    model.embed_latent = lambda **kwargs: {  # type: ignore[method-assign]
        "latent_embeds": torch.ones(kwargs["latents"].shape[0], 4, device=model.device, dtype=model.dtype)
    }
    model.decode_velocity = lambda hidden_states, **kwargs: {  # type: ignore[method-assign]
        "velocity": torch.ones(hidden_states.shape[0], 1, device=model.device, dtype=model.dtype)
    }
    conversation = [ConversationItem(type="text", value="prompt", role="user")]

    model._prepare_latent_query(
        conversation,
        {"image_height": 16, "image_width": 16, "latent_downsample": 16, "num_timesteps": 2},
    )
    item = conversation[-1]
    assert item.source == BAGEL_FLOW_QUERY
    assert item.meta.keys() == {"timestep"}

    item.source = BAGEL_FLOW_HIDDEN
    item.value = torch.ones(3, 4)
    model._decode_velocity_from_hidden(conversation)
    assert item.source == BAGEL_FLOW_VELOCITY
    assert item.value.shape == (1, 1)
    assert item.meta.keys() == {"timestep"}

    out = model._advance_denoise(conversation)
    assert out[FSM_SIGNAL_KEY] == SIGNAL_IMAGE_COMPLETE
    assert item.source == BAGEL_GENERATED_LATENT
    assert item.value.shape == (1, 1, 1)
    assert item.meta == {}


def test_bagel_qwen2_mot_phase_resolution_uses_source_not_timestep_or_hidden_size() -> None:
    model = _tiny_qwen2_mot()
    model._generation_state.main.cache = object()

    query = ConversationItem(
        type="output",
        value=torch.zeros(3, int(model.config.hidden_size)),
        role="assistant",
        source=BAGEL_FLOW_QUERY,
        meta={},
    )
    assert model._resolve_inference_phase([query], {"infer_mode": "gen"}) == "denoise_branch"

    velocity = ConversationItem(
        type="output",
        value=torch.zeros(3, int(model.config.hidden_size)),
        role="assistant",
        source=BAGEL_FLOW_VELOCITY,
        meta={"timestep": torch.tensor(0.5)},
    )
    assert model._resolve_inference_phase([velocity], {"infer_mode": "gen"}) == "velocity_collect"

    ambiguous = ConversationItem(
        type="output",
        value=torch.zeros(3, int(model.config.hidden_size)),
        role="assistant",
        meta={"timestep": torch.tensor(0.5)},
    )
    with pytest.raises(ValueError, match="requires a current output item"):
        model._resolve_inference_phase([ambiguous], {"infer_mode": "gen"})


def test_bagel_qwen2_mot_keeps_query_shape_validation_after_source_selection() -> None:
    model = _tiny_qwen2_mot()
    model._generation_state.main.cache = object()
    wrong_hidden = ConversationItem(
        type="output",
        value=torch.zeros(3, int(model.config.hidden_size) + 1),
        role="assistant",
        source=BAGEL_FLOW_QUERY,
        meta={},
    )

    with pytest.raises(ValueError, match="hidden-size mismatch"):
        model._resolve_inference_phase([wrong_hidden], {"infer_mode": "gen"})


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
        model._decode_velocity_from_hidden([wrong_source_hidden])

    wrong_size_hidden = ConversationItem(
        type="output",
        value=torch.zeros(3, int(model.config.hidden_size) + 1),
        role="assistant",
        source=BAGEL_FLOW_HIDDEN,
        meta={"timestep": torch.tensor(0.5)},
    )
    with pytest.raises(ValueError, match="hidden-size mismatch"):
        model._decode_velocity_from_hidden([wrong_size_hidden])

    model._generation_state.initialize(
        {"image_height": 16, "image_width": 16, "latent_downsample": 16, "num_timesteps": 2},
        resolution=16,
        patch_latent_dim=int(model.config.patch_latent_dim),
        device=model.device,
    )
    wrong_source_velocity = ConversationItem(type="output", value=torch.zeros(1, 1), role="assistant", meta={})
    with pytest.raises(ValueError, match="bagel_flow_velocity"):
        model._advance_denoise([wrong_source_velocity])

    wrong_shape_velocity = ConversationItem(
        type="output",
        value=torch.zeros(2, 1),
        role="assistant",
        source=BAGEL_FLOW_VELOCITY,
        meta={},
    )
    with pytest.raises(ValueError, match="velocity shape mismatch"):
        model._advance_denoise([wrong_shape_velocity])


def test_bagel_qwen2_mot_velocity_collect_requires_flow_velocity_source() -> None:
    model = _tiny_qwen2_mot()
    item = ConversationItem(
        type="output",
        value=torch.zeros(2, 4),
        role="assistant",
        meta={"timestep": torch.tensor(0.5)},
    )

    with pytest.raises(ValueError, match="bagel_flow_velocity"):
        model._collect_or_merge_velocity(conversation_list=[item], conversation=[item], generation_kwargs={})


def test_bagel_marker_wrapping_skips_velocity_and_generated_latent_sources() -> None:
    from veomni.models.seed_omni.modules.bagel.text_encoder.processing import image_embed_marker_items

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

    assert image_embed_marker_items([[context, query, velocity, generated]], item_types={"output"}) == [
        context,
        query,
    ]


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
