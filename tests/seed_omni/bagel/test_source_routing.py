from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.seed_omni.bagel.helpers import config_cls, model_cls, tiny_bagel_qwen2_cfg
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.modules.bagel.vae.processing import route_image_sources
from veomni.models.seed_omni.utils.conversation import _IMG_TAG_KEY, ConversationItem


def test_bagel_route_image_sources_prefers_img_tag() -> None:
    und = ConversationItem(
        type="image",
        value=torch.zeros(3, 2, 2),
        role="user",
        meta={_IMG_TAG_KEY: "und"},
    )
    gen = ConversationItem(
        type="image",
        value=torch.ones(3, 2, 2),
        role="assistant",
        meta={_IMG_TAG_KEY: "gen"},
    )
    edit = ConversationItem(
        type="image",
        value=torch.full((3, 2, 2), 2),
        role="user",
        meta={_IMG_TAG_KEY: "edit"},
    )
    sample = [und, gen, edit]

    route_image_sources([sample], inference=False, infer_type=None)

    assert sample[0] is und
    assert sample[0].source == BAGEL_SIGLIP_CONTEXT
    assert sample[1] is gen
    assert sample[1].source == BAGEL_VAE_CONTEXT
    assert sample[2].source == BAGEL_VAE_CONTEXT
    assert sample[3] is edit
    assert sample[3].source == BAGEL_SIGLIP_CONTEXT
    assert sample[2].meta == sample[3].meta == {_IMG_TAG_KEY: "edit"}
    assert sample[2] is not sample[3]
    sample[2].value[0, 0, 0] = 99
    assert int(sample[3].value[0, 0, 0]) == 2


def test_bagel_route_image_sources_requires_img_tag_for_raw_training_images() -> None:
    image = ConversationItem(type="image", value=torch.zeros(3, 2, 2), role="assistant")

    with pytest.raises(ValueError, match="_img_tag"):
        route_image_sources([[image]], inference=False, infer_type=None)


def test_bagel_route_image_sources_uses_infer_type_for_raw_inference_images() -> None:
    edit_user = ConversationItem(type="image", value=torch.full((3, 2, 2), 3), role="user")
    und_user = ConversationItem(type="image", value=torch.ones(3, 2, 2), role="user")
    gen_user = ConversationItem(type="image", value=torch.zeros(3, 2, 2), role="user")
    edit_sample = [edit_user]
    und_sample = [und_user]
    gen_sample = [gen_user]

    route_image_sources([edit_sample], inference=True, infer_type="infer_edit")
    route_image_sources([und_sample], inference=True, infer_type="infer_und")
    route_image_sources([gen_sample], inference=True, infer_type="infer_gen")

    assert edit_sample[0].source == BAGEL_VAE_CONTEXT
    assert edit_sample[1] is edit_user
    assert edit_sample[1].source == BAGEL_SIGLIP_CONTEXT
    assert und_sample[0] is und_user
    assert und_user.source == BAGEL_SIGLIP_CONTEXT
    assert gen_sample[0] is gen_user
    assert gen_user.source == BAGEL_SIGLIP_CONTEXT


def test_bagel_qwen2_mot_forward_pre_rejects_only_upstream_dummy_anchor(monkeypatch) -> None:
    model = _tiny_qwen2_mot()
    hidden_size = int(model.config.hidden_size)
    siglip_dummy = ConversationItem(
        type="image",
        value=torch.ones(1, hidden_size, requires_grad=True),
        role="dummy",
        source=BAGEL_SIGLIP_CONTEXT,
    )
    monkeypatch.setattr(model, "_has_valid_upstream_embeddings", lambda conversation_list: (True, False))

    with pytest.raises(ValueError, match="got no packable tokens"):
        model.forward_pre(conversation_list=[[siglip_dummy]])


def test_bagel_qwen2_mot_rejects_context_parallel_training(monkeypatch) -> None:
    from veomni.models.seed_omni.modules.bagel.qwen2_mot import accelerated

    model = _tiny_qwen2_mot()
    monkeypatch.setattr(
        accelerated,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_size=2, cp_size=2),
    )
    conversation = [
        [
            ConversationItem(
                type="text",
                value=torch.zeros(1, int(model.config.hidden_size), device=model.device, dtype=model.dtype),
                role="user",
            )
        ]
    ]

    with pytest.raises(ValueError, match="Ulysses sequence parallelism only"):
        model.forward_pre(conversation_list=conversation)


def test_bagel_flow_dummy_embed_anchors_to_vae_dummy_output() -> None:
    model = _tiny_flow_connector()
    vae_dummy = ConversationItem(
        type="image",
        value=torch.ones(1, 1, 1, requires_grad=True),
        role="dummy",
        source=BAGEL_VAE_CONTEXT,
    )
    conversation = [[vae_dummy]]

    inputs = model.embed_latent_pre(conversation_list=conversation)
    latent_embeds = torch.ones(1, int(model.config.hidden_size), device=model.device, dtype=model.dtype)
    latent_embeds = latent_embeds + inputs["latents"].sum() * 0.0
    out = model.embed_latent_post(latent_embeds)

    assert inputs["latents"].requires_grad
    assert out["conversation_list"] is conversation
    flow_dummy = conversation[0][-1]
    assert flow_dummy.type == "output"
    assert flow_dummy.role == "dummy"
    assert flow_dummy.source == "bagel_flow_connector"
    assert flow_dummy.meta == {}
    assert torch.is_tensor(flow_dummy.value)
    assert flow_dummy.value.requires_grad


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
