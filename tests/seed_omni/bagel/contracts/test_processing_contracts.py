from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls, tiny_bagel_qwen2_cfg
from veomni.models.seed_omni.modules.bagel.sources import (
    BAGEL_FLOW_VELOCITY,
    BAGEL_GENERATED_LATENT,
    BAGEL_SIGLIP_CONTEXT,
    BAGEL_VAE_CONTEXT,
)
from veomni.models.seed_omni.utils.conversation import _IMG_TAG_KEY, ConversationItem


def test_bagel_training_text_embed_meta_preserves_grad():
    from veomni.models.seed_omni.modules.bagel.text_encoder.modulemixin import BagelTextEncoderModuleMixin

    item = ConversationItem(
        type="text",
        value=torch.tensor([11, 12]),
        role="assistant",
        meta={"input_ids": torch.tensor([11, 12])},
    )
    packed_text_embeds = torch.randn(2, 4, requires_grad=True)
    mixin = BagelTextEncoderModuleMixin()
    mixin.device = torch.device("cpu")
    mixin.dtype = torch.float32

    mixin._scatter_text_embeds(
        [[item]],
        [packed_text_embeds],
    )
    assert item.value.requires_grad

    item.value.sum().backward()
    assert packed_text_embeds.grad is not None
    assert packed_text_embeds.grad.abs().sum() > 0


def test_bagel_training_flow_metadata_matches_packed_noising_dtype():
    from veomni.models.seed_omni.modules.bagel.flow_connector.processing import (
        preprocess_latent_embed,
    )

    latent = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    item = ConversationItem(
        type="image",
        value=latent,
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={},
    )

    inputs, lengths = preprocess_latent_embed(
        [item],
        config=SimpleNamespace(z_channels=1, latent_patch_size=2, max_latent_size=2),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        timestep_shift=3.0,
    )

    clean = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.bfloat16)
    expected_noise = item.meta["noise"]
    expected_shifted = item.meta["timestep"]
    expected_noised = (1.0 - expected_shifted.reshape(-1, 1)) * clean + expected_shifted.reshape(
        -1, 1
    ) * expected_noise
    expected_noised = expected_noised.to(dtype=torch.bfloat16)

    assert lengths == [1]
    assert item.meta["timestep"].dtype == torch.float32
    assert expected_noise.shape == clean.shape
    assert torch.equal(item.meta["flow_velocity_target"], expected_noise - clean)
    assert torch.equal(inputs["latents"], expected_noised)


@pytest.mark.parametrize("conversation_list", [None, []])
def test_bagel_mot_packing_rejects_missing_conversation_list(conversation_list):
    from veomni.models.seed_omni.modules.bagel.qwen2_mot.processing import preprocess_mot_inputs

    with pytest.raises(ValueError, match="requires a non-empty conversation_list"):
        preprocess_mot_inputs(
            conversation_list,
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_size=4,
        )


def test_bagel_mot_packing_treats_tagged_edit_vae_as_clean_context():
    from veomni.models.seed_omni.modules.bagel.qwen2_mot.processing import preprocess_mot_inputs

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
    assert torch.equal(packed.packed_token_type_ids, torch.tensor([0, 0, 1, 1, 0]))
    assert packed.sample_splits == [[2, 2, 1]]
    assert packed.sample_attn_modes == [["full", "noise", "full"]]


def test_bagel_mot_packing_treats_untagged_vae_as_inference_context():
    from veomni.models.seed_omni.modules.bagel.qwen2_mot.processing import preprocess_mot_inputs

    context = ConversationItem(
        type="image",
        value=torch.ones(2, 4),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={},
    )

    packed = preprocess_mot_inputs([[context]], device=torch.device("cpu"), dtype=torch.float32, hidden_size=4)

    assert packed is not None
    assert torch.equal(packed.packed_token_type_ids, torch.tensor([0, 0]))
    assert packed.sample_splits == [[2]]
    assert packed.sample_attn_modes == [["full"]]


def test_bagel_mot_packing_exposes_sp_metadata_without_rewriting_spans():
    from veomni.models.seed_omni.modules.bagel.qwen2_mot.processing import preprocess_mot_inputs

    text = ConversationItem(type="text", value=torch.ones(2, 4), role="user")
    gen_target = ConversationItem(
        type="image",
        value=torch.full((3, 4), 2),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={_IMG_TAG_KEY: "gen"},
    )

    packed = preprocess_mot_inputs(
        [[text, gen_target]], device=torch.device("cpu"), dtype=torch.float32, hidden_size=4
    )

    assert packed is not None
    assert [span.start for span in packed.spans] == [0, 2]
    assert packed.sample_splits == [[2, 3]]
    assert packed.sample_attn_modes == [["causal", "noise"]]
    assert torch.equal(packed.packed_token_type_ids, torch.tensor([0, 0, 1, 1, 1]))


def test_bagel_mot_forward_pre_builds_non_sp_routing_from_token_types():
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

    assert "packed_token_type_ids" not in inputs
    assert inputs["sample_lens"] == [5]
    assert torch.equal(inputs["packed_und_token_indexes"], torch.tensor([0, 1]))
    assert torch.equal(inputs["packed_gen_token_indexes"], torch.tensor([2, 3, 4]))


def test_bagel_mot_packing_rejects_incompatible_vae_img_tag():
    from veomni.models.seed_omni.modules.bagel.qwen2_mot.processing import preprocess_mot_inputs

    item = ConversationItem(
        type="image",
        value=torch.ones(2, 4),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={_IMG_TAG_KEY: "und"},
    )

    with pytest.raises(ValueError, match="_img_tag"):
        preprocess_mot_inputs([[item]], device=torch.device("cpu"), dtype=torch.float32, hidden_size=4)


def test_bagel_vae_infer_encode_updates_vae_context_image_in_place():
    from veomni.models.seed_omni.modules.bagel.vae.modulemixin import BAGEL_VAE_PIXEL_SHAPE
    from veomni.models.seed_omni.modules.bagel.vae.processing import BagelVAEProcessor

    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    model = BagelVAE(
        BagelVAEConfig(
            resolution=8,
            ch=32,
            ch_mult=[1],
            num_res_blocks=1,
            z_channels=2,
            downsample=4,
            max_image_size=8,
            min_image_size=8,
            image_stride=4,
            max_pixels=64,
        )
    )
    model._image_processor = BagelVAEProcessor.from_config(model.config)
    model.encode = lambda pixel_values, **kwargs: {  # type: ignore[method-assign]
        "latents": torch.ones(int(pixel_values.shape[0]), 2, 2, 2, device=model.device, dtype=model.dtype)
    }
    image = torch.zeros(3, 8, 8)
    vae_context = ConversationItem(
        type="image",
        value=image.clone(),
        role="user",
        source=BAGEL_VAE_CONTEXT,
        meta={BAGEL_VAE_PIXEL_SHAPE: torch.tensor([8, 4])},
    )
    siglip_context = ConversationItem(type="image", value=image, role="user")
    conversation = [
        vae_context,
        siglip_context,
        ConversationItem(type="text", value="edit", role="user"),
    ]

    out = model.encode_context(conversation_list=conversation)

    assert out["conversation_list"] is conversation
    assert [item.type for item in conversation] == ["image", "image", "text"]
    assert conversation[0] is vae_context
    assert conversation[0].role == "user"
    assert conversation[0].source == BAGEL_VAE_CONTEXT
    assert torch.equal(conversation[0].meta[BAGEL_VAE_PIXEL_SHAPE], torch.tensor([8, 4]))
    assert conversation[0].value.shape == (2, 2, 1)
    assert conversation[1] is siglip_context
    assert conversation[1].value is image


def test_bagel_vae_training_encode_marks_context_latent_source():
    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    model = BagelVAE(
        BagelVAEConfig(
            resolution=8,
            ch=32,
            ch_mult=[1],
            num_res_blocks=1,
            z_channels=2,
        )
    )
    item = ConversationItem(type="image", value=torch.zeros(3, 8, 8), role="assistant")
    conversation = [[item]]

    model._conversation_carrier = conversation
    model._encode_items = [item]
    out = model.encode_post(torch.ones(1, 2, 2, 2))

    assert out["conversation_list"] is conversation
    assert item.type == "image"
    assert item.source == BAGEL_VAE_CONTEXT
    assert item.meta == {}
    assert item.value.shape == (2, 2, 2)


def test_bagel_vae_offline_encode_reuses_training_encode_hooks():
    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    model = BagelVAE(
        BagelVAEConfig(
            resolution=8,
            ch=32,
            ch_mult=[1],
            num_res_blocks=1,
            z_channels=2,
        )
    )
    item = ConversationItem(
        type="image",
        value=torch.zeros(3, 8, 8),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
    )
    conversation = [[item]]

    pre = model.pre_forward("offline_encode", conversation_list=conversation)
    assert pre["pixel_values"].shape == (1, 3, 8, 8)

    out = model.post_forward("offline_encode", encoded_cache=torch.ones(1, 2, 2, 2, 2))

    assert out["conversation_list"] is conversation
    assert item.source == BAGEL_VAE_CONTEXT
    assert item.meta == {}
    assert item.value.shape == (2, 2, 2, 2)


def test_bagel_vae_online_process_selects_cached_latents_by_source():
    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    model = BagelVAE(
        BagelVAEConfig(
            resolution=8,
            ch=32,
            ch_mult=[1],
            num_res_blocks=1,
            z_channels=2,
        )
    )
    item = ConversationItem(
        type="image",
        value=torch.ones(2, 2, 2, 2),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
    )
    conversation = [[item]]

    pre = model.pre_forward("online_process", conversation_list=conversation)
    assert isinstance(pre["encoded_cache"], list)
    assert len(pre["encoded_cache"]) == 1
    assert torch.equal(pre["encoded_cache"][0], torch.ones_like(pre["encoded_cache"][0]))
    assert pre["encoded_cache"][0].shape == (2, 2, 2, 2)

    sampled_latents = [torch.ones(2, 2, 2, device=pre["encoded_cache"][0].device)]
    post = model.post_forward("online_process", latents=sampled_latents)

    assert post["conversation_list"] is conversation
    assert item.source == BAGEL_VAE_CONTEXT
    assert item.value.device == sampled_latents[0].device
    assert torch.equal(item.value, torch.ones_like(item.value))


def test_bagel_vae_cache_modes_do_not_construct_reg_and_process_only_has_no_codec() -> None:
    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")

    full = BagelVAE(
        BagelVAEConfig(
            resolution=8,
            ch=32,
            ch_mult=[1],
            num_res_blocks=1,
            z_channels=2,
        )
    )
    process_only = BagelVAE(
        BagelVAEConfig(
            resolution=8,
            ch=32,
            ch_mult=[1],
            num_res_blocks=1,
            z_channels=2,
            support_cache=True,
            train_type="train_with_cache",
        )
    )

    assert not hasattr(full, "reg")
    assert hasattr(full, "encoder")
    assert hasattr(full, "decoder")
    assert not hasattr(process_only, "reg")
    assert not hasattr(process_only, "encoder")
    assert not hasattr(process_only, "decoder")

    try:
        process_only.encode(pixel_values=torch.zeros(1, 3, 8, 8))
    except RuntimeError as exc:
        assert "VAE encoder" in str(exc)
    else:
        raise AssertionError("expected process_only encode to require an encoder")


def test_bagel_vae_training_encode_selects_source_routed_assistant_image():
    from veomni.models.seed_omni.modules.bagel.vae.processing import BagelVAEProcessor

    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    model = BagelVAE(
        BagelVAEConfig(
            resolution=8,
            ch=32,
            ch_mult=[1],
            num_res_blocks=1,
            z_channels=2,
            max_image_size=8,
            min_image_size=8,
            image_stride=4,
            max_pixels=64,
        )
    )
    model._image_processor = BagelVAEProcessor.from_config(model.config)
    item = ConversationItem(
        type="image",
        value=torch.zeros(3, 8, 8),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
    )
    conversation = [[item]]

    out = model.encode_pre(conversation_list=conversation)

    assert model._encode_items == [item]
    assert out["pixel_values"].shape == (1, 3, 8, 8)


def test_bagel_vae_training_encode_crops_padded_latents_before_flow_patchify():
    from veomni.models.seed_omni.modules.bagel.flow_connector.processing import preprocess_latent_embed
    from veomni.models.seed_omni.modules.bagel.vae.modulemixin import BAGEL_VAE_PIXEL_SHAPE
    from veomni.models.seed_omni.modules.bagel.vae.processing import BagelVAEProcessor

    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    model = BagelVAE(
        BagelVAEConfig(
            resolution=8,
            downsample=4,
            ch=32,
            ch_mult=[1],
            num_res_blocks=1,
            z_channels=1,
            max_image_size=8,
            min_image_size=4,
            image_stride=4,
            max_pixels=64,
        )
    )
    model._image_processor = BagelVAEProcessor.from_config(model.config)
    preprocessor = model.build_cpu_preprocessor()
    assert preprocessor is not None

    tall = ConversationItem(
        type="image",
        value=Image.new("RGB", (4, 8), color=(255, 0, 0)),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
    )
    wide = ConversationItem(
        type="image",
        value=Image.new("RGB", (8, 4), color=(0, 255, 0)),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
    )
    conversation = [[tall, wide]]

    preprocessor(conversation)
    encode_inputs = model.encode_pre(conversation_list=conversation)

    assert encode_inputs["pixel_values"].shape == (2, 3, 8, 8)
    assert torch.equal(tall.meta[BAGEL_VAE_PIXEL_SHAPE], torch.tensor([8, 4]))
    assert torch.equal(wide.meta[BAGEL_VAE_PIXEL_SHAPE], torch.tensor([4, 8]))

    padded_latents = torch.arange(8, dtype=model.dtype).reshape(2, 1, 2, 2)
    model.encode_post(padded_latents)

    assert tall.value.shape == (1, 2, 1)
    assert wide.value.shape == (1, 1, 2)
    assert torch.equal(tall.value, padded_latents[0, :, :2, :1])
    assert torch.equal(wide.value, padded_latents[1, :, :1, :2])

    flow_inputs, flow_lengths = preprocess_latent_embed(
        [tall, wide],
        config=SimpleNamespace(z_channels=1, latent_patch_size=1, max_latent_size=2),
        device=torch.device("cpu"),
        dtype=torch.float32,
        timestep_shift=1.0,
    )
    assert flow_lengths == [2, 2]
    assert flow_inputs["latents"].shape == (4, 1)
    assert tall.meta["flow_velocity_target"].shape == (2, 1)
    assert wide.meta["flow_velocity_target"].shape == (2, 1)


def test_bagel_flow_embed_latent_infer_context_keeps_numeric_state_out_of_meta():
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
        )
    )
    item = ConversationItem(
        type="image",
        value=torch.ones(1, 2, 2),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={},
    )
    conversation = [item, ConversationItem(type="text", value="prompt", role="user")]
    captured_timesteps = []
    original_time_embedder_forward = model.time_embedder.forward

    def capture_time_embedder(timesteps: torch.Tensor) -> torch.Tensor:
        captured_timesteps.append(timesteps.detach().clone())
        return original_time_embedder_forward(timesteps)

    model.time_embedder.forward = capture_time_embedder  # type: ignore[method-assign]

    out = model.embed_context_latents(conversation_list=conversation)

    assert out["conversation_list"] is conversation
    assert item.value.shape == (4, 4)
    assert len(captured_timesteps) == 1
    assert torch.equal(captured_timesteps[0], torch.zeros(4, dtype=torch.float32))
    assert "timestep" not in item.meta
    assert "noise" not in item.meta
    assert "flow_velocity_target" not in item.meta


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


def test_bagel_vae_decode_skips_context_hidden_outputs():
    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    model = BagelVAE(
        BagelVAEConfig(
            resolution=8,
            ch=32,
            ch_mult=[1],
            num_res_blocks=1,
            z_channels=2,
        )
    )
    context_hidden = ConversationItem(type="output", value=torch.zeros(6, 8), role="assistant", meta={})
    vae_context = ConversationItem(
        type="image",
        value=torch.zeros(4, 2, 2),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={},
    )
    training_latent = ConversationItem(type="output", value=torch.zeros(4, 2, 2), role="assistant", meta={})
    velocity = ConversationItem(
        type="output",
        value=torch.zeros(4, 2),
        role="assistant",
        source=BAGEL_FLOW_VELOCITY,
        meta={},
    )
    final_latent = ConversationItem(
        type="output",
        value=torch.zeros(4, 2, 2),
        role="assistant",
        source=BAGEL_GENERATED_LATENT,
        meta={},
    )
    conversation = [[context_hidden, vae_context, training_latent, velocity, final_latent]]

    assert model._select_vae_decode_items(conversation) == [final_latent]


def test_bagel_vae_decode_generated_returns_generated_image():
    from veomni.models.seed_omni.modules.bagel.vae.processing import BagelVAEProcessor

    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    model = BagelVAE(
        BagelVAEConfig(
            resolution=8,
            ch=32,
            ch_mult=[1],
            num_res_blocks=1,
            z_channels=2,
        )
    )
    model._image_processor = BagelVAEProcessor.from_config(model.config)
    model.decode = lambda latents, **kwargs: {  # type: ignore[method-assign]
        "pixel_values": torch.ones(int(latents.shape[0]), 3, 2, 2, device=model.device, dtype=model.dtype)
    }
    item = ConversationItem(
        type="output",
        value=torch.zeros(2, 2, 2),
        role="assistant",
        source=BAGEL_GENERATED_LATENT,
        meta={},
    )
    conversation = [item]

    out = model.decode_generated(conversation_list=conversation)

    assert out["conversation_list"] is conversation
    assert out["generated"]["type"] == "image"
    assert isinstance(out["generated"]["value"], Image.Image)
    assert item.type == "image"
    assert item.value.shape == (3, 2, 2)


def test_bagel_flow_generation_state_tracks_denoise_round():
    from veomni.models.seed_omni.modules.bagel.flow_connector.generation_state import FlowGenerationState

    state = FlowGenerationState()
    state.initialize(
        {"image_height": 32, "image_width": 32, "num_timesteps": 2, "timestep_shift": 1.0},
        resolution=64,
        patch_latent_dim=4,
        device=torch.device("cpu"),
    )

    assert state.initialized
    assert state.step_index == 0
    assert state.token_count == 4
    assert state.grid_shape == (2, 2)
    assert state.current_timestep_tokens().shape == (4,)

    complete = state.advance(torch.zeros_like(state.latents))
    assert complete
    assert state.is_complete()

    state.reset()
    assert not state.initialized
    assert state.step_index == 0
    assert state.token_count == 0
    assert state.image_shape is None


def test_bagel_siglip_processor_call_matches_saved_config(tmp_path):
    from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
        BagelSiglipNavitProcessor,
    )

    BagelSiglip = model_cls("bagel_siglip_navit")
    BagelSiglipConfig = config_cls("bagel_siglip_navit")
    siglip = BagelSiglip(
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
    assert siglip.image_processor_class is BagelSiglipNavitProcessor

    processor = BagelSiglipNavitProcessor.from_config(siglip.config)
    processor.save_pretrained(tmp_path)
    loaded_processor = BagelSiglipNavitProcessor.from_pretrained(tmp_path)
    image_item = ConversationItem(
        type="image",
        value=Image.new("RGB", (20, 10), color=(255, 0, 0)),
        role="user",
    )
    inputs = processor(
        images=[image_item.value],
        return_tensors="pt",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    processor_inputs = loaded_processor(
        images=[image_item.value],
        return_tensors="pt",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    call_inputs = loaded_processor(images=[image_item.value], return_tensors="pt")

    assert inputs["patchified_pixel_values"].shape == (2, 14 * 14 * 3)
    assert inputs["patchified_position_ids"].tolist() == [0, 1]
    assert inputs["token_lens"].tolist() == [2]
    assert torch.equal(inputs["patchified_pixel_values"], processor_inputs["patchified_pixel_values"])
    assert torch.equal(inputs["patchified_position_ids"], processor_inputs["patchified_position_ids"])
    assert torch.equal(inputs["cu_seqlens"], processor_inputs["cu_seqlens"])
    assert inputs["max_seqlen"] == processor_inputs["max_seqlen"]
    assert torch.equal(inputs["token_lens"], processor_inputs["token_lens"])
    assert isinstance(call_inputs["max_seqlen"], int)
    assert torch.equal(inputs["patchified_pixel_values"], call_inputs["patchified_pixel_values"])
    assert image_item.meta == {}


def test_bagel_siglip_from_pretrained_loads_processor_config(tmp_path):
    from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
        BagelSiglipNavitProcessor,
    )

    BagelSiglip = model_cls("bagel_siglip_navit")
    BagelSiglipConfig = config_cls("bagel_siglip_navit")
    model = BagelSiglip(
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
    assert model._image_processor is None

    model.save_pretrained(tmp_path)
    BagelSiglipNavitProcessor.from_config(model.config).save_pretrained(tmp_path)
    loaded = BagelSiglip.from_pretrained(tmp_path)
    assert loaded._image_processor.image_size == 28

    loaded.config.image_size = 56
    loaded.config.min_image_size = 56
    loaded.config.max_pixels = 56 * 56
    image_item = ConversationItem(
        type="image",
        value=Image.new("RGB", (20, 10), color=(255, 0, 0)),
        role="user",
    )

    inputs = loaded._image_processor(
        images=[image_item.value],
        return_tensors="pt",
        device=loaded.device,
        dtype=loaded.dtype,
    )

    assert inputs["token_lens"].tolist() == [2]
    assert inputs["patchified_pixel_values"].shape == (2, 14 * 14 * 3)


def test_bagel_vae_processor_call_matches_saved_config(tmp_path):
    from veomni.models.seed_omni.modules.bagel.vae.processing import BagelVAEProcessor

    BagelVAEConfig = config_cls("bagel_vae")
    config = BagelVAEConfig(
        max_image_size=8,
        min_image_size=8,
        image_stride=4,
        max_pixels=64,
    )
    processor = BagelVAEProcessor.from_config(config)
    processor.save_pretrained(tmp_path)
    loaded_processor = BagelVAEProcessor.from_pretrained(tmp_path)
    image = Image.new("RGB", (20, 10), color=(255, 0, 0))

    inputs = processor(images=[image], return_tensors="pt", device=torch.device("cpu"), dtype=torch.float32)
    loaded_inputs = loaded_processor(
        images=[image], return_tensors="pt", device=torch.device("cpu"), dtype=torch.float32
    )

    assert inputs["pixel_values"].shape == (1, 3, 4, 8)
    assert torch.equal(inputs["pixel_values"], loaded_inputs["pixel_values"])


def test_bagel_vae_processor_pads_mixed_image_sizes_for_batch_stack():
    from veomni.models.seed_omni.modules.bagel.vae.processing import BagelVAEProcessor

    processor = BagelVAEProcessor(
        max_image_size=8,
        min_image_size=4,
        image_stride=4,
        max_pixels=64,
    )
    tall = Image.new("RGB", (4, 8), color=(255, 0, 0))
    wide = Image.new("RGB", (8, 4), color=(0, 255, 0))

    inputs = processor(images=[tall, wide], return_tensors="pt", device=torch.device("cpu"), dtype=torch.float32)

    assert inputs["pixel_values"].shape == (2, 3, 8, 8)
    assert inputs["pixel_shapes"].tolist() == [[8, 4], [4, 8]]
    assert torch.equal(inputs["pixel_values"][0, :, :, 4:], torch.zeros(3, 8, 4))
    assert torch.equal(inputs["pixel_values"][1, :, 4:, :], torch.zeros(3, 4, 8))


def test_bagel_vae_from_pretrained_loads_processor_config(tmp_path):
    from veomni.models.seed_omni.modules.bagel.vae.processing import BagelVAEProcessor

    BagelVAE = model_cls("bagel_vae")
    BagelVAEConfig = config_cls("bagel_vae")
    model = BagelVAE(
        BagelVAEConfig(
            resolution=8,
            ch=32,
            ch_mult=[1],
            num_res_blocks=1,
            z_channels=2,
            max_image_size=8,
            min_image_size=8,
            image_stride=4,
            max_pixels=64,
        )
    )
    assert model._image_processor is None

    model.save_pretrained(tmp_path)
    BagelVAEProcessor.from_config(model.config).save_pretrained(tmp_path)
    loaded = BagelVAE.from_pretrained(tmp_path)

    assert loaded._image_processor.max_image_size == 8
    assert loaded._image_processor.image_stride == 4
