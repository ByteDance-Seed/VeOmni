from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls
from veomni.models.seed_omni.conversation import ConversationItem
from veomni.models.seed_omni.modules.bagel.sources import (
    BAGEL_FLOW_VELOCITY,
    BAGEL_GENERATED_LATENT,
    BAGEL_SIGLIP_CONTEXT,
    BAGEL_VAE_CONTEXT,
)


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
        patchify_latent_grid,
        prepare_embed_latent_inputs,
    )

    latent = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    noise = torch.linspace(-1.0, 1.0, steps=4, dtype=torch.float32).reshape(1, 4)
    raw_timestep_logits = torch.tensor([0.25], dtype=torch.float32)
    item = ConversationItem(
        type="output",
        value=latent,
        role="assistant",
        meta={
            "timestep": raw_timestep_logits,
            "noise": noise,
        },
    )

    inputs, lengths = prepare_embed_latent_inputs(
        [item],
        config=SimpleNamespace(z_channels=1, latent_patch_size=2, max_latent_size=2),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        timestep_shift=3.0,
    )

    clean, _ = patchify_latent_grid(latent, z_channels=1, latent_patch_size=2)
    clean = clean.to(dtype=torch.bfloat16)
    expected_noise = noise.to(dtype=clean.dtype)
    expected_shifted = raw_timestep_logits
    expected_noised = (1.0 - expected_shifted.reshape(-1, 1)) * clean + expected_shifted.reshape(
        -1, 1
    ) * expected_noise
    expected_noised = expected_noised.to(dtype=torch.bfloat16)

    assert lengths == [1]
    assert item.meta["timestep"].dtype == torch.float32
    assert torch.equal(item.meta["flow_velocity_target"], expected_noise - clean)
    assert torch.equal(inputs["latents"], expected_noised)


def test_bagel_vae_infer_encode_inserts_context_latent_before_user_image():
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
    model._encode_pixel_values = lambda pixel_values: {  # type: ignore[method-assign]
        "latents": torch.ones(int(pixel_values.shape[0]), 2, 2, 2, device=model.device, dtype=model.dtype)
    }
    image = torch.zeros(3, 8, 8)
    conversation = [
        ConversationItem(type="image", value=image, role="user"),
        ConversationItem(type="text", value="edit", role="user"),
    ]

    out = model.encode(conversation_list=conversation)

    assert out["conversation_list"] is conversation
    assert [item.type for item in conversation] == ["output", "image", "text"]
    assert conversation[0].role == "assistant"
    assert conversation[0].meta == {}
    assert conversation[0].value.shape == (2, 2, 2)
    assert conversation[1].value is image


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
        type="output",
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

    out = model.embed_latent(conversation_list=conversation)

    assert out["conversation_list"] is conversation
    assert item.value.shape == (4, 4)
    assert len(captured_timesteps) == 1
    assert torch.equal(captured_timesteps[0], torch.zeros(1, dtype=torch.float32))
    assert "timestep" not in item.meta
    assert "noise" not in item.meta
    assert "flow_velocity_target" not in item.meta


def test_bagel_vae_decode_skips_context_hidden_outputs():
    from veomni.models.seed_omni.modules.bagel.vae.processing import latent_decode_items

    context_hidden = ConversationItem(type="output", value=torch.zeros(6, 8), role="assistant", meta={})
    vae_context = ConversationItem(
        type="output",
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

    assert latent_decode_items([[context_hidden, vae_context, training_latent, velocity, final_latent]]) == [
        final_latent
    ]


def test_bagel_text_encoder_marker_and_token_helpers():
    from veomni.models.seed_omni.modules.bagel.text_encoder.processing import (
        apply_image_marker,
        is_image_item,
    )

    image_item = ConversationItem(
        type="image",
        value=torch.ones(1, 2, 3),
        role="user",
        source=BAGEL_SIGLIP_CONTEXT,
    )
    text_item = ConversationItem(type="text", value="prompt", role="user")
    assert is_image_item(image_item)
    assert not is_image_item(text_item)

    marker_embeds = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    apply_image_marker(image_item, marker_embeds, device=torch.device("cpu"), dtype=torch.float32)
    assert image_item.value.shape == (4, 3)
    wrapped_once = image_item.value.clone()
    apply_image_marker(image_item, marker_embeds, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(image_item.value, wrapped_once)


def test_bagel_flow_generation_state_tracks_denoise_round():
    from veomni.models.seed_omni.modules.bagel.flow_connector.generation_state import FlowGenerationState

    state = FlowGenerationState()
    state.initialize(
        {"image_height": 32, "image_width": 32, "num_timesteps": 2, "timestep_shift": 1.0},
        resolution=64,
        patch_latent_dim=4,
        device=torch.device("cpu"),
    )

    assert state.phase == "prepare_query"
    assert state.token_count == 4
    assert state.require_latent_grid_shape() == (2, 2)
    assert state.current_timestep_tokens().shape == (4,)

    hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    assert torch.equal(state.strip_query_markers(hidden), hidden[1:-1])

    complete = state.advance(torch.zeros_like(state.require_latents()))
    assert complete
    assert state.is_complete()

    state.reset()
    assert not state.initialized
    assert state.phase == "prepare_query"


def test_bagel_raw_image_preprocessing_builds_official_metadata(tmp_path):
    from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
        BagelSiglipNavitProcessor,
        prepare_image_batch,
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
    inputs = prepare_image_batch(
        [image_item],
        config=siglip.config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    processor_inputs = loaded_processor.prepare_image_batch(
        [image_item.value],
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

    BagelTextEncoder = model_cls("bagel_text_encoder")
    BagelTextEncoderConfig = config_cls("bagel_text_encoder")
    text_encoder = BagelTextEncoder(BagelTextEncoderConfig(vocab_size=16, hidden_size=8))
    text_encoder.tokenizer = _BagelInterleaveTokenizer()
    image_item.value = torch.zeros(2, 8)
    image_item.source = BAGEL_SIGLIP_CONTEXT
    text_item = ConversationItem(type="text", value="prompt", role="user")
    text_encoder.generate(conversation_list=[image_item, text_item])
    text_encoder.encode_image_markers(conversation_list=[image_item, text_item])

    assert image_item.value.shape == (4, 8)
    assert text_item.value.shape == (3, 8)
    assert text_item.meta["input_ids"].tolist() == [1, 7, 2]


class _BagelInterleaveTokenizer:
    eos_token_id = 2
    unk_token_id = 0

    _token_ids = {
        "<|im_start|>": 1,
        "<|vision_start|>": 5,
        "<|vision_end|>": 6,
        "prompt": 7,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_ids.get(token, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._token_ids.get(part, 8) for part in text.split()]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        special = {1, 2, 5, 6} if skip_special_tokens else set()
        return " ".join(str(token_id) for token_id in token_ids if token_id not in special)
