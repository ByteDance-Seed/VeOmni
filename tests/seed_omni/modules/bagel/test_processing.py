from __future__ import annotations

import copy
import pickle
from inspect import Parameter, signature
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.seed_omni.modules.bagel.helpers import (
    config_cls,
    model_cls,
    native_model_cls,
    tiny_bagel_qwen2_cfg,
)
from veomni.models.seed_omni import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated.accelerated import TrainingMixin
from veomni.models.seed_omni.modules.bagel.qwen2_mot.processing import preprocess_mot_inputs
from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
    _OMNI_POSITION_IDS as BAGEL_SIGLIP_POSITION_IDS,
)
from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
    _OMNI_TOKEN_LEN as BAGEL_SIGLIP_TOKEN_LEN,
)
from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
    BagelSiglipNavitPreprocessor,
    BagelSiglipNavitProcessor,
)
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_CONTEXT_KEY, BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.modules.bagel.text_encoder.chat_template import BagelChatTemplate
from veomni.models.seed_omni.modules.bagel.text_encoder.modeling import (
    _OMNI_TOKENIZED as BAGEL_TOK,
)
from veomni.models.seed_omni.modules.bagel.text_encoder.processing import (
    BagelTextEncoderPreprocessor,
)
from veomni.models.seed_omni.modules.bagel.vae.processing import (
    BAGEL_VAE_PIXEL_SHAPE,
    BAGEL_VAE_POSTERIOR,
    BagelVAEPreprocessor,
    BagelVAEProcessor,
)
from veomni.models.seed_omni.utils.conversation import _IMG_TAG_KEY, ConversationItem, iter_desired_items


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize("conversation_list", [None, []])
def test_bagel_mot_packing_rejects_missing_conversation_list(conversation_list):
    """Verify ``preprocess_mot_inputs`` rejects ``None`` or empty ``conversation_list``."""
    with pytest.raises(ValueError, match="requires a non-empty conversation_list"):
        preprocess_mot_inputs(
            conversation_list,
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_size=4,
        )


def test_bagel_mot_packing_rejects_incompatible_vae_img_tag():
    """Verify ``preprocess_mot_inputs`` rejects VAE context items tagged with ``_img_tag='und'``."""
    item = ConversationItem(
        type="image",
        value=torch.ones(2, 4),
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, _IMG_TAG_KEY: "und"},
    )

    with pytest.raises(ValueError, match="_img_tag"):
        preprocess_mot_inputs([[item]], device=torch.device("cpu"), dtype=torch.float32, hidden_size=4)


def test_bagel_mot_packing_routes_tagged_edit_vae_through_generation_expert():
    """Verify ``edit`` and ``gen`` VAE context items route through the generation expert while SigLIP routes through und."""
    edit_context = ConversationItem(
        type="image",
        value=torch.ones(2, 4),
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, _IMG_TAG_KEY: "edit"},
    )
    gen_target = ConversationItem(
        type="image",
        value=torch.full((2, 4), 2),
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, _IMG_TAG_KEY: "gen"},
    )
    siglip_context = ConversationItem(
        type="image",
        value=torch.full((1, 4), 3),
        role="user",
        meta={BAGEL_CONTEXT_KEY: BAGEL_SIGLIP_CONTEXT, _IMG_TAG_KEY: "edit"},
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
    """Verify ``BagelQwen2MoTAccelerated.forward_pre`` returns the packed tensor contract expected by ``forward``."""
    BagelQwen2MoT = model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**tiny_bagel_qwen2_cfg())).train()
    hidden_size = int(model.config.hidden_size)
    text = ConversationItem(type="text", value=torch.ones(2, hidden_size), role="user")
    gen_target = ConversationItem(
        type="image",
        value=torch.ones(3, hidden_size),
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, _IMG_TAG_KEY: "gen"},
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
    """Verify ``forward_pre`` under Ulysses SP pads sequence tensors with ``-1`` token types and keeps full metadata."""
    from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated import accelerated

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
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, _IMG_TAG_KEY: "gen"},
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
    """Verify ``embed_latent_pre`` leaves ``edit`` context at timestep 0 without targets and noises ``gen`` targets."""
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
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, _IMG_TAG_KEY: "edit"},
    )
    gen_target = ConversationItem(
        type="image",
        value=torch.full((1, 1, 2), 2.0),
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, _IMG_TAG_KEY: "gen"},
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
    assert torch.equal(gen_target.meta["timestep"][0], gen_target.meta["timestep"][1])
    assert torch.equal(inputs["timesteps"][2], inputs["timesteps"][3])


def test_bagel_flow_preprocess_latent_embed_samples_one_shared_timestep_per_image() -> None:
    """Verify ``preprocess_latent_embed`` samples one scalar timestep per image and expands it across patch tokens."""
    from veomni.models.seed_omni.modules.bagel.flow_connector.processing import preprocess_latent_embed

    BagelFlowConnectorConfig = config_cls("bagel_flow_connector")
    cfg = BagelFlowConnectorConfig(
        hidden_size=4,
        z_channels=1,
        latent_patch_size=1,
        patch_latent_dim=1,
        max_latent_size=4,
        timestep_frequency_embedding_size=4,
        timestep_shift=3.0,
    )
    first = ConversationItem(
        type="image",
        value=torch.zeros(1, 2, 2),
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, _IMG_TAG_KEY: "gen"},
    )
    second = ConversationItem(
        type="image",
        value=torch.ones(1, 2, 3),
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, _IMG_TAG_KEY: "gen"},
    )
    torch.manual_seed(1234)
    inputs, lengths = preprocess_latent_embed(
        [first, second],
        config=cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
        timestep_shift=3.0,
    )
    assert lengths == [4, 6]
    assert torch.all(first.meta["timestep"] == first.meta["timestep"][0])
    assert torch.all(second.meta["timestep"] == second.meta["timestep"][0])
    assert not torch.equal(first.meta["timestep"][0], second.meta["timestep"][0])
    assert torch.equal(inputs["timesteps"][:4], first.meta["timestep"])
    assert torch.equal(inputs["timesteps"][4:], second.meta["timestep"])


def test_mot_forward_post_scatters_virtual_marker_triplet_hidden_states() -> None:
    """Verify ``forward_post`` scatters packed hidden states back across ``(start, image, end)`` triplet items."""
    hidden_size = 4
    markers = torch.tensor([[101.0, 102.0, 103.0, 104.0], [201.0, 202.0, 203.0, 204.0]])

    def marker(row: int, context: str) -> ConversationItem:
        return ConversationItem(
            type="text",
            value=markers[row : row + 1],
            role="user",
            meta={"labels": torch.full((1,), -100), BAGEL_CONTEXT_KEY: context},
        )

    latent_item = ConversationItem(
        type="image",
        value=torch.arange(8, dtype=torch.float32).reshape(2, hidden_size) + 20,
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT},
    )
    siglip_item = ConversationItem(
        type="image",
        value=torch.arange(12, dtype=torch.float32).reshape(3, hidden_size),
        role="user",
        meta={BAGEL_CONTEXT_KEY: BAGEL_SIGLIP_CONTEXT},
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


def test_bagel_vae_online_process_runs_on_a_meta_built_model() -> None:
    """``offline_training`` builds the VAE on meta with no weights, because it mixes in ``OfflineEncodingMixin``."""
    encode_model = _tiny_vae()
    process_model = _tiny_vae(meta=True)
    encoded_cache = encode_model.offline_encode(pixel_values=torch.zeros(1, 3, 8, 8))["encoded_cache"]
    item_cache = encoded_cache[0].reshape(2, process_model.config.z_channels, *encoded_cache.shape[-2:])

    assert process_model.encoder.conv_in.weight.is_meta
    latents = process_model.online_process(encoded_cache=item_cache)["latents"]
    assert isinstance(latents, list)
    assert latents[0].shape == item_cache.shape[1:]
    assert not latents[0].is_meta


def test_bagel_vae_online_process_consumes_variable_size_cache_items_without_padding() -> None:
    """Verify ``online_process`` samples variable-size cached posteriors without spatial batch padding."""
    model = _tiny_vae(meta=True)
    first = ConversationItem(
        type="image",
        value=torch.zeros(2, 2, 2, 1),
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT},
    )
    second = ConversationItem(
        type="image",
        value=torch.zeros(2, 2, 2, 2),
        role="assistant",
        meta={BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT},
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


def test_bagel_vae_offline_cache_replay_keeps_img_tag_for_flow_connector(tmp_path) -> None:
    """Verify offline VAE cache parquet round-trip preserves ``_img_tag`` for downstream flow connector routing."""
    from datasets import load_dataset

    from veomni.data.seed_omni.seedomni_transform import process_seedomni_cached_example
    from veomni.models.seed_omni.utils.offline_cache import SeedOmniOfflineCacheWriter

    encode_model = _tiny_vae()
    process_model = _tiny_vae(meta=True)

    def vae_image(tag: str) -> ConversationItem:
        return ConversationItem(
            type="image",
            value=torch.zeros(3, 4, 4),
            role="assistant",
            meta={
                BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT,
                _IMG_TAG_KEY: tag,
                BAGEL_VAE_PIXEL_SHAPE: torch.tensor([4, 4]),
            },
        )

    pre = encode_model.pre_forward("offline_encode", conversation_list=[[vae_image("edit"), vae_image("gen")]])
    out = encode_model.offline_encode(**pre)
    encoded = encode_model.post_forward("offline_encode", **out)["conversation_list"]

    writer = SeedOmniOfflineCacheWriter(str(tmp_path))
    writer.open()
    writer.save_conversation_list(encoded)
    writer.flush()
    dataset = load_dataset("parquet", data_files=[str(tmp_path / "shard_000000.parquet")], split="train")
    replayed = [process_seedomni_cached_example(dataset[0])[0]["conversation_list"]]
    assert all(item.meta[BAGEL_VAE_POSTERIOR] for item in replayed[0])

    pre = process_model.pre_forward("online_process", conversation_list=replayed)
    out = process_model.online_process(**pre)
    processed = process_model.post_forward("online_process", **out)["conversation_list"]

    assert [item.meta[_IMG_TAG_KEY] for item in processed[0]] == ["edit", "gen"]
    assert all(item.meta[BAGEL_CONTEXT_KEY] == BAGEL_VAE_CONTEXT for item in processed[0])

    BagelFlowConnector = model_cls("bagel_flow_connector")
    BagelFlowConnectorConfig = config_cls("bagel_flow_connector")
    flow_connector = BagelFlowConnector(
        BagelFlowConnectorConfig(
            hidden_size=4,
            z_channels=2,
            latent_patch_size=1,
            patch_latent_dim=2,
            max_latent_size=4,
            timestep_frequency_embedding_size=4,
            timestep_shift=1.0,
        )
    )
    edit_item, gen_item = processed[0]
    inputs = flow_connector.embed_latent_pre(conversation_list=processed)

    assert flow_connector._embed_lengths == [16, 16]
    assert torch.equal(inputs["timesteps"][:16], torch.zeros(16))
    assert "flow_velocity_target" not in edit_item.meta
    assert gen_item.meta["flow_velocity_target"].shape == (16, 2)


def _tiny_vae(meta=False, **config_overrides):
    """Construct a tiny ``BagelVAE`` on CPU or meta device with a bound processor."""
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
    with torch.device("meta" if meta else "cpu"):
        model = BagelVAE(BagelVAEConfig(**config_kwargs))
    model._image_processor = BagelVAEProcessor.from_config(model.config)
    return model


def _worker_dummies(conversation_list, role):
    """Return worker-appended ``is_dummy`` image placeholders standing in for ``role`` images."""
    return [it for it in iter_desired_items(conversation_list, types=["image"], roles=[role]) if it.is_dummy]


class _FakeBagelTokenizer:
    """Char-ordinal tokenizer with the marker tokens/ids that ``BagelChatTemplate`` resolves."""

    bos_token = "<s>"
    eos_token = "</s>"
    boi_token = "<boi>"
    eoi_token = "<eoi>"
    bos_token_id = 1
    eos_token_id = 2
    boi_token_id = 3
    eoi_token_id = 4
    pad_token_id = 0
    unk_token_id = -1
    special_tokens = {
        "<s>": 1,
        "</s>": 2,
        "<boi>": 3,
        "<eoi>": 4,
        "<|im_start|>": 5,
        "<|im_end|>": 2,
        "<|vision_start|>": 7,
        "<|vision_end|>": 8,
    }

    def convert_tokens_to_ids(self, token):
        return self.special_tokens.get(token, self.unk_token_id)

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        special_by_id = {v: k for k, v in self.special_tokens.items()}
        return "".join(special_by_id.get(int(t), chr(max(32, int(t)))) for t in token_ids)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        input_ids = []
        index = 0
        while index < len(text):
            matched = False
            for token, token_id in sorted(self.special_tokens.items(), key=lambda item: len(item[0]), reverse=True):
                if text.startswith(token, index):
                    input_ids.append(token_id)
                    index += len(token)
                    matched = True
                    break
            if matched:
                continue
            input_ids.append(ord(text[index]))
            index += 1
        return {"input_ids": input_ids}


def _bagel_template() -> BagelChatTemplate:
    return BagelChatTemplate(_FakeBagelTokenizer())


def _b(conversation_list):
    return {"conversation_list": conversation_list}


def test_bagel_registry_entries_and_processors() -> None:
    """Verify all five BAGEL modules and image processors are registered in the SeedOmni registries."""
    expected = {
        "bagel_text_encoder": ("BagelTextEncoder", "BagelTextEncoderAccelerated"),
        "bagel_siglip_navit": ("BagelSiglipNavit", "BagelSiglipNavitAccelerated"),
        "bagel_qwen2_mot": ("BagelQwen2MoT", "BagelQwen2MoTAccelerated"),
        "bagel_flow_connector": ("BagelFlowConnector", "BagelFlowConnectorAccelerated"),
        "bagel_vae": ("BagelVAE", "BagelVAEAccelerated"),
    }
    for model_type, (native_name, accel_name) in expected.items():
        assert model_type in OMNI_CONFIG_REGISTRY.valid_keys()
        assert OMNI_MODEL_REGISTRY[model_type]().__name__ == native_name
        assert OMNI_ACCELERATED_MODEL_REGISTRY[model_type]().__name__ == accel_name
    assert {"bagel_siglip_navit", "bagel_vae"}.issubset(set(OMNI_PROCESSOR_REGISTRY.valid_keys()))


def test_bagel_text_preprocessor_tokenizes_plain_items_and_is_idempotent() -> None:
    """Verify ``BagelTextEncoderPreprocessor`` tokenizes text items, wraps images with markers, and is idempotent."""
    pre = BagelTextEncoderPreprocessor(_bagel_template())
    batch = [
        [
            ConversationItem(type="text", value="hi", role="user"),
            ConversationItem(
                type="image", value=torch.zeros(3, 4, 4), role="user", meta={BAGEL_CONTEXT_KEY: BAGEL_SIGLIP_CONTEXT}
            ),
            ConversationItem(type="text", value="ok", role="assistant"),
        ]
    ]

    pre(_b(batch))
    user_text, image_start, image, image_end, assistant_text = batch[0]

    assert image.type == "image"
    assert image.meta.get(BAGEL_CONTEXT_KEY) == BAGEL_SIGLIP_CONTEXT
    assert image_start.type == "text"
    assert image_start.meta.get(BAGEL_CONTEXT_KEY) == BAGEL_SIGLIP_CONTEXT
    assert torch.equal(image_start.value, torch.tensor([7]))
    assert image_end.type == "text"
    assert image_end.meta.get(BAGEL_CONTEXT_KEY) == BAGEL_SIGLIP_CONTEXT
    assert torch.equal(image_end.value, torch.tensor([8]))
    assert torch.equal(user_text.value, torch.tensor([5, ord("h"), ord("i"), 2]))
    assert user_text.value.device.type == "cpu"
    assert user_text.meta[BAGEL_TOK] is True
    assert torch.equal(user_text.meta["labels"], torch.full_like(user_text.value, -100))
    assert torch.equal(user_text.meta["attention_mask"], torch.ones_like(user_text.value))

    assert torch.equal(assistant_text.value, torch.tensor([5, ord("o"), ord("k"), 2]))
    assert assistant_text.meta[BAGEL_TOK] is True
    assert torch.equal(assistant_text.meta["labels"], assistant_text.value)

    snapshot = copy.deepcopy(batch)
    pre(_b(batch))
    assert len(batch[0]) == len(snapshot[0])
    for actual, expected in zip(batch[0], snapshot[0]):
        if isinstance(actual.value, torch.Tensor):
            assert torch.equal(actual.value, expected.value)


def test_bagel_siglip_preprocessor_patchifies_and_tags_context() -> None:
    """Verify ``BagelSiglipNavitPreprocessor`` patchifies user images and populates position/length metadata."""
    pre = BagelSiglipNavitPreprocessor(
        BagelSiglipNavitProcessor(
            patch_size=2,
            image_size=4,
            min_image_size=2,
            max_pixels=16,
            vit_max_num_patch_per_side=2,
        ),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(1, 2 * 2 * 3, dtype=torch.bfloat16),
    )
    batch = [
        [
            ConversationItem(
                type="image",
                value=torch.full((3, 4, 4), 7, dtype=torch.uint8),
                role="user",
                meta={BAGEL_CONTEXT_KEY: BAGEL_SIGLIP_CONTEXT},
            )
        ]
    ]

    pre(_b(batch))
    item = batch[0][0]
    assert item.meta.get(BAGEL_CONTEXT_KEY) == BAGEL_SIGLIP_CONTEXT
    assert item.meta[BAGEL_SIGLIP_TOKEN_LEN] == 4
    assert item.meta[BAGEL_SIGLIP_POSITION_IDS].tolist() == [0, 1, 2, 3]
    assert item.value.shape == (4, 2 * 2 * 3)
    assert item.value.dtype == torch.bfloat16


def test_bagel_siglip_preprocessor_appends_per_sample_dummy_for_missing_context() -> None:
    """Verify ``BagelSiglipNavitPreprocessor`` appends a dummy SigLIP image to samples without one."""
    pre = BagelSiglipNavitPreprocessor(
        BagelSiglipNavitProcessor(
            patch_size=2,
            image_size=4,
            min_image_size=2,
            max_pixels=16,
            vit_max_num_patch_per_side=2,
        ),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(1, 2 * 2 * 3, dtype=torch.bfloat16),
    )
    batch = [
        [
            ConversationItem(
                type="image",
                value=torch.full((3, 4, 4), 7, dtype=torch.uint8),
                role="user",
                meta={BAGEL_CONTEXT_KEY: BAGEL_SIGLIP_CONTEXT},
            )
        ],
        [ConversationItem(type="text", value="text-only", role="user")],
    ]

    pre(_b(batch))

    assert len(_worker_dummies(batch, "user")) == 1
    assert len(batch[0]) == 1
    dummy = batch[1][-1]
    assert dummy.type == "image"
    assert dummy.role == "user" and dummy.is_dummy
    assert dummy.meta[_IMG_TAG_KEY] == "und"
    assert dummy.meta[BAGEL_CONTEXT_KEY] == BAGEL_SIGLIP_CONTEXT
    assert dummy.meta[BAGEL_SIGLIP_TOKEN_LEN] == 1
    assert dummy.meta[BAGEL_SIGLIP_POSITION_IDS].tolist() == [0]
    assert dummy.value.shape == (1, 2 * 2 * 3)
    assert dummy.value.dtype == torch.bfloat16


def test_bagel_siglip_preprocessor_keeps_bs4_sample_aligned_for_0_2_4_images() -> None:
    """Verify ``BagelSiglipNavitPreprocessor`` maintains one SigLIP item per sample across mixed 4-sample batches."""
    pre = BagelSiglipNavitPreprocessor(
        BagelSiglipNavitProcessor(
            patch_size=2,
            image_size=4,
            min_image_size=2,
            max_pixels=16,
            vit_max_num_patch_per_side=2,
        ),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(1, 2 * 2 * 3, dtype=torch.bfloat16),
    )

    for real_count in (0, 2, 4):
        batch = []
        for index in range(4):
            if index < real_count:
                batch.append(
                    [
                        ConversationItem(
                            type="image",
                            value=torch.full((3, 4, 4), 7, dtype=torch.uint8),
                            role="user",
                            meta={BAGEL_CONTEXT_KEY: BAGEL_SIGLIP_CONTEXT},
                        )
                    ]
                )
            else:
                batch.append([ConversationItem(type="text", value=f"text-{index}", role="user")])

        pre(_b(batch))

        assert len(_worker_dummies(batch, "user")) == 4 - real_count
        for sample in batch:
            context_items = [
                item
                for item in sample
                if item.type == "image" and item.meta.get(BAGEL_CONTEXT_KEY) == BAGEL_SIGLIP_CONTEXT
            ]
            assert len(context_items) == 1


def test_bagel_vae_preprocessor_leaves_cached_posteriors_alone() -> None:
    """An ``offline_training`` batch: ``offline_encode`` already turned every VAE
    row (its dummies included) into a posterior, so nothing is re-processed or added."""
    pre = BagelVAEPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
        dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
    )
    posterior = torch.randn(2, 4, 1, 1)
    meta = {_IMG_TAG_KEY: "gen", BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT, BAGEL_VAE_POSTERIOR: True}
    batch = [
        [ConversationItem(type="image", value=posterior, role="assistant", meta=dict(meta))],
        [ConversationItem(type="image", value=posterior.clone(), role="assistant", meta=dict(meta), is_dummy=True)],
    ]

    pre(_b(batch))

    assert [len(sample) for sample in batch] == [1, 1]
    assert batch[0][0].value is posterior


def test_bagel_preprocessors_route_inference_edit_prompt_context() -> None:
    """Verify ``infer_edit`` preprocessing expands a user image into both VAE and SigLIP context triplets."""
    text_pre = BagelTextEncoderPreprocessor(_bagel_template())
    siglip_pre = BagelSiglipNavitPreprocessor(
        BagelSiglipNavitProcessor(patch_size=2, image_size=4, min_image_size=2, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(1, 2 * 2 * 3, dtype=torch.bfloat16),
    )
    vae_pre = BagelVAEPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
        dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
    )
    user_image = torch.full((3, 4, 4), 7, dtype=torch.uint8)
    assistant_image = torch.full((3, 4, 4), 9, dtype=torch.uint8)
    batch = [
        [
            ConversationItem(type="text", value="hi", role="user"),
            ConversationItem(type="image", value=user_image.clone(), role="user"),
            ConversationItem(
                type="image", value=assistant_image.clone(), role="assistant", meta={_IMG_TAG_KEY: "gen"}
            ),
        ]
    ]

    for preprocessor in (vae_pre, siglip_pre, text_pre):
        preprocessor(_b(batch), inference=True, generation_kwargs={"infer_type": "infer_edit"})

    sample = batch[0]
    assert [item.type for item in sample] == [
        "text",
        "text",
        "image",
        "text",
        "text",
        "image",
        "text",
        "text",
        "image",
        "text",
    ]
    assert torch.equal(sample[0].value, torch.tensor([5, ord("h"), ord("i"), 2]))
    assert sample[2].meta.get(BAGEL_CONTEXT_KEY) == BAGEL_VAE_CONTEXT
    assert sample[5].meta.get(BAGEL_CONTEXT_KEY) == BAGEL_SIGLIP_CONTEXT
    assert sample[8].meta.get(BAGEL_CONTEXT_KEY) == BAGEL_VAE_CONTEXT
    assert sample[2].value.shape == (3, 4, 4)
    assert sample[2].value.dtype == torch.bfloat16
    assert sample[5].value.shape == (4, 2 * 2 * 3)
    assert sample[5].value.dtype == torch.bfloat16
    assert sample[5].meta[BAGEL_SIGLIP_TOKEN_LEN] == 4
    assert sample[8].value.shape == (3, 4, 4)
    assert sample[8].value.dtype == torch.bfloat16
    assert [sample[i].meta.get(BAGEL_CONTEXT_KEY) for i in [1, 3, 4, 6, 7, 9]] == [
        BAGEL_VAE_CONTEXT,
        BAGEL_VAE_CONTEXT,
        BAGEL_SIGLIP_CONTEXT,
        BAGEL_SIGLIP_CONTEXT,
        BAGEL_VAE_CONTEXT,
        BAGEL_VAE_CONTEXT,
    ]
    assert [int(sample[i].value.numel()) for i in [1, 3, 4, 6, 7, 9]] == [1, 1, 1, 1, 1, 1]


def test_bagel_preprocessors_route_tagged_edit_without_infer_type() -> None:
    """Verify training preprocessing routes ``_img_tag='edit'`` images into VAE and SigLIP copies."""
    text_pre = BagelTextEncoderPreprocessor(_bagel_template())
    siglip_pre = BagelSiglipNavitPreprocessor(
        BagelSiglipNavitProcessor(patch_size=2, image_size=4, min_image_size=2, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(1, 12, dtype=torch.bfloat16),
    )
    vae_pre = BagelVAEPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
        dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
    )
    user_image = torch.full((3, 4, 4), 7, dtype=torch.uint8)
    assistant_image = torch.full((3, 4, 4), 9, dtype=torch.uint8)
    batch = [
        [
            ConversationItem(type="text", value="edit", role="user"),
            ConversationItem(type="image", value=user_image.clone(), role="user", meta={_IMG_TAG_KEY: "edit"}),
            ConversationItem(
                type="image", value=assistant_image.clone(), role="assistant", meta={_IMG_TAG_KEY: "gen"}
            ),
        ]
    ]

    for preprocessor in (vae_pre, siglip_pre, text_pre):
        preprocessor(_b(batch))

    sample = batch[0]
    assert [item.meta.get(BAGEL_CONTEXT_KEY) for item in sample if item.type == "image"] == [
        BAGEL_VAE_CONTEXT,
        BAGEL_SIGLIP_CONTEXT,
        BAGEL_VAE_CONTEXT,
    ]
    assert [item.meta.get(_IMG_TAG_KEY) for item in sample if item.type == "image"] == ["edit", "edit", "gen"]
    vae_source, siglip_source, vae_target = (item for item in sample if item.type == "image")
    assert vae_source.value.shape == (3, 4, 4)
    assert vae_source.value.dtype == torch.bfloat16
    assert siglip_source.value.shape == (4, 2 * 2 * 3)
    assert siglip_source.value.dtype == torch.bfloat16
    assert vae_target.value.shape == (3, 4, 4)
    assert vae_target.value.dtype == torch.bfloat16


def test_bagel_preprocessors_route_inference_und_user_image_to_siglip_only() -> None:
    """Verify ``infer_und`` preprocessing routes user images to SigLIP context only."""
    vae_pre = BagelVAEPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
        dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
    )
    text_pre = BagelTextEncoderPreprocessor(_bagel_template())
    image = torch.full((3, 4, 4), 7, dtype=torch.uint8)
    batch = [[ConversationItem(type="image", value=image.clone(), role="user")]]

    vae_pre(_b(batch), inference=True, generation_kwargs={"infer_type": "infer_und"})
    text_pre(_b(batch), inference=True, generation_kwargs={"infer_type": "infer_und"})

    assert [item.type for item in batch[0]] == ["text", "image", "text"]
    assert batch[0][1].meta.get(BAGEL_CONTEXT_KEY) == BAGEL_SIGLIP_CONTEXT
    assert torch.equal(batch[0][1].value, image)


def test_bagel_vae_preprocessor_appends_per_sample_dummy_for_missing_context() -> None:
    """Verify ``BagelVAEPreprocessor`` appends a shape-matched dummy VAE image to samples without one."""
    pre = BagelVAEPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
        dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
    )
    batch = [
        [
            ConversationItem(
                type="image",
                value=torch.full((3, 4, 4), 9, dtype=torch.uint8),
                role="assistant",
                meta={_IMG_TAG_KEY: "gen"},
            )
        ],
        [ConversationItem(type="text", value="text-only", role="user")],
    ]

    pre(_b(batch))

    real = batch[0][0]
    assert real.meta.get(BAGEL_CONTEXT_KEY) == BAGEL_VAE_CONTEXT
    assert real.value.shape == (3, 4, 4)
    assert real.value.dtype == torch.bfloat16
    assert real.meta[BAGEL_VAE_PIXEL_SHAPE].tolist() == [4, 4]

    assert len(_worker_dummies(batch, "assistant")) == 1
    dummy = batch[1][-1]
    assert dummy.type == "image"
    assert dummy.role == "assistant" and dummy.is_dummy
    assert dummy.meta[_IMG_TAG_KEY] == "gen"
    assert dummy.meta[BAGEL_CONTEXT_KEY] == BAGEL_VAE_CONTEXT
    assert dummy.meta[BAGEL_VAE_PIXEL_SHAPE].tolist() == [4, 4]
    assert dummy.value.shape == real.value.shape
    assert dummy.value.dtype == torch.bfloat16
    torch.stack([real.value, dummy.value], dim=0)


def test_bagel_vae_preprocessor_keeps_bs4_sample_aligned_for_0_2_4_images() -> None:
    """Verify ``BagelVAEPreprocessor`` maintains uniform tensor shapes across mixed 4-sample batches."""
    pre = BagelVAEPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
        dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
    )

    for real_count in (0, 2, 4):
        batch = []
        for index in range(4):
            if index < real_count:
                batch.append(
                    [
                        ConversationItem(
                            type="image",
                            value=torch.full((3, 4, 4), 9, dtype=torch.uint8),
                            role="assistant",
                            meta={_IMG_TAG_KEY: "gen"},
                        )
                    ]
                )
            else:
                batch.append([ConversationItem(type="text", value=f"text-{index}", role="user")])

        pre(_b(batch))

        assert len(_worker_dummies(batch, "assistant")) == 4 - real_count
        for sample in batch:
            context_items = [
                item
                for item in sample
                if item.type == "image" and item.meta.get(BAGEL_CONTEXT_KEY) == BAGEL_VAE_CONTEXT
            ]
            assert len(context_items) == 1
        shapes = [tuple(sample[-1].value.shape) for sample in batch]
        assert len(set(shapes)) == 1
        expected_shape = (3, 2, 2) if real_count == 0 else (3, 4, 4)
        assert shapes[0] == expected_shape


def test_bagel_preprocessors_are_picklable() -> None:
    """Verify all three BAGEL worker preprocessors round-trip through ``pickle``."""
    for pre in (
        BagelTextEncoderPreprocessor(_bagel_template()),
        BagelSiglipNavitPreprocessor(
            BagelSiglipNavitProcessor(patch_size=2, image_size=4, min_image_size=2, max_pixels=16),
            dtype=torch.bfloat16,
            dummy_pixel_values=torch.zeros(1, 2 * 2 * 3, dtype=torch.bfloat16),
        ),
        BagelVAEPreprocessor(
            BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
            dtype=torch.bfloat16,
            dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
            dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
        ),
    ):
        restored = pickle.loads(pickle.dumps(pre))
        assert type(restored) is type(pre)


def test_bagel_preprocessors_populate_omni_token_len_for_dyn_bsz() -> None:
    """Verify BAGEL text, SigLIP, and VAE preprocessors populate ``_omni_token_len`` on real and dummy items."""
    from veomni.data.dataset import get_length_by_labels_fn, get_length_fn
    from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN

    text_pre = BagelTextEncoderPreprocessor(_bagel_template())
    siglip_pre = BagelSiglipNavitPreprocessor(
        BagelSiglipNavitProcessor(patch_size=2, image_size=4, min_image_size=2, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(1, 2 * 2 * 3, dtype=torch.bfloat16),
    )
    vae_pre = BagelVAEPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
        dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
    )
    gen_sample = [
        ConversationItem(type="text", value="draw", role="user"),
        ConversationItem(
            type="image",
            value=torch.full((3, 4, 4), 9, dtype=torch.uint8),
            role="assistant",
            meta={_IMG_TAG_KEY: "gen"},
        ),
    ]
    batch = [gen_sample]
    for pre in (vae_pre, siglip_pre, text_pre):
        pre(_b(batch))

    assert all(isinstance(item.meta.get(_OMNI_TOKEN_LEN), int) and item.meta[_OMNI_TOKEN_LEN] > 0 for item in batch[0])
    # User text (6 tokens) + VAE start/end markers (2 tokens) + 4 VAE tokens + 1 dummy SigLIP token = 13 total
    assert get_length_fn({"conversation_list": batch[0]}) == 13
    # Assistant VAE image (4 tokens) is supervised; user text and dummy SigLIP are not
    assert get_length_by_labels_fn({"conversation_list": batch[0]}) == 4


def test_bagel_siglip_preprocessor_from_pretrained_applies_config_overrides(tmp_path) -> None:
    """Verify ``BagelSiglipNavitPreprocessor.from_pretrained`` applies ``config_overrides`` to processor geometry."""
    from veomni.models.seed_omni.modules.bagel.siglip_navit.configuration import BagelSiglipNavitConfig

    BagelSiglipNavitConfig(image_size=28, min_image_size=14, patch_size=14).save_pretrained(tmp_path)
    pre = BagelSiglipNavitPreprocessor.from_pretrained(
        str(tmp_path),
        config_overrides={"image_size": 56, "min_image_size": 28},
    )
    assert pre._image_processor.image_size == 56
    assert pre._image_processor.min_image_size == 28


def test_bagel_vae_encode_and_offline_encode_preserve_native_shapes_and_match_independent_posteriors() -> None:
    """Verify ``encode`` and ``offline_encode`` group equal shapes without spatial padding and match independent encoding."""
    pre = BagelVAEPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=8, max_pixels=64),
        dtype=torch.float32,
        dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.float32),
        dummy_pixel_shape=torch.tensor([4, 4], dtype=torch.long),
    )
    torch.manual_seed(42)
    sample_rect_a = [
        ConversationItem(
            type="image",
            value=torch.randint(0, 255, (3, 4, 8), dtype=torch.uint8),
            role="assistant",
            meta={_IMG_TAG_KEY: "gen"},
        )
    ]
    sample_rect_b = [
        ConversationItem(
            type="image",
            value=torch.randint(0, 255, (3, 8, 4), dtype=torch.uint8),
            role="assistant",
            meta={_IMG_TAG_KEY: "gen"},
        )
    ]
    sample_rect_c = [
        ConversationItem(
            type="image",
            value=torch.randint(0, 255, (3, 4, 8), dtype=torch.uint8),
            role="assistant",
            meta={_IMG_TAG_KEY: "gen"},
        )
    ]
    sample_text_only = [ConversationItem(type="text", value="hello", role="user")]
    for sample in (sample_rect_a, sample_rect_b, sample_rect_c, sample_text_only):
        pre(_b([sample]))

    assert sample_rect_a[0].value.shape == (3, 4, 8)
    assert sample_rect_b[0].value.shape == (3, 8, 4)
    assert sample_rect_c[0].value.shape == (3, 4, 8)
    assert sample_text_only[-1].is_dummy
    assert sample_text_only[-1].value.shape == (3, 4, 4)

    vae = _tiny_vae(ch_mult=[1, 1], downsample=2)

    # Independent single-image offline_encode oracle
    independent_posteriors = []
    for sample in (sample_rect_a, sample_rect_b, sample_rect_c, sample_text_only):
        single = copy.deepcopy([sample])
        single_pre = vae.pre_forward("offline_encode", conversation_list=single)
        single_out = vae.offline_encode(**single_pre)
        vae.post_forward("offline_encode", **single_out)
        independent_posteriors.append(single[0][-1].value)

    # 1. Offline encode with ragged real + dummy micro-batch
    offline_batch = copy.deepcopy([sample_rect_a, sample_rect_b, sample_rect_c, sample_text_only])
    off_pre = vae.pre_forward("offline_encode", conversation_list=offline_batch)
    assert isinstance(off_pre["pixel_values"], list)
    assert [tuple(t.shape) for t in off_pre["pixel_values"]] == [(3, 4, 8), (3, 8, 4), (3, 4, 8), (3, 4, 4)]
    off_out = vae.offline_encode(**off_pre)
    vae.post_forward("offline_encode", **off_out)
    assert offline_batch[0][0].value.shape == (2, 2, 2, 4)
    assert offline_batch[1][0].value.shape == (2, 2, 4, 2)
    assert offline_batch[2][0].value.shape == (2, 2, 2, 4)
    assert offline_batch[3][-1].value.shape == (2, 2, 2, 2)
    for idx, expected_posterior in enumerate(independent_posteriors):
        torch.testing.assert_close(offline_batch[idx][-1].value, expected_posterior)

    # 2. Online encode with ragged real + dummy micro-batch (matching RNG sequence)
    torch.manual_seed(777)
    independent_latents = []
    for sample in (sample_rect_a, sample_rect_b, sample_rect_c, sample_text_only):
        single = copy.deepcopy([sample])
        single_pre = vae.pre_forward("encode", conversation_list=single)
        single_out = vae.encode(**single_pre)
        vae.post_forward("encode", **single_out)
        independent_latents.append(single[0][-1].value)

    torch.manual_seed(777)
    online_batch = copy.deepcopy([sample_rect_a, sample_rect_b, sample_rect_c, sample_text_only])
    pre_inputs = vae.pre_forward("encode", conversation_list=online_batch)
    encoded_out = vae.encode(**pre_inputs)
    vae.post_forward("encode", **encoded_out)
    assert online_batch[0][0].value.shape == (2, 2, 4)
    assert online_batch[1][0].value.shape == (2, 4, 2)
    assert online_batch[2][0].value.shape == (2, 2, 4)
    assert online_batch[3][-1].value.shape == (2, 2, 2)
    for idx, expected_latent in enumerate(independent_latents):
        torch.testing.assert_close(online_batch[idx][-1].value, expected_latent)


def test_bagel_flow_decode_velocity_reduces_loss_and_token_count_over_fsdp_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ``decode_velocity_post`` reduces flow loss and valid-token counts over ``fsdp_group`` including zero-target ranks."""
    import torch.distributed as dist

    from veomni.models.seed_omni.modules.bagel.flow_connector.accelerated import accelerated as flow_acc

    BagelFlowConnector = model_cls("bagel_flow_connector")
    BagelFlowConnectorConfig = config_cls("bagel_flow_connector")
    model = BagelFlowConnector(
        BagelFlowConnectorConfig(
            hidden_size=4,
            z_channels=1,
            latent_patch_size=1,
            patch_latent_dim=2,
            max_latent_size=4,
            timestep_frequency_embedding_size=4,
            timestep_shift=1.0,
        )
    )
    fake_fsdp_group = object()
    monkeypatch.setattr(
        flow_acc,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_size=1, cp_size=1, sp_group=None, fsdp_group=fake_fsdp_group),
    )
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)

    # Case 1: Generation rank (4 tokens) paired with a text-only rank (0 tokens on peer)
    peer_loss_sum = 0.0
    peer_tokens = 0

    def _fake_all_reduce_gen_vs_text(tensor: torch.Tensor, group=None):
        assert group is fake_fsdp_group
        if tensor.dtype == torch.long:
            tensor.add_(peer_tokens)
        else:
            tensor.add_(peer_loss_sum)

    monkeypatch.setattr(dist, "all_reduce", _fake_all_reduce_gen_vs_text)

    gen_item = ConversationItem(
        type="image",
        value=torch.ones(4, 4),
        role="assistant",
        meta={
            BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT,
            _IMG_TAG_KEY: "gen",
            "flow_velocity_target": torch.zeros(4, 2),
        },
    )
    gen_conv = [[gen_item]]
    model.decode_velocity_pre(conversation_list=gen_conv)
    pred_vel = torch.full((4, 2), 2.0, requires_grad=True)
    gen_out = model.decode_velocity_post(pred_vel)
    assert gen_out["_loss"].item() == pytest.approx(4.0)
    gen_out["_loss"].backward()
    # Single-rank unreduced grad per element is 2 * (2 - 0) / (4 * 2) = 0.5;
    # with world_size=2 and peer_tokens=0, ReduceLoss scales backward by 2.0 so FSDP's /2 preserves 0.5.
    torch.testing.assert_close(pred_vel.grad, torch.full_like(pred_vel, 1.0))

    # Zero-target rank paired with the 4-token generation rank on peer (peer_loss_sum = 16.0, peer_tokens = 4)
    peer_loss_sum = 16.0
    peer_tokens = 4
    text_conv = [[ConversationItem(type="text", value=torch.ones(2, 4), role="user")]]
    model.decode_velocity_pre(conversation_list=text_conv)
    dummy_vel = torch.ones(1, 2, requires_grad=True)
    zero_out = model.decode_velocity_post(dummy_vel)
    assert zero_out["_loss"].item() == pytest.approx(4.0)
    zero_out["_loss"].backward()
    torch.testing.assert_close(dummy_vel.grad, torch.zeros_like(dummy_vel))


def test_bagel_chat_template_rejects_multi_token_start_or_end_markers() -> None:
    """Verify ``BagelChatTemplate.tokenize`` raises when ``<|im_start|>`` or ``<|im_end|>`` splits into multiple tokens."""

    class _SplittingTokenizer(_FakeBagelTokenizer):
        def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
            assert add_special_tokens is False
            if text in {"<|im_start|>", "<|im_end|>"}:
                return {"input_ids": [1, 2]}
            return super().__call__(text, add_special_tokens=add_special_tokens)

    template = BagelChatTemplate(_SplittingTokenizer())
    for marker in ("<|im_start|>", "<|im_end|>"):
        with pytest.raises(ValueError, match="BAGEL start/end tokens must tokenize to exactly one token"):
            template.tokenize([ConversationItem(type="text", value=marker, role="user")])


def test_bagel_vae_aligns_encoder_passes_across_shard_group_and_keeps_blocks_unsplit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ``BagelVAE`` keeps ``_no_split_modules`` empty and aligns ``_encode_posterior_list`` pass counts across ``fsdp_group``."""
    import torch.distributed as dist

    from veomni.models.seed_omni.modules.bagel.vae.accelerated import accelerated as vae_acc

    native_vae = native_model_cls("bagel_vae")(
        config_cls("bagel_vae")(resolution=8, ch=32, ch_mult=[1, 1], num_res_blocks=1, z_channels=2, downsample=2)
    )
    assert set(native_vae._no_split_modules) == {"ResnetBlock", "AttnBlock"}

    vae = _tiny_vae(ch_mult=[1, 1], downsample=2)
    assert not vae._no_split_modules

    # Reference single-rank posterior for two equal-shape images (1 shape group natively)
    img_a = torch.randn(3, 4, 8)
    img_b = torch.randn(3, 4, 8)
    ref_out = vae.offline_encode([img_a, img_b])["encoded_cache"]

    fake_fsdp_group = object()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        vae_acc,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_size=2, cp_size=1, sp_group=object(), fsdp_group=fake_fsdp_group),
    )

    def _fake_all_reduce_max(tensor: torch.Tensor, op=None, group=None):
        assert group is fake_fsdp_group
        assert op == dist.ReduceOp.MAX
        # Peer rank has 3 distinct image shapes, while this rank has only 1 shape group.
        tensor.fill_(3)

    monkeypatch.setattr(dist, "all_reduce", _fake_all_reduce_max)

    encoder_calls = 0
    orig_encoder_forward = vae.encoder.forward

    def _counting_encoder_forward(x: torch.Tensor) -> torch.Tensor:
        nonlocal encoder_calls
        encoder_calls += 1
        return orig_encoder_forward(x)

    monkeypatch.setattr(vae.encoder, "forward", _counting_encoder_forward)

    aligned_out = vae.offline_encode([img_a, img_b])["encoded_cache"]
    assert encoder_calls == 3
    for actual, expected in zip(aligned_out, ref_out, strict=True):
        torch.testing.assert_close(actual, expected)


def test_bagel_vae_sp_samples_posteriors_after_gather_and_keeps_flow_rng_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SP-sharded VAE encoding defers posterior sampling until after gather so uneven image sizes keep flow RNG aligned across SP ranks."""
    from veomni.models.seed_omni.modules.bagel.flow_connector.accelerated import accelerated as flow_acc
    from veomni.models.seed_omni.modules.bagel.vae.accelerated import accelerated as vae_acc

    vae = _tiny_vae(ch_mult=[1, 1], downsample=2)
    BagelFlowConnector = model_cls("bagel_flow_connector")
    BagelFlowConnectorConfig = config_cls("bagel_flow_connector")
    flow = BagelFlowConnector(
        BagelFlowConnectorConfig(
            hidden_size=4,
            z_channels=2,
            latent_patch_size=1,
            patch_latent_dim=2,
            max_latent_size=8,
            timestep_frequency_embedding_size=4,
            timestep_shift=3.0,
        )
    )

    torch.manual_seed(99)
    small_img = torch.randn(3, 4, 4)
    large_img = torch.randn(3, 8, 8)

    def _make_batch() -> list[list[ConversationItem]]:
        return [
            [
                ConversationItem(
                    type="image",
                    value=small_img.clone(),
                    role="assistant",
                    meta={
                        BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT,
                        _IMG_TAG_KEY: "gen",
                        BAGEL_VAE_PIXEL_SHAPE: torch.tensor([4, 4], dtype=torch.long),
                    },
                ),
                ConversationItem(
                    type="image",
                    value=large_img.clone(),
                    role="assistant",
                    meta={
                        BAGEL_CONTEXT_KEY: BAGEL_VAE_CONTEXT,
                        _IMG_TAG_KEY: "gen",
                        BAGEL_VAE_PIXEL_SHAPE: torch.tensor([8, 8], dtype=torch.long),
                    },
                ),
            ]
        ]

    # 1. Baseline sp_size=1 run
    monkeypatch.setattr(
        vae_acc,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_size=1, sp_rank=0, cp_size=1, sp_group=None, fsdp_group=None),
    )
    monkeypatch.setattr(
        flow_acc,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_size=1, sp_rank=0, cp_size=1, sp_group=None, fsdp_group=None),
    )
    ref_batch = _make_batch()
    torch.manual_seed(2026)
    ref_pre = vae.pre_forward("encode", conversation_list=ref_batch)
    ref_enc = vae.encode(**ref_pre)
    vae.post_forward("encode", **ref_enc)
    ref_rng_after_vae = torch.get_rng_state()
    ref_flow_inputs = flow.pre_forward("embed_latent", conversation_list=ref_batch)

    # 2. Simulate 2-rank SP where rank 0 encodes small_img (4x4) and rank 1 encodes large_img (8x8)
    rank_caches: dict[int, list[torch.Tensor]] = {}
    for sp_rank in (0, 1):
        monkeypatch.setattr(
            vae_acc,
            "get_parallel_state",
            lambda r=sp_rank: SimpleNamespace(sp_size=2, sp_rank=r, cp_size=1, sp_group=object(), fsdp_group=None),
        )
        batch_r = _make_batch()
        torch.manual_seed(2026)
        pre_r = vae.pre_forward("encode", conversation_list=batch_r)
        enc_r = vae.encode(**pre_r)
        # No RNG must be consumed during the SP-sharded encode step before gather
        assert torch.equal(torch.get_rng_state(), torch.manual_seed(2026).get_state())
        rank_caches[sp_rank] = enc_r["latents"]

    def _fake_gather_outputs(x: torch.Tensor, gather_dim: int = 0, group=None) -> torch.Tensor:
        del group
        assert gather_dim == 0
        # Pad rank 0's flat posterior tensor to match max_rank_numel (rank 1's 2*2*4*4 = 64 elements)
        flat_0 = rank_caches[0][0].reshape(-1)
        flat_1 = rank_caches[1][0].reshape(-1)
        max_numel = max(int(flat_0.numel()), int(flat_1.numel()))
        pad_0 = torch.nn.functional.pad(flat_0, (0, max_numel - int(flat_0.numel()))).unsqueeze(0)
        pad_1 = torch.nn.functional.pad(flat_1, (0, max_numel - int(flat_1.numel()))).unsqueeze(0)
        return torch.cat([pad_0, pad_1], dim=0)

    monkeypatch.setattr(vae_acc, "gather_outputs", _fake_gather_outputs)
    monkeypatch.setattr(
        flow_acc,
        "slice_sp_token_inputs",
        lambda method, **inputs: (int(inputs["latents"].shape[0]), inputs),
    )

    rank_batches: dict[int, list[list[ConversationItem]]] = {}
    rank_flow_inputs: dict[int, dict[str, torch.Tensor]] = {}
    for sp_rank in (0, 1):
        monkeypatch.setattr(
            vae_acc,
            "get_parallel_state",
            lambda r=sp_rank: SimpleNamespace(sp_size=2, sp_rank=r, cp_size=1, sp_group=object(), fsdp_group=None),
        )
        monkeypatch.setattr(
            flow_acc,
            "get_parallel_state",
            lambda r=sp_rank: SimpleNamespace(sp_size=2, sp_rank=r, cp_size=1, sp_group=object(), fsdp_group=None),
        )
        batch_r = _make_batch()
        torch.manual_seed(2026)
        pre_r = vae.pre_forward("encode", conversation_list=batch_r)
        enc_r = vae.encode(**pre_r)
        vae.post_forward("encode", **enc_r)
        assert torch.equal(torch.get_rng_state(), ref_rng_after_vae)
        flow_in_r = flow.pre_forward("embed_latent", conversation_list=batch_r)
        rank_batches[sp_rank] = batch_r
        rank_flow_inputs[sp_rank] = flow_in_r

    for idx in (0, 1):
        torch.testing.assert_close(rank_batches[0][0][idx].value, ref_batch[0][idx].value)
        torch.testing.assert_close(rank_batches[1][0][idx].value, ref_batch[0][idx].value)
        torch.testing.assert_close(
            rank_batches[0][0][idx].meta["flow_velocity_target"],
            rank_batches[1][0][idx].meta["flow_velocity_target"],
        )
        torch.testing.assert_close(
            rank_batches[0][0][idx].meta["flow_velocity_target"],
            ref_batch[0][idx].meta["flow_velocity_target"],
        )
        torch.testing.assert_close(
            rank_batches[0][0][idx].meta["timestep"],
            rank_batches[1][0][idx].meta["timestep"],
        )
    for key in ("latents", "position_ids", "timesteps"):
        torch.testing.assert_close(rank_flow_inputs[0][key], rank_flow_inputs[1][key])
        torch.testing.assert_close(rank_flow_inputs[0][key], ref_flow_inputs[key])
