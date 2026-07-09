from __future__ import annotations

import torch

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.modules.bagel.vae.modulemixin import BAGEL_VAE_PIXEL_SHAPE
from veomni.models.seed_omni.modules.bagel.vae.processing import BagelVAEProcessor
from veomni.models.seed_omni.utils.conversation import ConversationItem


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


def test_bagel_vae_processor_reports_unpadded_shapes_for_padded_batch() -> None:
    processor = BagelVAEProcessor(max_image_size=8, min_image_size=4, image_stride=4, max_pixels=64)

    out = processor(
        images=[torch.zeros(3, 8, 8), torch.zeros(3, 8, 4)],
        return_tensors="pt",
        dtype=torch.float32,
    )

    assert out["pixel_values"].shape == (2, 3, 8, 8)
    assert out["pixel_shapes"].tolist() == [[8, 8], [8, 4]]


def test_bagel_vae_modeling_offline_encode_returns_tensor_cache() -> None:
    model = _tiny_vae()
    encoded_cache = model.offline_encode(pixel_values=torch.zeros(1, 3, 8, 8))["encoded_cache"]

    assert torch.is_tensor(encoded_cache)
    assert encoded_cache.shape[0] == 1
    assert encoded_cache.shape[1] == 2 * model.config.z_channels


def test_bagel_vae_offline_encode_hook_writes_unpadded_image_cache_item() -> None:
    model = _tiny_vae(downsample=4)
    item = ConversationItem(
        type="image",
        value=torch.zeros(3, 8, 4),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={BAGEL_VAE_PIXEL_SHAPE: torch.tensor([8, 4])},
    )
    conversation = [[item]]
    model._conversation_carrier = conversation
    model._encode_items = [item]
    encoded_cache = torch.zeros(1, 2, 2, 2, 2)

    out = model.offline_encode_post(encoded_cache=encoded_cache)

    assert out["conversation_list"] is conversation
    assert item.type == "image"
    assert item.source == BAGEL_VAE_CONTEXT
    assert item.meta == {}
    assert item.value.shape == (2, 2, 2, 1)


def test_bagel_vae_offline_encode_hook_reshapes_flattened_posterior_item() -> None:
    model = _tiny_vae(downsample=4)
    item = ConversationItem(
        type="image",
        value=torch.zeros(3, 8, 4),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={BAGEL_VAE_PIXEL_SHAPE: torch.tensor([8, 4])},
    )
    model._conversation_carrier = [[item]]
    model._encode_items = [item]
    encoded_cache = torch.zeros(1, 2 * model.config.z_channels, 2, 2)

    model.offline_encode_post(encoded_cache=encoded_cache)

    assert item.value.shape == (2, model.config.z_channels, 2, 1)


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


def test_bagel_vae_online_process_preserves_cached_dummy_without_duplication() -> None:
    model = _tiny_vae(support_cache=True, train_type="train_with_cache")
    dummy = ConversationItem(
        type="image",
        value=torch.zeros(2, model.config.z_channels, 2, 1),
        role="dummy",
        source=BAGEL_VAE_CONTEXT,
    )
    conversation = [[dummy]]

    pre = model.online_process_pre(conversation_list=conversation)
    out = model.online_process(encoded_cache=pre["encoded_cache"])
    post = model.post_forward("online_process", latents=out["latents"])

    assert post["conversation_list"] is conversation
    assert conversation[0][0].source == BAGEL_VAE_CONTEXT
    assert conversation[0][0].role == "dummy"
    assert conversation[0][0].value.shape == (model.config.z_channels, 2, 1)
    assert conversation[0][0].meta == {}


def test_bagel_vae_online_process_rejects_flattened_item_cache() -> None:
    model = _tiny_vae(support_cache=True, train_type="train_with_cache")

    try:
        model.online_process(encoded_cache=torch.zeros(2 * model.config.z_channels, 2, 1))
    except ValueError as exc:
        assert "posterior cache tensor" in str(exc)
    else:
        raise AssertionError("expected flattened item cache to be rejected")


def test_bagel_vae_online_process_accepts_singleton_batched_item_cache() -> None:
    model = _tiny_vae(support_cache=True, train_type="train_with_cache")

    out = model.online_process(encoded_cache=torch.zeros(1, 2, model.config.z_channels, 2, 1))

    assert isinstance(out["latents"], list)
    assert out["latents"][0].shape == (model.config.z_channels, 2, 1)


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

    try:
        process_model.encode(pixel_values=torch.zeros(1, 3, 8, 8))
    except RuntimeError as exc:
        assert "VAE encoder" in str(exc)
    else:
        raise AssertionError("expected process_only encode to require an encoder")

    try:
        process_model.decode(latents=torch.zeros(1, 2, 2, 2))
    except RuntimeError as exc:
        assert "VAE decoder" in str(exc)
    else:
        raise AssertionError("expected process_only decode to require a decoder")


def test_bagel_vae_encode_only_skips_decoder_module() -> None:
    model = _tiny_vae(support_cache=True, train_type="offline_cache")

    assert hasattr(model, "encoder")
    assert not hasattr(model, "decoder")
    assert model.offline_encode(pixel_values=torch.zeros(1, 3, 8, 8))["encoded_cache"].shape[:2] == (
        1,
        2 * model.config.z_channels,
    )

    try:
        model.decode(latents=torch.zeros(1, 2, 2, 2))
    except RuntimeError as exc:
        assert "VAE decoder" in str(exc)
    else:
        raise AssertionError("expected encode_only decode to require a decoder")


def test_bagel_vae_from_pretrained_accepts_cache_overrides(tmp_path) -> None:
    model = _tiny_vae()
    model.save_pretrained(tmp_path)
    BagelVAEProcessor.from_config(model.config).save_pretrained(tmp_path)

    BagelVAE = model_cls("bagel_vae")
    loaded = BagelVAE.from_pretrained(tmp_path, support_cache=True, train_type="train_with_cache")

    assert loaded.cache_mode == "process_only"
    assert not hasattr(loaded, "encoder")
    assert not hasattr(loaded, "decoder")
    assert loaded._image_processor.max_image_size == 8
