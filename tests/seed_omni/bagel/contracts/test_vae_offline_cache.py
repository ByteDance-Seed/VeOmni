from __future__ import annotations

import torch

from tests.seed_omni.bagel.contracts.helpers import config_cls, model_cls
from veomni.models.seed_omni.mixins.offline_encoding import ENCODED_CACHE_KIND_META_KEY
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.modules.bagel.vae.cache import (
    BAGEL_VAE_POSTERIOR_CACHE_KIND,
    BagelVAEPosteriorCache,
)
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


def test_bagel_vae_posterior_cache_roundtrips_tensor_payload() -> None:
    mean = torch.ones(2, 3, 4)
    logvar = torch.full((2, 3, 4), -1.0)

    item_cache = BagelVAEPosteriorCache(mean=mean, logvar=logvar)
    item_tensor = item_cache.to_tensor()
    roundtrip = BagelVAEPosteriorCache.from_tensor(item_tensor)

    assert item_tensor.shape == (2, 2, 3, 4)
    assert torch.equal(roundtrip.mean, mean)
    assert torch.equal(roundtrip.logvar, logvar)

    batch_cache = BagelVAEPosteriorCache(mean=mean.unsqueeze(0), logvar=logvar.unsqueeze(0))
    batch_tensor = batch_cache.to_tensor()
    batch_roundtrip = BagelVAEPosteriorCache.from_tensor(batch_tensor)

    assert batch_tensor.shape == (1, 2, 2, 3, 4)
    assert torch.equal(batch_roundtrip.mean, mean.unsqueeze(0))
    assert torch.equal(batch_roundtrip.logvar, logvar.unsqueeze(0))


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
    assert encoded_cache.shape[1] == 2
    assert encoded_cache.shape[2] == model.config.z_channels


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
    assert item.meta == {ENCODED_CACHE_KIND_META_KEY: BAGEL_VAE_POSTERIOR_CACHE_KIND}
    assert item.value.shape == (2, 2, 2, 1)


def test_bagel_vae_online_process_consumes_variable_size_cache_items_without_padding() -> None:
    model = _tiny_vae(cache_mode="process_only")
    first = ConversationItem(
        type="image",
        value=torch.zeros(2, 2, 2, 1),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={ENCODED_CACHE_KIND_META_KEY: BAGEL_VAE_POSTERIOR_CACHE_KIND},
    )
    second = ConversationItem(
        type="image",
        value=torch.zeros(2, 2, 2, 2),
        role="assistant",
        source=BAGEL_VAE_CONTEXT,
        meta={ENCODED_CACHE_KIND_META_KEY: BAGEL_VAE_POSTERIOR_CACHE_KIND},
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


def test_bagel_vae_process_only_skips_codec_modules() -> None:
    encode_model = _tiny_vae()
    process_model = _tiny_vae(cache_mode="process_only")
    encoded_cache = encode_model.offline_encode(pixel_values=torch.zeros(1, 3, 8, 8))["encoded_cache"]

    assert not hasattr(process_model, "encoder")
    assert not hasattr(process_model, "decoder")
    latents = process_model.online_process(encoded_cache=encoded_cache)["latents"]
    assert isinstance(latents, list)
    assert latents[0].shape == encoded_cache[:, 0].shape

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
    model = _tiny_vae(cache_mode="encode_only")

    assert hasattr(model, "encoder")
    assert not hasattr(model, "decoder")
    assert model.offline_encode(pixel_values=torch.zeros(1, 3, 8, 8))["encoded_cache"].shape[:2] == (1, 2)

    try:
        model.decode(latents=torch.zeros(1, 2, 2, 2))
    except RuntimeError as exc:
        assert "VAE decoder" in str(exc)
    else:
        raise AssertionError("expected encode_only decode to require a decoder")


def test_bagel_vae_from_pretrained_accepts_cache_mode_override(tmp_path) -> None:
    model = _tiny_vae()
    model.save_pretrained(tmp_path)
    BagelVAEProcessor.from_config(model.config).save_pretrained(tmp_path)

    BagelVAE = model_cls("bagel_vae")
    loaded = BagelVAE.from_pretrained(tmp_path, cache_mode="process_only")

    assert loaded.cache_mode == "process_only"
    assert not hasattr(loaded, "encoder")
    assert not hasattr(loaded, "decoder")
    assert loaded._image_processor.max_image_size == 8
