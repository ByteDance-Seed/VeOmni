"""Janus CPU preprocessors: the per-item token count and the dummy shape.

Two properties matter for a training run:

* Every image row states how many backbone tokens it contributes
  (``item.meta[_omni_token_len]``). ``dyn_bsz`` measures a sample before
  collation, and an image row without the key makes the image span invisible to
  the token budget.
* The per-sample FSDP-anchor dummy is stacked with real images in a mixed batch,
  so it must be shaped like what the image processor emits. A ``vision_config``
  override that silently disagrees with ``preprocessor_config.json`` is rejected
  rather than left to fail at the first ``torch.stack``.
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.janus.image_processing_janus import JanusImageProcessor

from veomni.models.seed_omni.modules.janus.siglip.configuration import JanusSiglipConfig
from veomni.models.seed_omni.modules.janus.siglip.processing import JanusSiglipPreprocessor
from veomni.models.seed_omni.modules.janus.vqvae.configuration import JanusVqvaeConfig
from veomni.models.seed_omni.modules.janus.vqvae.processing import JanusVqvaePreprocessor
from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN, ConversationItem


IMAGE_SIZE = 32
PATCH_SIZE = 16
IMAGE_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2


def _processor() -> JanusImageProcessor:
    """``size`` (not ``image_size``) is what the processor actually emits at."""
    return JanusImageProcessor(size={"height": IMAGE_SIZE, "width": IMAGE_SIZE}, patch_size=PATCH_SIZE)


def _siglip() -> JanusSiglipPreprocessor:
    processor = _processor()
    pre = JanusSiglipPreprocessor(processor, dtype=torch.bfloat16)
    # ``image_size`` is a separate field from ``size``; the config must agree with
    # what the processor emits, so set both.
    pre.bind_dummy_inputs(
        JanusSiglipConfig(vision_config={**processor.to_dict(), "image_size": IMAGE_SIZE}), torch.bfloat16
    )
    return pre


def _vqvae() -> JanusVqvaePreprocessor:
    processor = _processor()
    pre = JanusVqvaePreprocessor(processor, dtype=torch.bfloat16)
    pre.bind_dummy_inputs(JanusVqvaeConfig(vq_config={"num_patches": IMAGE_SIZE // PATCH_SIZE}), torch.bfloat16)
    return pre


def test_siglip_records_the_patch_count_on_real_and_dummy_rows():
    pre = _siglip()
    real = [ConversationItem(type="image", value=torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), role="user")]
    empty = []

    pre.preprocess_conversations([real, empty])

    assert real[0].meta[_OMNI_TOKEN_LEN] == IMAGE_TOKENS
    # A text-only sample gets the FSDP-anchor dummy, which is a backbone row set too.
    assert empty[-1].is_dummy
    assert empty[-1].meta[_OMNI_TOKEN_LEN] == IMAGE_TOKENS


def test_vqvae_records_the_vq_token_count_on_real_and_dummy_rows():
    pre = _vqvae()
    real = [ConversationItem(type="image", value=torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), role="assistant")]
    empty = []

    pre.preprocess_conversations([real, empty])

    expected = pre._num_image_tokens
    assert real[0].meta[_OMNI_TOKEN_LEN] == expected
    assert empty[-1].is_dummy
    assert empty[-1].meta[_OMNI_TOKEN_LEN] == expected


def test_siglip_rejects_a_vision_size_that_disagrees_with_the_processor():
    """A mixed real/dummy batch stacks both, so the shapes have to match."""
    processor = _processor()
    pre = JanusSiglipPreprocessor(processor, dtype=torch.bfloat16)
    mismatched = JanusSiglipConfig(vision_config={**processor.to_dict(), "image_size": IMAGE_SIZE + 1})

    with pytest.raises(ValueError, match="does not match the image processor"):
        pre.bind_dummy_inputs(mismatched, torch.bfloat16)


def test_siglip_dummy_matches_a_real_image_shape():
    pre = _siglip()
    real = [ConversationItem(type="image", value=torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), role="user")]
    empty = []
    pre.preprocess_conversations([real, empty])

    stacked = torch.stack([real[0].value, empty[-1].value])
    assert stacked.shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_siglip_patch_count_does_not_depend_on_a_processor_patch_size():
    """``JanusImageProcessor`` normally has no ``patch_size``; the config supplies it.

    A processor loaded from a real checkpoint (and the one the family converter
    saves) does not carry the field. Reading it off the processor would raise, or
    with a permissive fallback count one token per pixel — 147456 instead of 576
    at 384/16. Build the processor the way a checkpoint does: no ``patch_size``.
    """
    processor = JanusImageProcessor(size={"height": IMAGE_SIZE, "width": IMAGE_SIZE})
    assert not hasattr(processor, "patch_size")

    pre = JanusSiglipPreprocessor(processor, dtype=torch.bfloat16)
    config = JanusSiglipConfig(vision_config={**processor.to_dict(), "image_size": IMAGE_SIZE})
    pre.bind_dummy_inputs(config, torch.bfloat16)

    assert pre._num_patch_tokens == IMAGE_TOKENS


def test_siglip_refuses_to_count_tokens_before_the_patch_size_is_known():
    processor = _processor()
    pre = JanusSiglipPreprocessor(processor, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="patch size unknown"):
        _ = pre._num_patch_tokens


def test_vqvae_token_count_follows_the_configured_grid_not_the_class_default():
    """``JanusVqvaeProcessor.num_image_tokens`` is a fixed 576; the grid decides."""
    processor = _processor()
    pre = JanusVqvaePreprocessor(processor, dtype=torch.bfloat16)
    # num_patches is the grid side length, so 24 -> 576 tokens.
    pre.bind_dummy_inputs(JanusVqvaeConfig(vq_config={"num_patches": 24}), torch.bfloat16)
    assert pre._num_image_tokens == 576

    pre.bind_dummy_inputs(JanusVqvaeConfig(vq_config={"num_patches": 2}), torch.bfloat16)
    assert pre._num_image_tokens == 4


def test_vqvae_config_defaults_to_the_grid_the_processor_emits():
    """HF's ``num_patches=32`` grid expects 1024 tokens; the processor emits 576."""
    config = JanusVqvaeConfig()
    assert config.vq_config.num_patches**2 == 576


def test_the_codec_refuses_a_trainable_setting_it_cannot_feed():
    """``freeze=False`` would open gradients with no loss path into the codec.

    The graphs consume only the discrete VQ tokens, so ``embedding_loss`` never
    reaches the total; the parameters would look trainable while nothing updated
    them. Rejecting is better than silently training nothing.
    """
    from veomni.models.seed_omni.modules.janus.vqvae.modeling import JanusVqvae

    config = JanusVqvaeConfig(vq_config={"num_patches": 24}, freeze=False)
    model = JanusVqvae.__new__(JanusVqvae)
    model.config = config

    with pytest.raises(NotImplementedError, match="freeze=False is unsupported"):
        model._encode_pixels(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
