"""Janus CPU preprocessors: per-item token counts, dummy shapes, codec freezing.

Two properties matter for a training run:

* Every image row states how many backbone tokens it contributes
  (``item.meta[_omni_token_len]``). ``dyn_bsz`` measures a sample before
  collation, and an image row without the key makes the image span invisible to
  the token budget. The count comes from the vision / VQ config, because
  ``JanusImageProcessor`` carries no ``patch_size`` of its own.
* The per-sample FSDP-anchor dummy is stacked with real images in a mixed batch,
  so it must be shaped like what the image processor emits.
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.janus.image_processing_janus import JanusImageProcessor

from veomni.models.seed_omni.modules.janus.siglip.configuration import JanusSiglipConfig
from veomni.models.seed_omni.modules.janus.siglip.processing import JanusSiglipPreprocessor
from veomni.models.seed_omni.modules.janus.vqvae.configuration import JanusVqvaeConfig
from veomni.models.seed_omni.modules.janus.vqvae.modeling import JanusVqvae
from veomni.models.seed_omni.modules.janus.vqvae.processing import JanusVqvaePreprocessor
from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN, ConversationItem


IMAGE_SIZE = 32
PATCH_SIZE = 16
IMAGE_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2


def _processor() -> JanusImageProcessor:
    """``size`` (not ``image_size``) is what the processor actually emits at.

    Deliberately no ``patch_size``: the one the converter saves has none either.
    """
    return JanusImageProcessor(size={"height": IMAGE_SIZE, "width": IMAGE_SIZE})


def _siglip() -> JanusSiglipPreprocessor:
    processor = _processor()
    pre = JanusSiglipPreprocessor(processor, dtype=torch.bfloat16)
    pre.bind_dummy_inputs(
        JanusSiglipConfig(vision_config={**processor.to_dict(), "image_size": IMAGE_SIZE}), torch.bfloat16
    )
    return pre


def _vqvae() -> JanusVqvaePreprocessor:
    pre = JanusVqvaePreprocessor(_processor(), dtype=torch.bfloat16)
    pre.bind_dummy_inputs(JanusVqvaeConfig(vq_config={"num_patches": 24}), torch.bfloat16)
    return pre


def test_siglip_records_the_patch_count_and_rejects_a_size_mismatch():
    """The count comes from the config; a mismatched processor size fails loudly.

    A mixed real/dummy batch is ``torch.stack``ed, so a ``vision_config`` override
    that moves ``image_size`` away from ``preprocessor_config.json`` has to be
    rejected rather than paired with real images at the first step.
    """
    pre = _siglip()
    real = [ConversationItem(type="image", value=torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), role="user")]
    empty = []
    pre.preprocess_conversations([real, empty])

    assert real[0].meta[_OMNI_TOKEN_LEN] == IMAGE_TOKENS
    assert empty[-1].is_dummy and empty[-1].meta[_OMNI_TOKEN_LEN] == IMAGE_TOKENS
    assert torch.stack([real[0].value, empty[-1].value]).shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)

    mismatched = JanusSiglipPreprocessor(_processor(), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="does not match the image processor"):
        mismatched.bind_dummy_inputs(
            JanusSiglipConfig(vision_config={**_processor().to_dict(), "image_size": IMAGE_SIZE + 1}),
            torch.bfloat16,
        )


def test_siglip_refuses_to_count_tokens_without_a_patch_size():
    """Neither the config nor the processor states one: raise, do not guess."""
    pre = JanusSiglipPreprocessor(_processor(), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="patch size unknown"):
        _ = pre._num_patch_tokens


def test_vqvae_records_the_configured_grid_and_refuses_an_unfed_trainable_codec():
    """The token count follows ``num_patches``; ``freeze=False`` is rejected.

    ``JanusVqvaeProcessor.num_image_tokens`` is a fixed 576, correct only for the
    384/16 grid. And the graphs consume just the discrete VQ tokens, so a
    trainable codec would open gradients with no loss path into it.
    """
    pre = _vqvae()
    real = [ConversationItem(type="image", value=torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), role="assistant")]
    empty = []
    pre.preprocess_conversations([real, empty])
    assert real[0].meta[_OMNI_TOKEN_LEN] == 576
    assert empty[-1].meta[_OMNI_TOKEN_LEN] == 576

    pre.bind_dummy_inputs(JanusVqvaeConfig(vq_config={"num_patches": 2}), torch.bfloat16)
    assert pre._num_image_tokens == 4

    # The default grid matches what the processor emits.
    assert JanusVqvaeConfig().vq_config.num_patches ** 2 == 576

    model = JanusVqvae.__new__(JanusVqvae)
    model.config = JanusVqvaeConfig(vq_config={"num_patches": 24}, freeze=False)
    with pytest.raises(NotImplementedError, match="freeze=False is unsupported"):
        model._encode_pixels(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
