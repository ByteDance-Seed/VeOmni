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
