"""Unit tests for the Qwen3-Omni text encoder module (config + wte/lm_head wiring)."""

import torch

from veomni.models.seed_omni.modules.qwen3omni.text_encoder.configuration import Qwen3OmniTextEncoderConfig
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.modeling import Qwen3OmniTextEncoder
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.processing import Qwen3OmniTextEncoderPreprocessor


def test_config_inherits_base_defaults():
    """transformers 5.x auto-generates ``__init__`` for annotation-less config subclasses.

    Without the explicit forward in ``Qwen3OmniTextEncoderConfig`` none of
    ``TextEncoderConfig``'s defaults land, and ``lm_head_bias`` in particular goes
    missing — which breaks the untied path below.
    """
    config = Qwen3OmniTextEncoderConfig()
    assert config.model_type == "qwen3omni_text_encoder"
    assert config.vocab_size == 32000
    assert config.lm_head_bias is False


def test_untied_module_builds_and_decodes():
    """Qwen3-Omni's thinker ships ``tie_word_embeddings=False``, so this is the only path."""
    config = Qwen3OmniTextEncoderConfig(vocab_size=64, hidden_size=32, tie_word_embeddings=False)
    module = Qwen3OmniTextEncoder(config)
    assert isinstance(module.lm_head, torch.nn.Linear)
    assert module.lm_head.bias is None

    # ``encode`` returns per-segment ``(L, D)`` rows for the carrier, not a batch.
    embeds = module.encode(torch.tensor([[1, 2, 3]]))["inputs_embeds"]
    assert embeds.shape == (3, 32)


def test_preprocessor_class_is_reachable_from_the_model():
    """The processing binding resolves the preprocessor off the native class."""
    assert Qwen3OmniTextEncoder.preprocessor_class is Qwen3OmniTextEncoderPreprocessor
