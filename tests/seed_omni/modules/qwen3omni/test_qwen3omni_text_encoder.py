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


def test_tts_embeds_reads_the_sharded_table_it_actually_has():
    """`tts_embeds` must go through the embedding module, not a raw table read.

    The table is a `ShardedEmbedding`, whose `forward` takes *global* ids and
    performs the `emb` all-to-all itself (falling through to `nn.Embedding` when
    the table is not split). Reading a `_embed_tokens` attribute raises
    `AttributeError` — on both the speech-inference and talker-training paths,
    i.e. before the talker ever runs.
    """
    from unittest import mock

    module = Qwen3OmniTextEncoder(Qwen3OmniTextEncoderConfig(vocab_size=64, hidden_size=32, tie_word_embeddings=False))
    # ``tts_embeds`` reads its ids off the chat template, which a real checkpoint
    # binds; stub just the three ids it needs.
    module._chat_template = mock.MagicMock(tts_bos_token_id=1, tts_eos_token_id=2, tts_pad_token_id=3)

    # No ``_embed_tokens`` exists anywhere in the MRO — the bug was assuming one.
    assert not any("_embed_tokens" in cls.__dict__ for cls in type(module).__mro__)

    with torch.no_grad():
        embeds = module.tts_embeds()

    assert embeds.shape == (1, 3, 32)
    # ...and the ids really reached the table (bos / eos / pad rows, in order).
    with torch.no_grad():
        expected = module.embed_tokens(torch.tensor([[1, 2, 3]]))
    assert torch.equal(embeds, expected)
