"""Qwen3-Omni ``OMNI_HF_LAYOUT_REGISTRY["qwen3_omni_moe"]``: the module key map.

The layout is the single description of how an upstream
``Qwen3OmniMoeForConditionalGeneration`` checkpoint splits, so both the offline
converter and a direct ``model.model_path`` load read it. What is worth pinning
without a real 67 GB checkpoint is the *routing*, which is where the two
easy-to-get-wrong orderings live:

* ``thinker.model.embed_tokens.`` must beat the backbone's shorter
  ``thinker.model.``, so the vocabulary tensors land in the text encoder;
* ``thinker.lm_head.`` likewise, since the thinker ships an untied head.

A route that resolved either to the backbone would load cleanly (the backbone's
prefix matches) and leave the text encoder on random weights.

The one module this layout cannot supply alone is the codec: Qwen released only
the decoder half, so the encoder comes from ``kyutai/mimi`` and the layout
refuses rather than silently emitting a decode-only codec.
"""

from __future__ import annotations

import pytest

from veomni.models.seed_omni.modules.qwen3omni.hf_layout import QWEN3OMNI_HF_LAYOUT
from veomni.models.seed_omni.utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY


AUDIO = "qwen3omni_audio"
VISION = "qwen3omni_vision"
TEXT_ENCODER = "qwen3omni_text_encoder"
LLM = "qwen3omni_llm"
TALKER = "qwen3omni_talker"
CODEC = "qwen3omni_codec"


def test_qwen3omni_registers_a_layout_beside_its_converter():
    assert "qwen3_omni_moe" in set(OMNI_HF_LAYOUT_REGISTRY.valid_keys())
    assert set(QWEN3OMNI_HF_LAYOUT.modules) == {AUDIO, VISION, TEXT_ENCODER, LLM, TALKER, CODEC}
    # Speech output is what makes this family distinct, so it is the default.
    assert QWEN3OMNI_HF_LAYOUT.infer_type == "audio_output"
    assert set(QWEN3OMNI_HF_LAYOUT.generation_graphs) == {"audio_output", "thinker_only"}
    assert set(QWEN3OMNI_HF_LAYOUT.training_graphs) == {"thinker", "talker"}


def test_the_vocabulary_tensors_go_to_the_text_encoder_not_the_backbone():
    """The two orderings that make the split work, stated as routes.

    Both source keys also match the backbone's shorter ``thinker.model.`` prefix,
    so the longest-prefix rule is the whole mechanism — and a mistake here is
    silent, because the backbone would load either tensor without complaint.
    """
    layout = QWEN3OMNI_HF_LAYOUT

    assert layout.route("thinker.model.embed_tokens.weight") == (TEXT_ENCODER, "embed_tokens.weight")
    assert layout.route("thinker.lm_head.weight") == (TEXT_ENCODER, "lm_head.weight")
    # The backbone keeps the layers and the final norm, renamed onto its own root.
    assert layout.route("thinker.model.layers.0.self_attn.q_proj.weight") == (
        LLM,
        "language_model.layers.0.self_attn.q_proj.weight",
    )
    assert layout.route("thinker.model.norm.weight") == (LLM, "language_model.norm.weight")


def test_each_tower_claims_only_its_own_prefix():
    layout = QWEN3OMNI_HF_LAYOUT

    assert layout.route("thinker.audio_tower.conv2d1.weight") == (AUDIO, "conv2d1.weight")
    assert layout.route("thinker.visual.blocks.0.attn.qkv.weight") == (VISION, "visual.blocks.0.attn.qkv.weight")
    assert layout.route("talker.model.layers.0.self_attn.q_proj.weight") == (
        TALKER,
        "talker.model.layers.0.self_attn.q_proj.weight",
    )
    assert layout.route("code2wav.decoder.0.weight") == (CODEC, "code2wav.decoder.0.weight")


def test_every_routed_key_maps_back_to_itself():
    """An export runs the map backwards; a key that does not round-trip would be
    written under the wrong name."""
    layout = QWEN3OMNI_HF_LAYOUT

    for key in (
        "thinker.audio_tower.conv2d1.weight",
        "thinker.visual.blocks.0.attn.qkv.weight",
        "thinker.model.embed_tokens.weight",
        "thinker.lm_head.weight",
        "thinker.model.layers.1.mlp.gate.weight",
        "thinker.model.norm.weight",
        "talker.model.norm.weight",
        "code2wav.decoder.0.weight",
    ):
        module, module_key = layout.route(key)
        assert layout.source_key(module, module_key) == key


def test_the_codec_refuses_to_be_built_from_one_checkpoint():
    """Half a codec is worse than none: a decode-only module loads cleanly and
    then breaks talker training far from here."""
    from transformers import Qwen3OmniMoeConfig

    with pytest.raises(NotImplementedError, match="kyutai/mimi"):
        QWEN3OMNI_HF_LAYOUT.modules[CODEC].build_config(Qwen3OmniMoeConfig())


def test_the_thinker_states_flag_is_the_graphs_call_not_the_checkpoints():
    """Whether the thinker keeps every layer's hidden states depends on the graph
    being run, so the layout must leave it off and let ``modules_*.yaml`` opt in."""
    from transformers import Qwen3OmniMoeConfig

    config = QWEN3OMNI_HF_LAYOUT.modules[LLM].build_config(Qwen3OmniMoeConfig())
    assert config.collect_thinker_states is False
