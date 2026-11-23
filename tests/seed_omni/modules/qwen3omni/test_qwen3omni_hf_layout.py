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

    # The audio module wraps the tower as ``self.audio_tower``, so the prefix is
    # kept: stripping it would produce ``conv2d1.weight``, which names no
    # parameter and would leave the tower unloaded — silently, since nothing
    # raises on a key that matches nothing.
    assert layout.route("thinker.audio_tower.conv2d1.weight") == (AUDIO, "audio_tower.conv2d1.weight")
    assert layout.route("thinker.visual.blocks.0.attn.qkv.weight") == (VISION, "visual.blocks.0.attn.qkv.weight")
    assert layout.route("talker.model.layers.0.self_attn.q_proj.weight") == (
        TALKER,
        "talker.model.layers.0.self_attn.q_proj.weight",
    )
    assert layout.route("code2wav.decoder.0.weight") == (CODEC, "code2wav.decoder.0.weight")


@pytest.mark.parametrize("module_name", [AUDIO, VISION, TEXT_ENCODER, LLM, TALKER, CODEC])
def test_every_layout_destination_names_a_real_parameter_prefix(module_name):
    """A destination that matches no parameter loads nothing and raises nothing.

    Checked against a really-built module rather than the string, because that is
    the only way to catch a prefix that looks plausible (``conv2d1.weight`` under
    an audio module) but is not where the weights live.
    """
    import json
    from pathlib import Path

    import torch
    from transformers import MimiConfig

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY

    toy = json.loads(
        (Path(__file__).parents[4] / "tests" / "toy_config" / "qwen3omni_toy" / "config.json").read_text()
    )
    thinker, talker = toy["thinker_config"], toy["talker_config"]
    ops = make_eager_ops_config()
    builds = {
        AUDIO: dict(audio_config=thinker["audio_config"], audio_token_id=thinker["audio_token_id"]),
        VISION: dict(vision_config=thinker["vision_config"]),
        TEXT_ENCODER: dict(
            vocab_size=thinker["text_config"]["vocab_size"],
            hidden_size=thinker["text_config"]["hidden_size"],
            tie_word_embeddings=False,
            lm_head_bias=False,
        ),
        LLM: dict(
            text_config=thinker["text_config"],
            spatial_merge_size=thinker["vision_config"]["spatial_merge_size"],
            position_id_per_seconds=thinker["position_id_per_seconds"],
        ),
        TALKER: dict(talker_config=talker),
        CODEC: dict(code2wav_config=toy["code2wav_config"], mimi_config=MimiConfig().to_dict()),
    }
    with torch.device("meta"):
        module = OMNI_MODEL_REGISTRY[module_name]()(
            OMNI_CONFIG_REGISTRY[module_name]()(ops_implementation=ops, **builds[module_name])
        )
    # ``_layout_parameter_names`` is the check itself: every destination prefix
    # must be a real prefix of some parameter name on this module.
    names = set(dict(module.named_parameters()))

    for source_prefix, destination_prefix in QWEN3OMNI_HF_LAYOUT.modules[module_name].key_prefixes:
        del source_prefix
        assert any(name.startswith(destination_prefix) or not destination_prefix for name in names), (
            f"{module_name}: destination prefix {destination_prefix!r} names no parameter on "
            f"{type(module).__name__}; its weights would never load"
        )


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
