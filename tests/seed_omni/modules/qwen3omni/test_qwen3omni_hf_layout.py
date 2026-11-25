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


def test_every_bucket_key_matches_the_module_it_is_loaded_into():
    """A route that renames is only half the contract; the load target must agree.

    ``QWEN3OMNI_HF_LAYOUT.route()`` yields *wrapper-relative* keys
    (``audio_tower.conv2d1.weight``), so loading the inner module
    (``audio.audio_tower``) expects ``conv2d1.weight`` and rejects every key. That
    mismatch fails at the first strict load of an offline conversion — which is
    why the converter now loads the wrappers, and why this test exists: the
    routing test alone cannot see it.
    """
    import json
    from pathlib import Path

    import torch

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY

    toy = json.loads(
        (Path(__file__).parents[4] / "tests" / "toy_config" / "qwen3omni_toy" / "config.json").read_text()
    )
    thinker, talker = toy["thinker_config"], toy["talker_config"]
    ops = make_eager_ops_config()

    with torch.device("meta"):
        audio = OMNI_MODEL_REGISTRY["qwen3omni_audio"]()(
            OMNI_CONFIG_REGISTRY["qwen3omni_audio"]()(
                ops_implementation=ops,
                audio_config=thinker["audio_config"],
                audio_token_id=thinker["audio_token_id"],
            )
        )
        vision = OMNI_MODEL_REGISTRY["qwen3omni_vision"]()(
            OMNI_CONFIG_REGISTRY["qwen3omni_vision"]()(ops_implementation=ops, vision_config=thinker["vision_config"])
        )
        talker_module = OMNI_MODEL_REGISTRY["qwen3omni_talker"]()(
            OMNI_CONFIG_REGISTRY["qwen3omni_talker"]()(ops_implementation=ops, talker_config=talker)
        )
        llm = OMNI_MODEL_REGISTRY["qwen3omni_llm"]()(
            OMNI_CONFIG_REGISTRY["qwen3omni_llm"]()(
                ops_implementation=ops,
                text_config=thinker["text_config"],
                spatial_merge_size=2,
                position_id_per_seconds=13,
                accept_hidden_layer=0,
            )
        )

    cases = [
        (audio, "thinker.audio_tower.conv2d1.weight"),
        (vision, "thinker.visual.blocks.0.attn.qkv.weight"),
        (talker_module, "talker.model.norm.weight"),
    ]
    for module, source_key in cases:
        routed = QWEN3OMNI_HF_LAYOUT.route(source_key)
        assert routed is not None
        _, module_key = routed
        params = set(dict(module.named_parameters()))
        assert module_key in params, (
            f"{type(module).__name__} has no parameter {module_key!r}; "
            f"the converter must load this bucket into the wrapper, not an inner module"
        )

    # Every routed key must be loadable into the object the converter loads it
    # into. Asserted by looking the key up on that object rather than by matching
    # the call site's text: a source check only proves spelling, and it missed the
    # codec because that bucket was not in the list above.
    import torch as _torch
    from transformers import MimiConfig

    from veomni.models.seed_omni.modules.qwen3omni.codec.modeling import Qwen3OmniCodec

    with _torch.device("meta"):
        codec = OMNI_MODEL_REGISTRY["qwen3omni_codec"]()(
            OMNI_CONFIG_REGISTRY["qwen3omni_codec"]()(
                ops_implementation=ops,
                code2wav_config=toy["code2wav_config"],
                mimi_config=MimiConfig().to_dict(),
            )
        )
        text_encoder = OMNI_MODEL_REGISTRY["qwen3omni_text_encoder"]()(
            OMNI_CONFIG_REGISTRY["qwen3omni_text_encoder"]()(
                ops_implementation=ops,
                vocab_size=thinker["text_config"]["vocab_size"],
                hidden_size=thinker["text_config"]["hidden_size"],
                tie_word_embeddings=False,
                lm_head_bias=False,
            )
        )

    # (load target, a source key that belongs to it). The target is the object the
    # converter calls `load_state_dict` on — all six modules, codec included.
    destinations = [
        (audio, "thinker.audio_tower.conv2d1.weight"),
        (vision, "thinker.visual.blocks.0.attn.qkv.weight"),
        (text_encoder, "thinker.model.embed_tokens.weight"),
        (llm.language_model, "thinker.model.norm.weight"),
        (talker_module, "talker.model.norm.weight"),
        (codec, "code2wav.decoder.0.conv.weight"),
    ]
    for target, source_key in destinations:
        routed = QWEN3OMNI_HF_LAYOUT.route(source_key)
        assert routed is not None, source_key
        _, module_key = routed
        # The backbone is the only one whose bucket keeps a prefix its target does
        # not use; the converter strips it, so mirror that rather than special-case
        # the assertion.
        if module_key.startswith("language_model."):
            module_key = module_key.removeprefix("language_model.")
        assert module_key in set(dict(target.named_parameters())), (
            f"{type(target).__name__} has no parameter {module_key!r} — the routed key for "
            f"{source_key!r} cannot be loaded into this object"
        )

    # The codec's vocoder half is reachable from the module the converter loads:
    # `codec.load_state_dict(buckets["code2wav"])` works, while
    # `codec.code2wav.load_state_dict(...)` would reject every `code2wav.` key.
    # Asserted by *attempting the load* on both targets, which is what makes this
    # a behavioural check: with `strict=True` the wrong target raises.
    assert isinstance(codec, Qwen3OmniCodec)
    # Real shape, so the *only* variable is which module the key is loaded into.
    param_name = QWEN3OMNI_HF_LAYOUT.route("code2wav.decoder.0.conv.weight")[1]
    shape = dict(codec.named_parameters())[param_name].shape
    routed_codec = {param_name: torch.zeros(shape, device="meta")}
    # Right target: the wrapper accepts the prefixed key.
    codec.load_state_dict(routed_codec, assign=True, strict=False)
    # Wrong target: the inner vocoder has no such key, so a strict load rejects it
    # — which is exactly how the offline conversion failed.
    with pytest.raises(RuntimeError):
        codec.code2wav.load_state_dict(routed_codec, assign=True, strict=True)


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


def test_the_codec_defers_its_second_checkpoint_to_build_time():
    """Half a codec is worse than none — but the refusal belongs at build.

    ``resolve_omni_checkpoint_root`` materialises every module's config before
    any module is selected, so raising here would block a thinker-only run that
    never touches the codec. The config therefore builds (recording that its
    encoder half did not come from this checkpoint), and ``Qwen3OmniCodec``
    refuses in ``__init__`` — which only runs when the module is actually built.
    """
    from transformers import Qwen3OmniMoeConfig

    from tests.tools.training_utils import make_eager_ops_config

    config = QWEN3OMNI_HF_LAYOUT.modules[CODEC].build_config(Qwen3OmniMoeConfig())
    config.ops_implementation = make_eager_ops_config().to_dict()
    assert config.encoder_from_checkpoint is False

    from veomni.models.seed_omni.modules.qwen3omni.codec.modeling import Qwen3OmniCodec

    with pytest.raises(NotImplementedError, match="kyutai/mimi"):
        Qwen3OmniCodec(config)

    # The offline converter names the Mimi config, so it does not trip the guard.
    from transformers import MimiConfig

    from veomni.models.seed_omni.modules.qwen3omni.codec.configuration import Qwen3OmniCodecConfig

    assert Qwen3OmniCodecConfig(mimi_config=MimiConfig().to_dict()).encoder_from_checkpoint is True


def test_the_thinker_states_flag_is_the_graphs_call_not_the_checkpoints():
    """Whether the thinker keeps every layer's hidden states depends on the graph
    being run, so the layout must leave it off and let ``modules_*.yaml`` opt in."""
    from transformers import Qwen3OmniMoeConfig

    config = QWEN3OMNI_HF_LAYOUT.modules[LLM].build_config(Qwen3OmniMoeConfig())
    assert config.collect_thinker_states is False
