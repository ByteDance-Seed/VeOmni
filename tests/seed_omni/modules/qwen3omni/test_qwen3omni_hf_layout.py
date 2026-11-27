"""Tests for ``OMNI_HF_LAYOUT_REGISTRY["qwen3_omni_moe"]``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import MimiConfig, Qwen3OmniMoeConfig

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.training_utils import make_eager_ops_config
from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY
from veomni.models.seed_omni.modules.qwen3omni.codec.configuration import Qwen3OmniCodecConfig
from veomni.models.seed_omni.modules.qwen3omni.codec.modeling import Qwen3OmniCodec
from veomni.models.seed_omni.modules.qwen3omni.hf_layout import QWEN3OMNI_HF_LAYOUT
from veomni.models.seed_omni.utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


AUDIO = "qwen3omni_audio"
VISION = "qwen3omni_vision"
TEXT_ENCODER = "qwen3omni_text_encoder"
LLM = "qwen3omni_llm"
TALKER = "qwen3omni_talker"
CODEC = "qwen3omni_codec"
TOY_CONFIG_PATH = Path(__file__).parents[4] / "tests" / "toy_config" / "qwen3omni_toy" / "config.json"


def _build_meta_modules() -> dict[str, torch.nn.Module]:
    toy = json.loads(TOY_CONFIG_PATH.read_text())
    thinker, talker = toy["thinker_config"], toy["talker_config"]
    ops = make_eager_ops_config()
    specs = {
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
            accept_hidden_layer=0,
        ),
        TALKER: dict(talker_config=talker),
        CODEC: dict(code2wav_config=toy["code2wav_config"], mimi_config=MimiConfig().to_dict()),
    }
    with torch.device("meta"):
        return {
            name: OMNI_MODEL_REGISTRY[name]()(OMNI_CONFIG_REGISTRY[name]()(ops_implementation=ops, **kwargs))
            for name, kwargs in specs.items()
        }


def test_qwen3omni_registers_a_layout_beside_its_converter():
    assert "qwen3_omni_moe" in set(OMNI_HF_LAYOUT_REGISTRY.valid_keys())
    assert set(QWEN3OMNI_HF_LAYOUT.modules) == {AUDIO, VISION, TEXT_ENCODER, LLM, TALKER, CODEC}
    assert QWEN3OMNI_HF_LAYOUT.infer_type == "audio_output"
    assert set(QWEN3OMNI_HF_LAYOUT.generation_graphs) == {"audio_output", "thinker_only"}
    assert set(QWEN3OMNI_HF_LAYOUT.training_graphs) == {"thinker", "talker", "joint"}


def test_layout_routes_each_key_to_the_longest_claiming_prefix_and_back():
    layout = QWEN3OMNI_HF_LAYOUT
    expected = {
        "thinker.model.embed_tokens.weight": (TEXT_ENCODER, "embed_tokens.weight"),
        "thinker.lm_head.weight": (TEXT_ENCODER, "lm_head.weight"),
        "thinker.model.layers.0.self_attn.q_proj.weight": (LLM, "language_model.layers.0.self_attn.q_proj.weight"),
        "thinker.model.norm.weight": (LLM, "language_model.norm.weight"),
        "thinker.audio_tower.conv2d1.weight": (AUDIO, "audio_tower.conv2d1.weight"),
        "thinker.visual.blocks.0.attn.qkv.weight": (VISION, "visual.blocks.0.attn.qkv.weight"),
        "talker.model.layers.0.self_attn.q_proj.weight": (TALKER, "talker.model.layers.0.self_attn.q_proj.weight"),
        "code2wav.decoder.0.weight": (CODEC, "code2wav.decoder.0.weight"),
    }
    for source_key, expected_route in expected.items():
        routed = layout.route(source_key)
        assert routed == expected_route
        assert layout.source_key(*routed) == source_key


def test_routed_keys_and_destination_prefixes_match_real_module_parameters():
    modules = _build_meta_modules()

    for module_name, module in modules.items():
        param_names = set(dict(module.named_parameters()))
        for _, destination_prefix in QWEN3OMNI_HF_LAYOUT.modules[module_name].key_prefixes:
            assert any(name.startswith(destination_prefix) or not destination_prefix for name in param_names), (
                f"{module_name}: destination prefix {destination_prefix!r} matches no parameter"
            )

    load_targets = [
        (modules[AUDIO], "thinker.audio_tower.conv2d1.weight"),
        (modules[VISION], "thinker.visual.blocks.0.attn.qkv.weight"),
        (modules[TEXT_ENCODER], "thinker.model.embed_tokens.weight"),
        (modules[LLM].language_model, "thinker.model.norm.weight"),
        (modules[TALKER], "talker.model.norm.weight"),
        (modules[CODEC], "code2wav.decoder.0.conv.weight"),
    ]
    for target, source_key in load_targets:
        _, module_key = QWEN3OMNI_HF_LAYOUT.route(source_key)
        if module_key.startswith("language_model."):
            module_key = module_key.removeprefix("language_model.")
        assert module_key in dict(target.named_parameters()), (source_key, module_key)

    # Codec routed keys (`code2wav.*`) match the wrapper, not the inner vocoder.
    codec = modules[CODEC]
    assert isinstance(codec, Qwen3OmniCodec)
    param_name = QWEN3OMNI_HF_LAYOUT.route("code2wav.decoder.0.conv.weight")[1]
    routed_codec = {param_name: torch.zeros(dict(codec.named_parameters())[param_name].shape, device="meta")}
    assert codec.load_state_dict(routed_codec, assign=True, strict=False).unexpected_keys == []
    assert codec.code2wav.load_state_dict(routed_codec, assign=True, strict=False).unexpected_keys == [param_name]


def test_layout_config_defaults_and_codec_mimi_guard():
    upstream = Qwen3OmniMoeConfig()

    # Codec defers its missing-Mimi refusal to build time so thinker-only runs can resolve the root.
    codec_cfg = QWEN3OMNI_HF_LAYOUT.modules[CODEC].build_config(upstream)
    codec_cfg.ops_implementation = make_eager_ops_config().to_dict()
    assert codec_cfg.encoder_from_checkpoint is False
    with pytest.raises(NotImplementedError, match="kyutai/mimi"):
        Qwen3OmniCodec(codec_cfg)
    assert Qwen3OmniCodecConfig(mimi_config=MimiConfig().to_dict()).encoder_from_checkpoint is True

    # Thinker state collection stays off by default and is opted into by graph configs.
    assert QWEN3OMNI_HF_LAYOUT.modules[LLM].build_config(upstream).collect_thinker_states is False
