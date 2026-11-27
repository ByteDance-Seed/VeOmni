"""How a Qwen3-Omni checkpoint splits into SeedOmni modules.

Registered under ``OMNI_HF_LAYOUT_REGISTRY["qwen3_omni_moe"]``. Point
``model.model_path`` at an upstream ``Qwen3OmniMoeForConditionalGeneration``
checkpoint to train / infer straight off it, or split it offline with
``scripts/seed_omni/convert_model.py`` (which goes through the family converter —
see :mod:`veomni.models.seed_omni.modules.qwen3omni.convert_model`, because this
family needs a second checkpoint for its codec). Graphs come from
``configs/seed_omni/Qwen/qwen3omni_30b_a3b/`` (``train/`` DAG, ``infer/`` FSMs).

Modules::

    qwen3omni_audio         # thinker.audio_tower.*            (+ mel feature extractor)
    qwen3omni_vision        # thinker.visual.*                 (+ image / video processors)
    qwen3omni_text_encoder  # thinker.model.embed_tokens + thinker.lm_head + tokenizer
    qwen3omni_llm           # the rest of thinker.model.*      (decoder layers, no wte/head)
    qwen3omni_talker        # talker.*                         (the speech head)
    qwen3omni_codec         # code2wav.*                       (vocoder; encoder half from Mimi)

The prefixes are the upstream ones, so every source tensor routes to exactly one
module. Two orderings are load-bearing and expressed by prefix length rather than
by list order, because a source key belongs to the module whose prefix is the
**longest** match: ``thinker.model.embed_tokens.`` beats the backbone's shorter
``thinker.model.``, and ``thinker.lm_head.`` is claimed by the text encoder.

``qwen3omni_codec`` is the one module this layout cannot complete on its own:
Qwen3-Omni released only the decoder half of its audio codec, so the encoder
comes from ``kyutai/mimi``. See the family converter, which is what an offline
split of this family runs.
"""

from __future__ import annotations

from typing import Any

from ...utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY, OmniHFLayout, OmniHFModuleLayout
from .audio.configuration import Qwen3OmniAudioEncoderConfig
from .codec.configuration import Qwen3OmniCodecConfig
from .llm.configuration import Qwen3OmniLlmConfig
from .talker.configuration import Qwen3OmniTalkerConfig
from .text_encoder.configuration import Qwen3OmniTextEncoderConfig
from .vision.configuration import Qwen3OmniVisionEncoderConfig


def _thinker_and_talker(hf_config: Any) -> tuple[Any, Any]:
    """The two nested configs upstream keeps, read defensively.

    A real ``Qwen3OmniMoeConfig`` answers both as attributes; a config rebuilt
    from a plain dict answers them the same way once transformers has parsed it.
    """
    thinker = getattr(hf_config, "thinker_config", None)
    talker = getattr(hf_config, "talker_config", None)
    if thinker is None or talker is None:
        raise ValueError(
            "qwen3omni: the upstream config carries no 'thinker_config' / 'talker_config'. This layout only "
            "describes a Qwen3OmniMoeForConditionalGeneration checkpoint."
        )
    return thinker, talker


def _audio_config(hf_config: Any) -> Qwen3OmniAudioEncoderConfig:
    thinker, _ = _thinker_and_talker(hf_config)
    return Qwen3OmniAudioEncoderConfig(
        audio_config=thinker.audio_config.to_dict(),
        audio_token_id=thinker.audio_token_id,
    )


def _vision_config(hf_config: Any) -> Qwen3OmniVisionEncoderConfig:
    thinker, _ = _thinker_and_talker(hf_config)
    return Qwen3OmniVisionEncoderConfig(vision_config=thinker.vision_config.to_dict())


def _text_encoder_config(hf_config: Any) -> Qwen3OmniTextEncoderConfig:
    thinker, _ = _thinker_and_talker(hf_config)
    text_config = thinker.text_config
    if bool(getattr(text_config, "tie_word_embeddings", False)):
        raise ValueError(
            "qwen3omni: this checkpoint reports tied embeddings, but Qwen3-Omni's thinker ships a separate "
            "lm_head. Refusing to guess which one to keep."
        )
    return Qwen3OmniTextEncoderConfig(
        vocab_size=text_config.vocab_size,
        hidden_size=text_config.hidden_size,
        tie_word_embeddings=False,
        lm_head_bias=False,
    )


def _llm_config(hf_config: Any) -> Qwen3OmniLlmConfig:
    thinker, talker = _thinker_and_talker(hf_config)
    # ``spatial_merge_size`` / ``position_id_per_seconds`` are copied so the
    # backbone can rebuild TMRoPE positions from each item's ``grid_thw`` without
    # holding a reference to the vision module.
    #
    # ``collect_thinker_states`` is deliberately left at its default (off): whether
    # the thinker has to keep every layer's hidden states is a property of the
    # graph being run — only a talker consumes them — not of the weights, so the
    # scenario's ``modules_*.yaml`` opts in.
    return Qwen3OmniLlmConfig(
        text_config=thinker.text_config.to_dict(),
        spatial_merge_size=thinker.vision_config.spatial_merge_size,
        position_id_per_seconds=thinker.position_id_per_seconds,
        # The talker reads an intermediate thinker layer rather than the last one;
        # the depth is the talker's to declare, so it is copied from there.
        accept_hidden_layer=talker.accept_hidden_layer,
    )


def _talker_config(hf_config: Any) -> Qwen3OmniTalkerConfig:
    _, talker = _thinker_and_talker(hf_config)
    return Qwen3OmniTalkerConfig(talker_config=talker.to_dict())


def _codec_config(hf_config: Any) -> Qwen3OmniCodecConfig:
    """The decoder half, plus a default Mimi config for the encoder half.

    Built rather than refused: ``resolve_omni_checkpoint_root`` materialises every
    registered module's config before any module is chosen, so raising here would
    block a thinker-only run that never selects the codec. The encoder's *weights*
    are what genuinely cannot come from this checkpoint, and
    ``Qwen3OmniCodec`` refuses at construction when its Mimi half is unloaded —
    the right place, because that fires only when the module is actually built.
    """
    return Qwen3OmniCodecConfig(
        code2wav_config=getattr(hf_config, "code2wav_config", None).to_dict()
        if getattr(hf_config, "code2wav_config", None) is not None
        else None
    )


def _tokenizer(model_path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    return {"tokenizer": AutoTokenizer.from_pretrained(model_path)}


def _media_processors(model_path: str) -> dict[str, Any]:
    """The two processor sidecars the vision module binds.

    Qwen3-Omni ships a **single** ``preprocessor_config.json`` serving three
    processors at once (``feature_extractor_type: WhisperFeatureExtractor`` for the
    audio tower, ``image_processor_type: Qwen2VLImageProcessor`` here) and no video
    config at all — the video processor resolves through the auto-map and reads
    that same file. Its class default ``patch_size`` is 14 while Qwen3-Omni uses
    16, so both must come from the checkpoint directory.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    return {"image_processor": processor.image_processor, "video_processor": processor.video_processor}


def _mel_extractor(model_path: str) -> dict[str, Any]:
    from transformers import AutoProcessor

    return {"processor": AutoProcessor.from_pretrained(model_path).feature_extractor}


QWEN3OMNI_HF_LAYOUT = OmniHFLayout(
    modules={
        "qwen3omni_audio": OmniHFModuleLayout(
            # The module wraps the tower as ``self.audio_tower`` (the converter
            # loads the same bucket into it), so the prefix is *kept* rather than
            # stripped: ``conv2d1.weight`` would name no parameter.
            key_prefixes=(("thinker.audio_tower.", "audio_tower."),),
            build_config=_audio_config,
            build_assets=_mel_extractor,
        ),
        "qwen3omni_vision": OmniHFModuleLayout(
            key_prefixes=(("thinker.visual.", "visual."),),
            build_config=_vision_config,
            build_assets=_media_processors,
        ),
        "qwen3omni_text_encoder": OmniHFModuleLayout(
            key_prefixes=(
                ("thinker.model.embed_tokens.", "embed_tokens."),
                ("thinker.lm_head.", "lm_head."),
            ),
            build_config=_text_encoder_config,
            build_assets=_tokenizer,
        ),
        "qwen3omni_llm": OmniHFModuleLayout(
            key_prefixes=(("thinker.model.", "language_model."),),
            build_config=_llm_config,
        ),
        "qwen3omni_talker": OmniHFModuleLayout(
            key_prefixes=(("talker.", "talker."),),
            build_config=_talker_config,
        ),
        "qwen3omni_codec": OmniHFModuleLayout(
            key_prefixes=(("code2wav.", "code2wav."),),
            build_config=_codec_config,
        ),
    },
    graph_dir="configs/seed_omni/Qwen/qwen3omni_30b_a3b",
    training_graphs={
        "thinker": "train/thinker/graph_train.yaml",
        "talker": "train/talker/graph_train.yaml",
        "joint": "train/joint/graph_train.yaml",
    },
    generation_graphs={
        "audio_output": "infer/audio_output/graph.yaml",
        "thinker_only": "infer/thinker_only/graph.yaml",
    },
    infer_type="audio_output",
)


@OMNI_HF_LAYOUT_REGISTRY.register("qwen3_omni_moe")
def _register_qwen3omni_hf_layout(source_path: str | None = None) -> OmniHFLayout:
    return QWEN3OMNI_HF_LAYOUT


__all__ = ["QWEN3OMNI_HF_LAYOUT"]
