"""Qwen3-Omni audio codec OmniModule — waveform <-> RVQ codes."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("qwen3omni_codec")
def register_qwen3omni_codec_config():
    from .configuration import Qwen3OmniCodecConfig

    return Qwen3OmniCodecConfig


@OMNI_MODEL_REGISTRY.register("qwen3omni_codec")
def register_qwen3omni_codec_model():
    from .modeling import Qwen3OmniCodec

    return Qwen3OmniCodec


@OMNI_ACCELERATED_MODEL_REGISTRY.register("qwen3omni_codec")
def register_qwen3omni_codec_accelerated_model():
    from .accelerated import Qwen3OmniCodecAccelerated

    return Qwen3OmniCodecAccelerated
