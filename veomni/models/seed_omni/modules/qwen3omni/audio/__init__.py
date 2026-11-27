"""Qwen3-Omni audio tower OmniModule."""

from ... import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
)


@OMNI_CONFIG_REGISTRY.register("qwen3omni_audio")
def register_qwen3omni_audio_config():
    from .configuration import Qwen3OmniAudioEncoderConfig

    return Qwen3OmniAudioEncoderConfig


@OMNI_MODEL_REGISTRY.register("qwen3omni_audio")
def register_qwen3omni_audio_model():
    from .modeling import Qwen3OmniAudioEncoder

    return Qwen3OmniAudioEncoder


@OMNI_ACCELERATED_MODEL_REGISTRY.register("qwen3omni_audio")
def register_qwen3omni_audio_accelerated_model():
    from .accelerated import Qwen3OmniAudioEncoderAccelerated

    return Qwen3OmniAudioEncoderAccelerated


@OMNI_PROCESSOR_REGISTRY.register("qwen3omni_audio")
def register_qwen3omni_audio_processor():
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor
