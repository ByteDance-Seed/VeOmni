"""Qwen3-Omni thinker vision tower OmniModule."""

from ... import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
)


@OMNI_CONFIG_REGISTRY.register("qwen3omni_vision")
def register_qwen3omni_vision_config():
    from .configuration import Qwen3OmniVisionEncoderConfig

    return Qwen3OmniVisionEncoderConfig


@OMNI_MODEL_REGISTRY.register("qwen3omni_vision")
def register_qwen3omni_vision_model():
    from .modeling import Qwen3OmniVisionEncoder

    return Qwen3OmniVisionEncoder


@OMNI_ACCELERATED_MODEL_REGISTRY.register("qwen3omni_vision")
def register_qwen3omni_vision_accelerated_model():
    from .accelerated import Qwen3OmniVisionEncoderAccelerated

    return Qwen3OmniVisionEncoderAccelerated


@OMNI_PROCESSOR_REGISTRY.register("qwen3omni_vision")
def register_qwen3omni_vision_processor():
    from .processing import Qwen3OmniVisionImageProcessor

    return Qwen3OmniVisionImageProcessor
