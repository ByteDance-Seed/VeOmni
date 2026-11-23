"""Qwen3-Omni-specific :class:`TextEncoder` with ChatML (text + image + video + audio) templating."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("qwen3omni_text_encoder")
def register_qwen3omni_text_encoder_config():
    from .configuration import Qwen3OmniTextEncoderConfig

    return Qwen3OmniTextEncoderConfig


@OMNI_MODEL_REGISTRY.register("qwen3omni_text_encoder")
def register_qwen3omni_text_encoder_model():
    from .modeling import Qwen3OmniTextEncoder

    return Qwen3OmniTextEncoder


@OMNI_ACCELERATED_MODEL_REGISTRY.register("qwen3omni_text_encoder")
def register_qwen3omni_text_encoder_accelerated_model():
    from .accelerated import Qwen3OmniTextEncoderAccelerated

    return Qwen3OmniTextEncoderAccelerated
