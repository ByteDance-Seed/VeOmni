"""Qwen3-specific :class:`TextEncoder` with ChatML templating."""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("qwen3_text_encoder")
def register_qwen3_text_encoder_config():
    from .configuration import Qwen3TextEncoderConfig

    return Qwen3TextEncoderConfig


@OMNI_MODEL_REGISTRY.register("qwen3_text_encoder")
def register_qwen3_text_encoder_model():
    from .modeling import Qwen3TextEncoder

    return Qwen3TextEncoder
