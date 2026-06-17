"""Qwen3-VL-specific :class:`TextEncoder` with ChatML (text + image) templating."""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("qwen3vl_text_encoder")
def register_qwen3vl_text_encoder_config():
    from .configuration import Qwen3VLTextEncoderConfig

    return Qwen3VLTextEncoderConfig


@OMNI_MODEL_REGISTRY.register("qwen3vl_text_encoder")
def register_qwen3vl_text_encoder_model():
    from .modeling import Qwen3VLTextEncoder

    return Qwen3VLTextEncoder
