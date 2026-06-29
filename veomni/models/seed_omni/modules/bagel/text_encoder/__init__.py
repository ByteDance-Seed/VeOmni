"""BAGEL token embedding and LM head module."""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("bagel_text_encoder")
def register_bagel_text_encoder_config():
    from .configuration import BagelTextEncoderConfig

    return BagelTextEncoderConfig


@OMNI_MODEL_REGISTRY.register("bagel_text_encoder")
def register_bagel_text_encoder_model():
    from .modeling import BagelTextEncoder

    return BagelTextEncoder
