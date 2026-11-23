"""Cosmos3 prompt encoder: the reasoner tower's word embedding."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("cosmos3_text_encoder")
def register_cosmos3_text_encoder_config():
    from .configuration import Cosmos3TextEncoderConfig

    return Cosmos3TextEncoderConfig


@OMNI_MODEL_REGISTRY.register("cosmos3_text_encoder")
def register_cosmos3_text_encoder_model():
    from .modeling import Cosmos3TextEncoder

    return Cosmos3TextEncoder


@OMNI_ACCELERATED_MODEL_REGISTRY.register("cosmos3_text_encoder")
def register_cosmos3_text_encoder_accelerated_model():
    from .accelerated import Cosmos3TextEncoderAccelerated

    return Cosmos3TextEncoderAccelerated
