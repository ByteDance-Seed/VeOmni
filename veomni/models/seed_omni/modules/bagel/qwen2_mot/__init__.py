"""BAGEL Qwen2 MoT backbone module."""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("bagel_qwen2_mot")
def register_bagel_qwen2_mot_config():
    from .configuration import BagelQwen2MoTConfig

    return BagelQwen2MoTConfig


@OMNI_MODEL_REGISTRY.register("bagel_qwen2_mot")
def register_bagel_qwen2_mot_model():
    from .modeling import BagelQwen2MoT

    return BagelQwen2MoT
