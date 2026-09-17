"""Lazy registration for SeedVR2's independently implemented NaDiT and VAE."""

from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("seedvr2")
def register_seedvr2_config():
    from .configuration_seedvr2 import SeedVR2Config

    return SeedVR2Config


@MODELING_REGISTRY.register("seedvr2")
def register_seedvr2_model(architecture=None):
    from .modeling_seedvr2 import SeedVR2Model

    return SeedVR2Model


@MODEL_CONFIG_REGISTRY.register("seedvr2_condition")
def register_seedvr2_condition_config():
    from .conditioning_seedvr2 import SeedVR2ConditionConfig

    return SeedVR2ConditionConfig


@MODELING_REGISTRY.register("seedvr2_condition")
def register_seedvr2_condition_model(architecture=None):
    from .conditioning_seedvr2 import SeedVR2ConditionModel

    return SeedVR2ConditionModel
