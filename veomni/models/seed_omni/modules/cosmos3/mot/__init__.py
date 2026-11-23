"""Cosmos3 MoT module: both towers' decoder blocks behind one joint attention."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("cosmos3_mot")
def register_cosmos3_mot_config():
    from .configuration import Cosmos3MotConfig

    return Cosmos3MotConfig


@OMNI_MODEL_REGISTRY.register("cosmos3_mot")
def register_cosmos3_mot_model():
    from .modeling import Cosmos3Mot

    return Cosmos3Mot


@OMNI_ACCELERATED_MODEL_REGISTRY.register("cosmos3_mot")
def register_cosmos3_mot_accelerated_model():
    from .accelerated import Cosmos3MotAccelerated

    return Cosmos3MotAccelerated
