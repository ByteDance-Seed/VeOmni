"""Cosmos3 VAE module: the frozen Wan2.2 video VAE, cacheable."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("cosmos3_vae")
def register_cosmos3_vae_config():
    from .configuration import Cosmos3VaeConfig

    return Cosmos3VaeConfig


@OMNI_MODEL_REGISTRY.register("cosmos3_vae")
def register_cosmos3_vae_model():
    from .modeling import Cosmos3Vae

    return Cosmos3Vae


@OMNI_ACCELERATED_MODEL_REGISTRY.register("cosmos3_vae")
def register_cosmos3_vae_accelerated_model():
    from .accelerated import Cosmos3VaeAccelerated

    return Cosmos3VaeAccelerated
