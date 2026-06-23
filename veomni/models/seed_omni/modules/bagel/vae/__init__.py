"""BAGEL latent autoencoder module."""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_PROCESSOR_REGISTRY


@OMNI_CONFIG_REGISTRY.register("bagel_vae")
def register_bagel_vae_config():
    from .configuration import BagelVAEConfig

    return BagelVAEConfig


@OMNI_MODEL_REGISTRY.register("bagel_vae")
def register_bagel_vae_model():
    from .modeling import BagelVAE

    return BagelVAE


@OMNI_PROCESSOR_REGISTRY.register("bagel_vae")
def register_bagel_vae_processor():
    from .processing import BagelVAEProcessor

    return BagelVAEProcessor
