"""BAGEL SigLIP NaViT vision encoder module."""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_PROCESSOR_REGISTRY


@OMNI_CONFIG_REGISTRY.register("bagel_siglip_navit")
def register_bagel_siglip_navit_config():
    from .configuration import BagelSiglipNavitConfig

    return BagelSiglipNavitConfig


@OMNI_MODEL_REGISTRY.register("bagel_siglip_navit")
def register_bagel_siglip_navit_model():
    from .modeling import BagelSiglipNavit

    return BagelSiglipNavit


@OMNI_PROCESSOR_REGISTRY.register("bagel_siglip_navit")
def register_bagel_siglip_navit_processor():
    from .processing import BagelSiglipNavitProcessor

    return BagelSiglipNavitProcessor
