"""Janus SigLIP vision tower + aligner OmniModule.

Sub-package layout:

* :mod:`.configuration` — :class:`JanusSiglipConfig`
* :mod:`.modeling`      — :class:`JanusSiglip` (HF-native ``PreTrainedModel``)
* :mod:`.accelerated`   — :class:`JanusSiglipAccelerated` (VeOmni training/runtime)
* :mod:`.processing`    — :class:`JanusSiglipProcessor`
"""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_PROCESSOR_REGISTRY


@OMNI_CONFIG_REGISTRY.register("janus_siglip")
def register_janus_siglip_config():
    from .configuration import JanusSiglipConfig

    return JanusSiglipConfig


@OMNI_MODEL_REGISTRY.register("janus_siglip")
def register_janus_siglip_model():
    from .modeling import JanusSiglip

    return JanusSiglip


@OMNI_ACCELERATED_MODEL_REGISTRY.register("janus_siglip")
def register_janus_siglip_accelerated_model():
    from .accelerated import JanusSiglipAccelerated

    return JanusSiglipAccelerated


@OMNI_PROCESSOR_REGISTRY.register("janus_siglip")
def register_janus_siglip_processor():
    from .processing import JanusSiglipProcessor

    return JanusSiglipProcessor
