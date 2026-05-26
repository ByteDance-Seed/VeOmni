"""Janus SigLIP vision tower + aligner OmniModule.

Sub-package layout (per :mod:`veomni.models.seed_omni.modules`):

* :mod:`.configuration` — :class:`JanusSiglipConfig`
* :mod:`.modeling`      — :class:`JanusSiglip`
* :mod:`.processing`    — :class:`JanusSiglipProcessor`
"""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_PROCESSOR_REGISTRY


@OMNI_CONFIG_REGISTRY.register("janus_siglip")
def register_janus_siglip_config():
    from .configuration import JanusSiglipConfig

    return JanusSiglipConfig


@OMNI_MODEL_REGISTRY.register("janus_siglip")
def register_janus_siglip_model():
    from .modeling import JanusSiglip

    return JanusSiglip


@OMNI_PROCESSOR_REGISTRY.register("janus_siglip")
def register_janus_siglip_processor():
    from .processing import JanusSiglipProcessor

    return JanusSiglipProcessor
