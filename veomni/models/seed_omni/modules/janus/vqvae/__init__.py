"""Janus VQ-VAE OmniModule (encode + unified decode).

Sub-package layout (per :mod:`veomni.models.seed_omni.modules`):

* :mod:`.configuration` — :class:`JanusVqvaeConfig`
* :mod:`.modeling`      — :class:`JanusVqvae`
* :mod:`.processing`    — :class:`JanusVqvaeProcessor`
"""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_PROCESSOR_REGISTRY


@OMNI_CONFIG_REGISTRY.register("janus_vqvae")
def register_janus_vqvae_config():
    from .configuration import JanusVqvaeConfig

    return JanusVqvaeConfig


@OMNI_MODEL_REGISTRY.register("janus_vqvae")
def register_janus_vqvae_model():
    from .modeling import JanusVqvae

    return JanusVqvae


@OMNI_PROCESSOR_REGISTRY.register("janus_vqvae")
def register_janus_vqvae_processor():
    from .processing import JanusVqvaeProcessor

    return JanusVqvaeProcessor
