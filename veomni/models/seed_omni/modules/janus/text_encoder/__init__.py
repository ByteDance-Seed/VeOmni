"""Janus-specific :class:`TextEncoder` with image boundary-token emitters.

Sub-package layout (per :mod:`veomni.models.seed_omni.modules`):

* :mod:`.configuration` — :class:`JanusTextEncoderConfig`
* :mod:`.modeling`      — :class:`JanusTextEncoder`
"""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("janus_text_encoder")
def register_janus_text_encoder_config():
    from .configuration import JanusTextEncoderConfig

    return JanusTextEncoderConfig


@OMNI_MODEL_REGISTRY.register("janus_text_encoder")
def register_janus_text_encoder_model():
    from .modeling import JanusTextEncoder

    return JanusTextEncoder
