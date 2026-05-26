"""Generic word-token embedding + LM head module.

Sub-package layout (per :mod:`veomni.models.seed_omni.modules`):

* :mod:`.configuration` — :class:`TextEncoderConfig`
* :mod:`.modeling`      — :class:`TextEncoder`
"""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("text_encoder")
def register_text_encoder_config():
    from .configuration import TextEncoderConfig

    return TextEncoderConfig


@OMNI_MODEL_REGISTRY.register("text_encoder")
def register_text_encoder_model():
    from .modeling import TextEncoder

    return TextEncoder
