"""Janus LLaMA backbone OmniModule.

Sub-package layout (per :mod:`veomni.models.seed_omni.modules`):

* :mod:`.configuration` — :class:`JanusLlamaConfig`
* :mod:`.modeling`      — :class:`JanusLlama`

(No per-module processor: the AR backbone consumes hidden tensors only.)
"""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("janus_llama")
def register_janus_llama_config():
    from .configuration import JanusLlamaConfig

    return JanusLlamaConfig


@OMNI_MODEL_REGISTRY.register("janus_llama")
def register_janus_llama_model():
    from .modeling import JanusLlama

    return JanusLlama
