"""Qwen3-Omni talker OmniModule — thinker states to Mimi RVQ codes."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("qwen3omni_talker")
def register_qwen3omni_talker_config():
    from .configuration import Qwen3OmniTalkerConfig

    return Qwen3OmniTalkerConfig


@OMNI_MODEL_REGISTRY.register("qwen3omni_talker")
def register_qwen3omni_talker_model():
    from .modeling import Qwen3OmniTalker

    return Qwen3OmniTalker


@OMNI_ACCELERATED_MODEL_REGISTRY.register("qwen3omni_talker")
def register_qwen3omni_talker_accelerated_model():
    from .accelerated import Qwen3OmniTalkerAccelerated

    return Qwen3OmniTalkerAccelerated
