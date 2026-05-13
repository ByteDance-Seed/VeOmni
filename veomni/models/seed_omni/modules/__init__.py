"""OmniModule implementations for various model families."""

from .janus import JanusLLM, JanusVisionEncoder, JanusVQDecoder


# ── Module registry ────────────────────────────────────────────────────────────
# Maps model_type strings (from OmniConfig.modules.<name>.model_type) to the
# OmniModule subclass that implements them.  Add new model families here.
MODULE_REGISTRY = {
    "janus_vision_encoder": JanusVisionEncoder,
    "janus_vq_decoder": JanusVQDecoder,
    "janus_llm": JanusLLM,
}

__all__ = ["JanusVisionEncoder", "JanusVQDecoder", "JanusLLM", "MODULE_REGISTRY"]
