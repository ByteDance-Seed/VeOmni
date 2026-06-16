"""Qwen3-MoE OmniModule mixins.

Splits a monolithic ``Qwen3MoeForCausalLM`` into composable sub-modules for the
SeedOmni V2 graph runtime.  The MoE backbone lives under ``qwen3_moe/llm/``; the
vocabulary modules (``embed_tokens`` + ``lm_head``) are MoE-agnostic and reuse
the dense ``qwen3/text_encoder`` module.
"""

from . import convert_model, llm  # noqa: F401
