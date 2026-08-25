"""Qwen3-VL OmniModule mixins.

Splits a monolithic ``Qwen3VLForConditionalGeneration`` into composable
sub-modules for the SeedOmni V2 graph runtime:

* ``qwen3vl_vision``       — ViT + patch merger + DeepStack mergers.
* ``qwen3vl_text_encoder`` — embed_tokens (+ lm_head) + tokenizer + chat template.
* ``qwen3vl_llm``          — text backbone (M-RoPE + DeepStack injection).
"""

from . import convert_model, llm, text_encoder, vision  # noqa: F401
