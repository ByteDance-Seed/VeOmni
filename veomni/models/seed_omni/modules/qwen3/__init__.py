"""Qwen3 OmniModule mixins.

Splits a monolithic ``Qwen3ForCausalLM`` into composable sub-modules for the
SeedOmni V2 graph runtime.  Each sub-module lives under
``qwen3/<sub_module>/`` with short-named inner files.
"""

from . import convert_model, llm, text_encoder  # noqa: F401
