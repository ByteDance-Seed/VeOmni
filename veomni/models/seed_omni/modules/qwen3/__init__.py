"""Qwen3 OmniModule mixins.

Splits a monolithic ``Qwen3ForCausalLM`` into composable sub-modules for the
SeedOmni graph runtime.  Each sub-module lives under
``qwen3/<sub_module>/`` with short-named inner files.
"""

from . import hf_layout, llm, text_encoder  # noqa: F401
