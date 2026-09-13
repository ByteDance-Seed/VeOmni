"""Cross-family OmniModule mixins.

Currently:
* :class:`TextEncoder` — generic word-token embedding + LM head, reusable
  across LLM families (model_type ``text_encoder``).
* :mod:`llm_packing` — shared ``conversation_list`` ↔ AR backbone carrier helpers:
  ``pack_llm_conversations_for_forward`` (1-D position backbones) and
  ``scatter_llm_hidden_states`` (all AR families including Qwen3-VL).

These modules don't depend on any specific HuggingFace family — they can be
mixed into any backbone setup that needs the conventional vocab-bound
encode/decode pair without re-deriving from a heavy HF base class.

Layout
------
Each sub-module lives in its own folder named after the module
(``text_encoder/``).  The folder contains short-named files —
``configuration.py``, ``modeling.py`` — instead of repeating the module
name in every filename.  See :mod:`veomni.models.seed_omni.modules`
for the registry that wires them up.
"""

from . import text_encoder  # noqa: F401
