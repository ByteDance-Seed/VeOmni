"""Config for :class:`JanusLlama`.

Mirrors HuggingFace :class:`transformers.LlamaConfig` (dropping nothing —
the backbone re-uses every Llama hyper-parameter), wrapped in
``text_config`` so the checkpoint payload matches the original
``JanusForConditionalGeneration`` schema and ``scripts/split_janus.py``
can copy it verbatim.

Adds two Janus-specific token IDs at the top level:

* ``image_token_id``      — placeholder for understanding image patches
                             (HF Janus default ``100581``; same field as
                             :class:`transformers.JanusConfig.image_token_id`).
* ``gen_image_token_id``  — placeholder for generation image patches
                             (Janus uses the same id ``100581`` for both
                             understanding and generation positions; we
                             keep the field separate so a derived model
                             with distinct slots can override it).

The ``model_type`` literal here is the lookup key for both HF
``AutoConfig`` (after registration in :mod:`...modules.__init__`) and the
SeedOmni V2 ``MODULE_MIXIN_REGISTRY``.
"""

from typing import Any, Dict, Optional

from transformers import PretrainedConfig


class JanusLlamaConfig(PretrainedConfig):
    """Top-level config for the Janus LLaMA backbone (no wte, no lm_head)."""

    model_type = "janus_llama"

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        image_token_id: int = 100581,
        gen_image_token_id: int = 100581,
        **kwargs,
    ):
        self.text_config = text_config or {}
        self.image_token_id = image_token_id
        self.gen_image_token_id = gen_image_token_id
        super().__init__(**kwargs)
