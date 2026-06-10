"""Config for :class:`JanusLlama`.

Mirrors HuggingFace :class:`transformers.LlamaConfig` (dropping nothing —
the backbone re-uses every Llama hyper-parameter), wrapped in
``text_config`` so the checkpoint payload matches the original
``JanusForConditionalGeneration`` schema and ``scripts/convert_model.py``
can copy it verbatim.

Adds two Janus-specific token IDs at the top level:

* ``image_token_id``      — placeholder for understanding image patches
                             (runtime, via tokenizer; same field as
                             :class:`transformers.JanusConfig.image_token_id`).
* ``gen_image_token_id``  — placeholder for generation image patches
                             (runtime, via tokenizer; kept separate so a
                             derived model with distinct slots can override).

The ``model_type`` literal here is the lookup key for
``OMNI_CONFIG_REGISTRY`` / ``OMNI_MODEL_REGISTRY``.
"""

from typing import Any, Dict, Optional

from transformers import LlamaConfig, PretrainedConfig


class JanusLlamaConfig(PretrainedConfig):
    """Top-level config for the Janus LLaMA backbone (no wte, no lm_head)."""

    model_type = "janus_llama"

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.text_config = LlamaConfig(**text_config) if text_config else LlamaConfig()
        super().__init__(**kwargs)
