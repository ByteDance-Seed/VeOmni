"""Config for :class:`Qwen3Llm`."""

from typing import Any, Dict, Optional

from transformers import PretrainedConfig, Qwen3Config


class Qwen3LlmConfig(PretrainedConfig):
    """Top-level config for the Qwen3 AR backbone (no wte, no lm_head)."""

    model_type = "qwen3_llm"

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        freeze: bool = False,
        **kwargs,
    ):
        self.text_config = Qwen3Config(**text_config) if text_config else Qwen3Config()
        # When True, ``Qwen3Llm.freeze_model`` freezes the whole backbone
        # (used to bootstrap a frozen LLM into a multimodal model).
        self.freeze = freeze
        super().__init__(**kwargs)
