"""Config for :class:`Qwen3Llm`."""

from typing import Any, Dict, Optional

from transformers import Qwen3Config

from ...module_configuration_base import OmniModuleConfig


class Qwen3LlmConfig(OmniModuleConfig):
    """Top-level config for the Qwen3 AR backbone (no wte, no lm_head)."""

    model_type = "qwen3_llm"
    sub_configs = {"text_config": Qwen3Config}

    def __init__(
        self,
        text_config: Optional[Dict[str, Any] | Qwen3Config] = None,
        freeze: bool = False,
        **kwargs,
    ):
        if isinstance(text_config, Qwen3Config):
            self.text_config = text_config
        else:
            self.text_config = self.sub_configs["text_config"](**(text_config or {}))
        # When True, ``Qwen3Llm.freeze_model`` freezes the whole backbone
        # (used to bootstrap a frozen LLM into a multimodal model).
        self.freeze = freeze
        super().__init__(**kwargs)
