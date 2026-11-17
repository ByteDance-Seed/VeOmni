"""Config for :class:`Qwen3MoeLlm`."""

from typing import Any, Dict, Optional

from transformers import Qwen3MoeConfig

from ...module_configuration_base import OmniModuleConfig


class Qwen3MoeLlmConfig(OmniModuleConfig):
    """Top-level config for the Qwen3-MoE AR backbone (no wte, no lm_head)."""

    model_type = "qwen3_moe_llm"
    # Register the nested backbone config so transformers propagates
    # ``attn_implementation`` (e.g. veomni_flash_attention_2) down to it —
    # otherwise the inner ``Qwen3MoeModel`` silently falls back to SDPA and breaks
    # under sequence parallelism (sliced q vs full-length mask).
    sub_configs = {"text_config": Qwen3MoeConfig}

    def __init__(
        self,
        text_config: Optional[Dict[str, Any] | Qwen3MoeConfig] = None,
        freeze: bool = False,
        **kwargs,
    ):
        # Accept an already-instantiated inner config, as the dense Qwen3 sibling
        # does: ``from_dict`` / a caller holding a ``Qwen3MoeConfig`` would
        # otherwise hit ``TypeError: argument after ** must be a mapping``. The
        # object is reused as given — nothing below changes it.
        if isinstance(text_config, Qwen3MoeConfig):
            self.text_config = text_config
        else:
            self.text_config = Qwen3MoeConfig(**(text_config or {}))
        # When True, ``Qwen3MoeLlm.freeze_model`` freezes the whole backbone
        # (used to bootstrap a frozen LLM into a multimodal model).
        self.freeze = freeze
        super().__init__(**kwargs)
