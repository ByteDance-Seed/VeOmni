"""Config for :class:`Qwen3MoeLlm`."""

from typing import Any, Dict, Optional

from transformers import PretrainedConfig, Qwen3MoeConfig


class Qwen3MoeLlmConfig(PretrainedConfig):
    """Top-level config for the Qwen3-MoE AR backbone (no wte, no lm_head)."""

    model_type = "qwen3_moe_llm"
    # Register the nested backbone config so transformers propagates
    # ``attn_implementation`` (e.g. veomni_flash_attention_2_with_sp) down to it —
    # otherwise the inner ``Qwen3MoeModel`` silently falls back to SDPA and breaks
    # under sequence parallelism (sliced q vs full-length mask).
    sub_configs = {"text_config": Qwen3MoeConfig}

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.text_config = Qwen3MoeConfig(**text_config) if text_config else Qwen3MoeConfig()
        super().__init__(**kwargs)
