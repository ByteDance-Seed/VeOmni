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
        # This backbone owns no wte / lm_head, so there is nothing to tie — the
        # real ``embed_tokens``<->head tie lives in ``qwen3_text_encoder``. The
        # source Qwen3 ``text_config`` defaults the flag True, which would drive
        # the shared post-load tie (``checkpoint.weights.post_process_after_weight_loading``)
        # into this module's ``nn.Identity()`` input embedding and crash with
        # ``KeyError: 'weight'``. Force it off on both the outer + nested config so
        # the tie is skipped (the check ANDs both sides).
        kwargs["tie_word_embeddings"] = False
        self.text_config.tie_word_embeddings = False
        super().__init__(**kwargs)
