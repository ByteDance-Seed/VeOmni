"""Config for :class:`Qwen3Llm`."""

from typing import Any, Dict, Optional

from transformers import PretrainedConfig, Qwen3Config


class Qwen3LlmConfig(PretrainedConfig):
    """Top-level config for the Qwen3 AR backbone (no wte, no lm_head)."""

    model_type = "qwen3_llm"
    # Register the nested backbone config so transformers propagates
    # ``attn_implementation`` (e.g. veomni_flash_attention_2_with_sp) down to it —
    # otherwise the inner ``Qwen3Model`` silently falls back to SDPA and breaks
    # under sequence parallelism (sliced q vs full-length mask).
    sub_configs = {"text_config": Qwen3Config}

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
        # This backbone owns no wte / lm_head, so there is nothing to tie — the
        # real ``embed_tokens``<->head tie lives in ``qwen3_text_encoder``. The
        # source Qwen3 ``text_config`` defaults the flag True, which would drive
        # the shared post-load tie (``module_utils.post_process_after_weight_loading``)
        # into this module's ``nn.Identity()`` input embedding and crash with
        # ``KeyError: 'weight'``. Force it off on both the outer + nested config so
        # the tie is skipped (the check ANDs both sides).
        kwargs["tie_word_embeddings"] = False
        self.text_config.tie_word_embeddings = False
        super().__init__(**kwargs)
