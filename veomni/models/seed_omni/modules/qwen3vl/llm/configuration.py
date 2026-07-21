"""Config for :class:`Qwen3VLLlm`."""

from typing import Any, Dict, Optional

from transformers import PretrainedConfig, Qwen3VLTextConfig


class Qwen3VLLlmConfig(PretrainedConfig):
    """Top-level config for the Qwen3-VL AR backbone (no wte, no lm_head).

    ``spatial_merge_size`` is copied from the vision config so the backbone can
    rebuild multimodal RoPE (M-RoPE) position ids from each image item's
    ``grid_thw`` without holding a reference to the vision module.
    """

    model_type = "qwen3vl_llm"
    # Register the nested backbone config so transformers propagates
    # ``attn_implementation`` (e.g. veomni_flash_attention_2_with_sp) down to it —
    # otherwise the inner ``Qwen3VLTextModel`` silently falls back to SDPA and
    # breaks under sequence parallelism (sliced q vs full-length mask).
    sub_configs = {"text_config": Qwen3VLTextConfig}

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        spatial_merge_size: int = 2,
        image_token_id: int = 151655,
        **kwargs,
    ):
        self.text_config = Qwen3VLTextConfig(**text_config) if text_config else Qwen3VLTextConfig()
        self.spatial_merge_size = spatial_merge_size
        self.image_token_id = image_token_id
        # This backbone owns no wte / lm_head, so there is nothing to tie — the
        # real ``embed_tokens``<->head tie lives in ``qwen3vl_text_encoder``. The
        # source Qwen3-VL ``text_config`` defaults the flag True, which would drive
        # the shared post-load tie (``module_utils.post_process_after_weight_loading``)
        # into this module's ``nn.Identity()`` input embedding and crash with
        # ``KeyError: 'weight'``. Force it off on both the outer + nested config so
        # the tie is skipped (the check ANDs both sides).
        kwargs["tie_word_embeddings"] = False
        self.text_config.tie_word_embeddings = False
        super().__init__(**kwargs)
