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
        # This wrapper owns no vocabulary: ``get_input_embeddings()`` is the
        # ``nn.Identity`` installed over the real ``embed_tokens``, and
        # ``get_output_embeddings()`` is ``None`` — the tie lives in
        # ``qwen3_text_encoder``, which keeps whatever the upstream config says.
        #
        # The shared post-load tie (``checkpoint.weights.post_process_after_weight_loading``)
        # ANDs the outer and nested flags. Left True on both, it would try to
        # assign ``None._parameters["weight"]`` here and raise
        # ``RuntimeError: Failed to tie input/output embeddings``. Turning the
        # outer flag off is enough to short-circuit it, and it is the only side
        # touched: the nested config is the caller's, and ``text_config`` is what
        # the upstream model's own tie is derived from.
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
