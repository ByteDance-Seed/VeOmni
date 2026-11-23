"""Config for :class:`Qwen3OmniLlm`."""

from typing import Any, Dict, Optional

from transformers import Qwen3OmniMoeTextConfig

from ...module_configuration_base import OmniModuleConfig


class Qwen3OmniLlmConfig(OmniModuleConfig):
    """Top-level config for the Qwen3-Omni thinker backbone (no wte, no lm_head).

    ``spatial_merge_size`` / ``position_id_per_seconds`` are copied from the
    thinker config so the backbone can rebuild TMRoPE position ids from each
    media item's own metadata, without holding a reference to the vision or
    audio module.
    """

    model_type = "qwen3omni_llm"
    # Register the nested backbone config so transformers propagates
    # ``attn_implementation`` (e.g. veomni_flash_attention_2_with_sp) down to it —
    # otherwise the inner text model silently falls back to SDPA and breaks under
    # sequence parallelism (sliced q vs full-length mask).
    sub_configs = {"text_config": Qwen3OmniMoeTextConfig}

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        spatial_merge_size: int = 2,
        position_id_per_seconds: int = 13,
        accept_hidden_layer: int = 24,
        collect_thinker_states: bool = False,
        freeze: bool = False,
        **kwargs,
    ):
        self.text_config = Qwen3OmniMoeTextConfig(**text_config) if text_config else Qwen3OmniMoeTextConfig()
        self.spatial_merge_size = spatial_merge_size
        self.position_id_per_seconds = position_id_per_seconds
        # Which decoder layer the talker consumes. The talker does NOT read the
        # final hidden state (the lm_head does) — it reads this intermediate
        # layer, so the backbone has to expose it during generation. Copied from
        # ``talker_config.accept_hidden_layer`` by the checkpoint split.
        self.accept_hidden_layer = accept_hidden_layer
        # Collecting the thinker traces means asking for all layers' hidden
        # states, so only the graphs whose talker consumes them turn it on: the
        # audio-output generation graph, and talker training. Text-only
        # inference and thinker-only training pay nothing. The two paths differ
        # in where the traces go — generation accumulates them per step and
        # hands them over through ``flush_thinker_states``, while training does
        # one pass and leaves each item's slice on the carrier.
        # Retaining every layer costs memory; only layers 0 and
        # ``accept_hidden_layer`` are read, so a forward hook on that single
        # layer is the escape hatch if it ever matters.
        self.collect_thinker_states = collect_thinker_states
        # Talker training freezes the thinker and trains only the speech head.
        # Note this is orthogonal to ``collect_thinker_states``: that round needs
        # the traces *and* no gradient through them, so it sets both.
        self.freeze = freeze
        # This backbone owns no wte / lm_head, so there is nothing to tie — the
        # real ``embed_tokens``<->head tie lives in ``qwen3omni_text_encoder``.
        # Leaving the flag True would drive the shared post-load tie
        # (``module_utils.post_process_after_weight_loading``) into this module's
        # ``nn.Identity()`` input embedding and crash with ``KeyError: 'weight'``.
        # Force it off on both the outer + nested config (the check ANDs both sides).
        kwargs["tie_word_embeddings"] = False
        self.text_config.tie_word_embeddings = False
        super().__init__(**kwargs)
