"""Config for :class:`Qwen3OmniAudioEncoder`."""

from typing import Any, Dict, Optional

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig

from ...module_configuration_base import OmniModuleConfig


class Qwen3OmniAudioEncoderConfig(OmniModuleConfig):
    """Wraps the upstream audio-tower config as a standalone OmniModule config.

    ``audio_token_id`` rides along so the sibling text encoder and this module
    agree on the placeholder id without either holding a reference to the other.
    """

    model_type = "qwen3omni_audio"
    # Registered so transformers propagates ``attn_implementation`` down to the
    # encoder layers, same reason as qwen3vl_llm's nested text_config.
    sub_configs = {"audio_config": Qwen3OmniMoeAudioEncoderConfig}

    def __init__(
        self,
        audio_config: Optional[Dict[str, Any]] = None,
        audio_token_id: int = 151646,
        freeze: bool = False,
        **kwargs,
    ):
        self.audio_config = (
            Qwen3OmniMoeAudioEncoderConfig(**audio_config) if audio_config else Qwen3OmniMoeAudioEncoderConfig()
        )
        self.audio_token_id = audio_token_id
        # Talker training freezes the whole thinker, this tower included: the
        # traces the talker reads are detached at source, so nothing would reach
        # these weights anyway and an optimizer over them is pure overhead.
        self.freeze = freeze
        # No wte / lm_head here, so there is nothing to tie. Left True, the
        # shared post-load tie would look for an input embedding this module
        # does not have. Same guard as Qwen3VLLlmConfig.
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
