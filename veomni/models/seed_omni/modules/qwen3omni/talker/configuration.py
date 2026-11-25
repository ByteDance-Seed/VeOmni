"""Config for :class:`Qwen3OmniTalker`."""

from typing import Any, Dict, Optional

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTalkerConfig

from ...module_configuration_base import OmniModuleConfig


class Qwen3OmniTalkerConfig(OmniModuleConfig):
    """Wraps the upstream talker config as a standalone OmniModule config.

    ``default_speaker`` selects an entry of ``talker_config.speaker_id`` when a
    request does not name one. The speech-span framing token ids are *not*
    duplicated here: they belong to the thinker vocabulary, so the text encoder
    resolves them from its tokenizer and passes them along with their
    embeddings (see ``Qwen3OmniTextEncoder.flush_talker_prompt``).
    """

    model_type = "qwen3omni_talker"
    # Registered so transformers propagates ``attn_implementation`` down to the
    # talker's decoder layers and its nested code predictor.
    sub_configs = {"talker_config": Qwen3OmniMoeTalkerConfig}

    def __init__(
        self,
        talker_config: Optional[Dict[str, Any]] = None,
        default_speaker: str = "ethan",
        max_new_tokens: int = 4096,
        semantic_loss_weight: float = 1.0,
        residual_loss_weight: float = 1.0,
        audio_n_window: int = 50,
        **kwargs,
    ):
        self.talker_config = Qwen3OmniMoeTalkerConfig(**talker_config) if talker_config else Qwen3OmniMoeTalkerConfig()
        # The *thinker's* mel window, which the talker's 3D RoPE needs and its own
        # config does not carry.
        #
        # `talker.get_rope_index` delegates to
        # `Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index`
        # passing ``self``, and that method reads
        # ``self.config.audio_config.n_window`` to expand an audio placeholder run
        # into TMRoPE coordinates. ``Qwen3OmniMoeTalkerConfig`` has no
        # ``audio_config``, so the call raised ``AttributeError`` as soon as a
        # prompt carried both a vision span and standalone audio. Declared here
        # (rather than grafted onto the HF config at call time) so the value is
        # part of this module's own schema and can be overridden per scenario.
        self.audio_n_window = audio_n_window
        self.default_speaker = default_speaker
        # Cap on codec frames per utterance (upstream's talker_max_new_tokens).
        # At 12.5 Hz, 4096 frames is ~5.5 minutes of speech.
        self.max_new_tokens = max_new_tokens
        # The two training heads. They score different element counts -- T+1
        # semantic rows against T*(G-1) residual codes -- so the residual term
        # dominates the sum at G=16 unless it is turned down here.
        self.semantic_loss_weight = semantic_loss_weight
        self.residual_loss_weight = residual_loss_weight
        # The talker owns an lm_head over the codec vocabulary but no input
        # embedding this module exposes at the top level, so the shared post-load
        # tie must stay a no-op. Same guard as the other qwen3omni modules.
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
