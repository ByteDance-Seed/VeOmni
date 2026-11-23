"""Config for :class:`Qwen3OmniTextEncoder`."""

from ...text_encoder.configuration import TextEncoderConfig


class Qwen3OmniTextEncoderConfig(TextEncoderConfig):
    """TextEncoder config for Qwen3-Omni ChatML tokenization (text + image + video + audio)."""

    model_type = "qwen3omni_text_encoder"

    def __init__(
        self,
        publish_talker_prompt: bool = False,
        freeze: bool = False,
        add_speech_system_prompt: bool = False,
        **kwargs,
    ):
        # Whether ``encode`` also builds the token-level prompt view the talker
        # rebuilds its 3D RoPE from. Only a graph with a talker reads it, so like
        # ``qwen3omni_llm.collect_thinker_states`` this is the scenario's call and
        # lives in its ``modules_*.yaml``. The talker says which flag is missing
        # if it runs without one, so the pair cannot half-fail in silence.
        self.publish_talker_prompt = publish_talker_prompt
        # Talker training freezes the whole thinker, this module's wte and
        # lm_head included. The two flags travel together in that round: the
        # talker needs the prompt view, and needs no gradient through it.
        self.freeze = freeze
        # Whether the template opens a round with the checkpoint's speech system
        # turn (``chat_template.SPEECH_SYSTEM_PROMPT``). The scenario's call, like
        # the flag above: speech output needs it or the thinker never reaches
        # ``<|im_end|>`` and the talker never runs, while a text-only round wants
        # it off — it caps replies at 50 words and imposes a voice persona, which
        # would skew a text evaluation. Talker training sets it too, so the round
        # trains on the same opening turn generation used.
        self.add_speech_system_prompt = add_speech_system_prompt
        # transformers 5.x auto-generates an ``__init__`` for an annotation-less
        # ``PreTrainedConfig`` subclass, which would shadow ``TextEncoderConfig``'s
        # and leave ``lm_head_bias`` / ``vocab_size`` unset. The Qwen3-Omni thinker
        # is untied (``tie_word_embeddings=False``), and that is the branch that
        # reads ``lm_head_bias`` — so without this forward the module cannot be
        # built at all. Keep the explicit forward even though it looks redundant.
        super().__init__(**kwargs)
