"""Configuration for :class:`TextEncoder`.

The ``model_type`` string is the lookup key used by
``OMNI_CONFIG_REGISTRY`` / ``OMNI_MODEL_REGISTRY`` (see
``modules/__init__.py``).  Every other field is plumbed through to
``TextEncoder.__init__``.
"""

from transformers import PretrainedConfig


class TextEncoderConfig(PretrainedConfig):
    """Config for the generic word-token embedding + LM head module.

    Parameters
    ----------
    vocab_size:
        Number of token IDs in the embedding / projection.
    hidden_size:
        LLM hidden-state dimension.  Must match the backbone model.
    tie_word_embeddings:
        If ``True``, ``decode`` projects via ``embed_tokens.weight`` (no
        separate ``lm_head``).  If ``False``, an independent
        ``nn.Linear`` is allocated.  Default: ``True``.
    lm_head_bias:
        Only meaningful when ``tie_word_embeddings`` is ``False``.  When
        ``True`` the untied ``lm_head`` gains a bias term.  Default:
        ``False``.
    """

    model_type = "text_encoder"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        tie_word_embeddings: bool = True,
        lm_head_bias: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tie_word_embeddings = tie_word_embeddings
        self.lm_head_bias = lm_head_bias
        super().__init__(**kwargs)
