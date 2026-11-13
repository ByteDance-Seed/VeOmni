"""Config for :class:`Cosmos3TextEncoder`."""

from typing import Optional, Sequence

from ...text_encoder.configuration import TextEncoderConfig


class Cosmos3TextEncoderConfig(TextEncoderConfig):
    """The reasoner tower's ``embed_tokens`` / ``lm_head`` and its prompt format.

    ``generation_marker_ids`` is ``[eos, <|vision_start|>]``, appended after the
    ``max_sequence_length`` truncation as the reference's packer does; the
    converter resolves both ids from the checkpoint's tokenizer.
    ``train_joint_reasoner`` releases ``embed_tokens`` together with the MoT's
    reasoner tower; ``lm_head`` stays frozen either way, since nothing trains an
    autoregressive loss.
    """

    model_type = "cosmos3_text_encoder"

    def __init__(
        self,
        generation_marker_ids: Optional[Sequence[int]] = None,
        max_sequence_length: int = 2048,
        train_joint_reasoner: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.generation_marker_ids = [int(token_id) for token_id in generation_marker_ids or ()]
        self.max_sequence_length = max_sequence_length
        self.train_joint_reasoner = train_joint_reasoner
