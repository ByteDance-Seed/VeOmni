"""Cosmos3 prompt encoder: the reasoner tower's ``embed_tokens`` (+ its unused ``lm_head``).

Every reasoner *block* runs inside the joint two-pass attention and therefore
lives in ``cosmos3_mot``; the word embedding is all that stays here, which is
exactly what the shared :class:`TextEncoder` is. ``lm_head`` is kept so the
module round-trips the released checkpoint, and is never trained: there is no
autoregressive loss.
"""

from ...text_encoder.modeling import TextEncoder
from .configuration import Cosmos3TextEncoderConfig
from .processing import Cosmos3TextEncoderPreprocessor


class Cosmos3TextEncoder(TextEncoder):
    config_class = Cosmos3TextEncoderConfig
    preprocessor_class = Cosmos3TextEncoderPreprocessor

    def freeze_model(self) -> None:
        self.embed_tokens.requires_grad_(self.config.train_joint_reasoner)
        if self.lm_head is not None:
            self.lm_head.requires_grad_(False)


__all__ = ["Cosmos3TextEncoder"]
