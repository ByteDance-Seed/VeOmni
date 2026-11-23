"""VeOmni-accelerated Cosmos3TextEncoder.

The shared text-encoder hooks already pack the tokenized prompts, embed them
and scatter the embeddings back onto the text items; only the template differs.
"""

from typing import Any, List

from ....text_encoder.accelerated import VeOmniMixin as BaseVeOmniMixin
from ..chat_template import Cosmos3ChatTemplate
from ..configuration import Cosmos3TextEncoderConfig
from ..modeling import Cosmos3TextEncoder


class VeOmniMixin(BaseVeOmniMixin):
    config: Cosmos3TextEncoderConfig
    _chat_template: Cosmos3ChatTemplate

    def estimate_flops(self, seqlens: List[int]) -> float:
        # The base counts the lm_head projection; Cosmos3 only looks embeddings up.
        del seqlens
        return 0.0

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = Cosmos3ChatTemplate(
            tokenizer,
            generation_marker_ids=self.config.generation_marker_ids,
            max_sequence_length=self.config.max_sequence_length,
        )


class Cosmos3TextEncoderAccelerated(VeOmniMixin, Cosmos3TextEncoder):
    pass


__all__ = ["Cosmos3TextEncoderAccelerated"]
