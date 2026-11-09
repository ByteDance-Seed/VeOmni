"""Weight-free CLEF preprocessing shared by training and prediction."""

from ...processing import DecisionPreprocessor
from .specification import encode_clef


class ClefPreprocessor(DecisionPreprocessor):
    def encode(self, request, *, max_length: int) -> tuple[list[int], dict]:
        return encode_clef(request, self.tokenizer, max_length=max_length)
