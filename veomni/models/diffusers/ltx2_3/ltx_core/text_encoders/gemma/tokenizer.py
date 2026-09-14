from enum import Enum

from transformers import AutoTokenizer


class PaddingSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class LTXVGemmaTokenizer:
    """Tokenizer wrapper for Gemma models compatible with LTXV processes."""

    def __init__(self, tokenizer_path: str, max_length: int = 256, padding_side: PaddingSide = PaddingSide.LEFT):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, model_max_length=max_length
        )
        self.tokenizer.padding_side = padding_side.value
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False) -> dict[str, list[tuple[int, int]]]:
        text = text.strip()
        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        tuples = [
            (token_id, attn, i) for i, (token_id, attn) in enumerate(zip(input_ids[0], attention_mask[0], strict=True))
        ]
        out = {"gemma": tuples}
        if not return_word_ids:
            out = {k: [(t, w) for t, w, _ in v] for k, v in out.items()}
        return out
