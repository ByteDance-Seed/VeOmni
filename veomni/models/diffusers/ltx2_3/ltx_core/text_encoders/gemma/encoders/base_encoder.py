import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import torch
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from transformers import Gemma3ForConditionalGeneration

from veomni.utils.device import IS_CUDA_AVAILABLE


class GemmaTextEncoder(torch.nn.Module):
    """Pure Gemma text encoder — runs the LLM and returns raw hidden states."""

    def __init__(
        self,
        model: Gemma3ForConditionalGeneration | None = None,
        tokenizer: LTXVGemmaTokenizer | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._dtype = dtype

    def encode(
        self,
        text: str,
        padding_side: str = "left",  # noqa: ARG002
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Run Gemma LLM and return raw hidden states + attention mask."""
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=self.model.device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=self.model.device)
        outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        del outputs
        return hidden_states, attention_mask

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer_path: str | None = None,
        max_length: int = 1024,
        device: torch.device | str | int | None = None,
    ):
        """Load Gemma text encoder from pretrained weights in 8-bit precision."""
        from transformers import BitsAndBytesConfig

        gemma_path = _find_gemma_subpath(model_path, "model*.safetensors")
        tokenizer_path = _find_gemma_subpath(tokenizer_path or model_path, "tokenizer.model")

        # Pin the entire model to a single device. `device_map="auto"` collides on cuda:0
        # in multi-process launches because every rank picks the same default device.
        device_map: str | dict[str, int | str | torch.device]
        if device is not None:
            device_map = {"": device}
        elif IS_CUDA_AVAILABLE:
            device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        else:
            device_map = "auto"

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        with _suppress_accelerate_memory_warnings():
            model = Gemma3ForConditionalGeneration.from_pretrained(
                gemma_path,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                local_files_only=True,
            )
        tokenizer = LTXVGemmaTokenizer(tokenizer_path, max_length=max_length)
        return cls(model=model, tokenizer=tokenizer, dtype=torch.bfloat16)


def _find_gemma_subpath(root_path: str | Path, pattern: str) -> str:
    matches = list(Path(root_path).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return str(matches[0].parent)


@contextmanager
def _suppress_accelerate_memory_warnings() -> Generator[None, None, None]:
    accelerate_logger = logging.getLogger("accelerate.utils.modeling")
    old_level = accelerate_logger.level
    accelerate_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        accelerate_logger.setLevel(old_level)
