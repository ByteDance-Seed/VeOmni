"""Minimal BAGEL transformers config for test-reference smoke coverage."""

from __future__ import annotations

from transformers import PretrainedConfig


class BagelReferenceConfig(PretrainedConfig):
    model_type = "bagel_test_reference"

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_size: int = 32,
        num_hidden_layers: int = 2,
        intermediate_size: int = 64,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
