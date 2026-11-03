"""Shared preprocessing for single-text-sequence decision models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class DecisionPreprocessor(ABC):
    """Build model features while architectures own request serialization."""

    tokenizer: object
    context_limit: int
    vocab_size: int
    position_hook: Callable | None = None

    @classmethod
    def from_assets(cls, tokenizer, config, *, position_hook=None):
        text_config = getattr(config, "text_config", config)
        return cls(
            tokenizer,
            text_config.max_position_embeddings,
            text_config.vocab_size,
            position_hook,
        )

    @abstractmethod
    def encode(self, request, *, max_length: int) -> tuple[list[int], dict]:
        """Return token IDs and architecture-specific decision metadata."""

    def build_features(self, input_ids: list[int]) -> dict:
        ids = torch.tensor(input_ids, dtype=torch.long)
        if ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError("Tokenizer token IDs exceed the native checkpoint vocabulary")
        features = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if self.position_hook is not None:
            positions = self.position_hook(
                input_ids=ids.unsqueeze(0), attention_mask=features["attention_mask"].unsqueeze(0)
            )
            features["position_ids"] = positions["position_ids"].squeeze(-2)
        return features

    def __call__(self, request, *, max_length: int) -> tuple[dict, dict]:
        input_ids, meta = self.encode(request, max_length=min(max_length, self.context_limit))
        return self.build_features(input_ids), meta
