"""CPU preprocessor helpers for direct V2 parity runners."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


def apply_training_cpu_preprocessors(model: Any, batch: MutableMapping[str, Any]) -> None:
    """Mirror SeedOmniCollator's worker preprocessing for direct train runners."""

    conversation_list = batch.get("conversation_list")
    if not isinstance(conversation_list, list):
        return

    for module in model.modules_dict.values():
        builder = getattr(module, "build_cpu_preprocessor", None)
        preprocessor = builder() if builder is not None else None
        if preprocessor is not None:
            preprocessor(conversation_list)


__all__ = ["apply_training_cpu_preprocessors"]
