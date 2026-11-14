"""Worker-side preprocessor for :class:`Qwen3TextEncoder`."""

from __future__ import annotations

import difflib
from typing import Any

from .....auto import build_tokenizer
from ...qwen3vl.text_encoder.chat_template import Qwen3VLChatTemplate
from ...text_encoder.chat_template import TextEncoderChatTemplate
from ...text_encoder.processing import TextEncoderPreprocessor
from .chat_template import Qwen3ChatTemplate
from .configuration import Qwen3TextEncoderConfig


class Qwen3TextEncoderPreprocessor(TextEncoderPreprocessor):
    """Qwen3, or Qwen3-VL (image mode), chat template + tokenize worker."""

    @classmethod
    def build_chat_template(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> TextEncoderChatTemplate:
        """Pick the image-aware template when the module config says so.

        ``config_overrides`` mirrors the module's YAML ``model_config:`` block, so a
        worker built independently of the live model still selects the template that
        model was configured with. It goes through the same ``model_config=`` path
        the model side uses.

        The block is applied with ``update``, which accepts a misspelled key as a
        stray attribute — ``enable_imag: true`` would load fine and quietly leave
        the worker on the text-only template. So a key that is one character or so
        away from a config field it does not match exactly is reported. Only
        *near-misses* are rejected: HF load options in the same dict
        (``device_map``, ``torch_dtype``, …) are not config fields at all, and a
        valid public property must be recognized before the fuzzy match runs.
        """
        config = Qwen3TextEncoderConfig.from_pretrained(module_path, model_config=config_overrides)
        # The reference set is a *fresh* config: after the overrides above a
        # misspelled key is an attribute too, so the loaded config cannot validate
        # itself. Public properties are included as well as the stored attributes —
        # transformers v5 keeps several under a leading underscore
        # (`output_attentions` -> `_output_attentions`), and those are valid keys,
        # not typos.
        reference = Qwen3TextEncoderConfig()
        known = set(vars(reference)) | {name for name in dir(type(reference)) if not name.startswith("__")}
        misspelled = {
            key: difflib.get_close_matches(key, known, n=1, cutoff=0.85)[0]
            for key in (config_overrides or {})
            if key not in known and difflib.get_close_matches(key, known, n=1, cutoff=0.85)
        }
        if misspelled:
            raise ValueError(
                "qwen3_text_encoder: model_config key(s) look misspelled — "
                + ", ".join(f"{bad!r} (did you mean {good!r}?)" for bad, good in sorted(misspelled.items()))
            )
        tokenizer = build_tokenizer(module_path)
        return Qwen3VLChatTemplate(tokenizer) if config.enable_image else Qwen3ChatTemplate(tokenizer)


__all__ = ["Qwen3TextEncoderPreprocessor"]
