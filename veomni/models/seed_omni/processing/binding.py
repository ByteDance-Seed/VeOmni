# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Copy HF asset handles from a CPU worker onto a live ``nn.Module`` instance."""

from __future__ import annotations

from typing import Any

from .base import ModulePreprocessorBase


# Attributes a ``XxxPreprocessor`` may hold that ``forward`` / ``generate`` read
# directly on the model (``self._image_processor``, ``self.tokenizer``, …).
MODULE_ASSET_ATTRS = ("_processor", "_image_processor", "_video_processor", "_tokenizer", "_chat_template")

_BOUND_CHECK_ATTRS = ("_image_processor", "_video_processor", "_tokenizer")


def bind_module_assets(
    model: Any,
    *,
    checkpoint_path: str | None = None,
    preprocessor: ModulePreprocessorBase | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> None:
    """Attach HF assets from a CPU worker onto ``model``.

    Pass either a built ``preprocessor``, or ``checkpoint_path`` (reads
    ``preprocessor_class`` off ``type(model)`` and calls
    :meth:`ModulePreprocessorBase.from_pretrained`).  Copies well-known asset attributes
    (``_image_processor``, ``_tokenizer``, …) onto the instance names that
    ``forward`` / ``generate`` already read — no rebuild at bind time.

    No-op when the module declares no ``preprocessor_class``, assets are already
    bound, or ``from_pretrained`` returns ``None`` (modules with no CPU worker).
    """
    if any(getattr(model, attr, None) is not None for attr in _BOUND_CHECK_ATTRS):
        return

    if preprocessor is None:
        preprocessor_cls = getattr(type(model), "preprocessor_class", None)
        if preprocessor_cls is None or checkpoint_path is None:
            return
        preprocessor = preprocessor_cls.from_pretrained(checkpoint_path, config_overrides=config_overrides)

    if preprocessor is None:
        return

    for attr in MODULE_ASSET_ATTRS:
        if hasattr(preprocessor, attr):
            setattr(model, attr, getattr(preprocessor, attr))


__all__ = ["MODULE_ASSET_ATTRS", "bind_module_assets"]
