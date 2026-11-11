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

"""Asset binder for :class:`LladaUniTextEncoder`.

Because ``LladaUniTextEncoder`` splices per-image VQ token ids produced by the
sibling ``llada2_uni_image_tokenizer``'s GPU ``encode`` forward, tokenization
and dLLM masking happen inside ``encode_pre`` after ``image_tokenizer.encode``
has run. This preprocessor therefore binds the HuggingFace tokenizer onto the
module (via ``bind_module_assets``) and performs no worker-side mutation on
``conversation_list``.
"""

from __future__ import annotations

from typing import Any

from .....auto import build_tokenizer
from ...module_processing_base import ModulePreprocessorBase


class LladaUniTextEncoderPreprocessor(ModulePreprocessorBase):
    """Tokenizer asset binder for :class:`LladaUniTextEncoder`."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "LladaUniTextEncoderPreprocessor":
        del config_overrides, kwargs
        return cls(build_tokenizer(module_path))

    def __call__(self, batch: dict[str, Any], inference: bool = False, **kwargs: Any) -> None:
        del batch, inference, kwargs
        return None


__all__ = ["LladaUniTextEncoderPreprocessor"]
