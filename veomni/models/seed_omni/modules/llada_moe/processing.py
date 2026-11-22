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

"""CPU worker for :class:`LladaMoeOmni`.

The objective is identical to :mod:`..llada`: chat-template, tokenize and draw the
dLLM corruption in one call, then hand the tensors to the graph. Only the chat
template differs (Bailing's ``llada_moe``), so this subclasses the dense worker
and overrides that one default rather than duplicating the pipeline -- a second
implementation would drift from the corruption the non-Omni trainer draws, which
is exactly the comparison this module exists to support.
"""

from __future__ import annotations

from typing import Any

from ..llada.processing import LladaOmniPreprocessor


class LladaMoeOmniPreprocessor(LladaOmniPreprocessor):
    """LLaDA's CPU preprocessor with the LLaDA-MoE chat template."""

    @classmethod
    def from_pretrained(
        cls,
        module_path: str,
        *,
        config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "LladaMoeOmniPreprocessor":
        """Build from the checkpoint subfolder, defaulting to the ``llada_moe`` template."""
        overrides = dict(config_overrides or {})
        overrides.update(kwargs)
        overrides.setdefault("chat_template", "llada_moe")
        return super().from_pretrained(module_path, config_overrides=overrides)


__all__ = ["LladaMoeOmniPreprocessor"]
