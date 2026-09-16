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

# Set once this function has run a preprocessor's assets onto a model, so a
# second bind is a no-op without having to guess which assets that module's
# preprocessor was supposed to provide.
_BOUND_FLAG_ATTR = "_omni_assets_bound"


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

    An asset already on ``model`` wins — a caller that set ``model.tokenizer``
    by hand keeps it, and still gets the module's other assets. Each asset is
    considered on its own: keying the whole bind on "some asset is set" would
    let one pre-set attribute (``tokenizer`` has a public setter) silently cost
    a module its image / video processors. No-op when the module declares no
    ``preprocessor_class``, when assets were already bound here, or when
    ``from_pretrained`` returns ``None`` (modules with no CPU worker).
    """
    if getattr(model, _BOUND_FLAG_ATTR, False):
        return

    if preprocessor is None:
        preprocessor_cls = getattr(type(model), "preprocessor_class", None)
        if preprocessor_cls is None or checkpoint_path is None:
            return
        preprocessor = preprocessor_cls.from_pretrained(checkpoint_path, config_overrides=config_overrides)

    if preprocessor is None:
        return

    for attr in MODULE_ASSET_ATTRS:
        if hasattr(preprocessor, attr) and getattr(model, attr, None) is None:
            setattr(model, attr, getattr(preprocessor, attr))
    setattr(model, _BOUND_FLAG_ATTR, True)


__all__ = ["MODULE_ASSET_ATTRS", "bind_module_assets"]
