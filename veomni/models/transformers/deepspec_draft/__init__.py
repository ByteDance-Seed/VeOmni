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

"""Register DeepSpec draft models into VeOmni's model registry.

DeepSpec draft models (``Qwen3DSparkModel``, ``Gemma4DSparkModel``,
``Qwen3Eagle3Model``, ``Gemma4Eagle3Model``) are ``transformers.PreTrainedModel``
subclasses. VeOmni already knows how to meta-init + FSDP2-shard any
``PreTrainedModel`` whose class it can look up from ``config.model_type`` via the
``MODELING_REGISTRY`` (when ``MODELING_BACKEND != "hf"``, the default).

The one wrinkle: a DeepSpec draft ``config.json`` carries the *target's*
``model_type`` (``"qwen3"`` / ``"gemma4"`` …) because the draft config is a deep
copy of the target config. If we registered under those keys we'd shadow
VeOmni's real Qwen3/Gemma models. Instead the draft-init prep step
(``scripts/deepspec/prepare_draft_init.py``) rewrites ``model_type`` to a
dedicated ``"deepspec_draft"`` key and records the concrete draft class in
``architectures`` (e.g. ``["Qwen3DSparkModel"]``). We register:

* a **config class** (``DeepSpecDraftConfig``) under ``MODEL_CONFIG_REGISTRY``,
  because HF ``AutoConfig`` does not know ``deepspec_draft``;
* a **modeling factory** under ``MODELING_REGISTRY`` that dispatches on the
  architecture string to the right DeepSpec draft class.

This module must be imported at package import time so the decorators run before
``build_foundation_model`` is called.
"""

from typing import Type

from ....integrations.deepspec import ensure_deepspec_importable
from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


# VeOmni ``model_type`` used for every DeepSpec draft variant. The concrete
# algorithm/architecture is disambiguated by ``config.architectures[0]``.
DEEPSPEC_DRAFT_MODEL_TYPE = "deepspec_draft"


# Map the draft class name (as written into ``config.architectures``) to the
# import path of the DeepSpec class that implements it. Kept as strings so this
# module imports cheaply; the actual class is imported lazily inside the factory
# (mirrors how VeOmni's qwen3/__init__.py defers the generated-module import).
_ARCHITECTURE_IMPORTS = {
    "Qwen3DSparkModel": (
        "veomni.models.transformers.deepspec_draft.generated.patched_modeling_dspark_qwen3_gpu",
        "Qwen3DSparkModel",
    ),
    "Gemma4DSparkModel": ("deepspec.modeling.dspark.gemma4", "Gemma4DSparkModel"),
    "Qwen3Eagle3Model": ("deepspec.modeling.eagle3.qwen3", "Qwen3Eagle3Model"),
    "Gemma4Eagle3Model": ("deepspec.modeling.eagle3.gemma4", "Gemma4Eagle3Model"),
}


def _import_draft_class(architecture: str) -> Type:
    ensure_deepspec_importable()
    import importlib

    if architecture not in _ARCHITECTURE_IMPORTS:
        raise ValueError(
            f"Unknown DeepSpec draft architecture {architecture!r}. "
            f"Expected one of {sorted(_ARCHITECTURE_IMPORTS)}. Make sure the "
            "draft config.json's `architectures` field names a supported draft "
            "class (set by scripts/deepspec/prepare_draft_init.py)."
        )
    module_path, class_name = _ARCHITECTURE_IMPORTS[architecture]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@MODEL_CONFIG_REGISTRY.register(DEEPSPEC_DRAFT_MODEL_TYPE)
def register_deepspec_draft_config():
    from .configuration_deepspec_draft import DeepSpecDraftConfig

    return DeepSpecDraftConfig


@MODELING_REGISTRY.register(DEEPSPEC_DRAFT_MODEL_TYPE)
def register_deepspec_draft_modeling(architecture: str):
    """Return the DeepSpec draft class named by ``architecture``.

    ``build_foundation_model`` calls this with ``config.architectures[0]``.
    """
    return _import_draft_class(architecture)


__all__ = [
    "DEEPSPEC_DRAFT_MODEL_TYPE",
    "register_deepspec_draft_config",
    "register_deepspec_draft_modeling",
]
