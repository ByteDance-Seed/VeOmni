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

"""VeOmni per-module runtime config — the accelerated counterpart of ``OmniModuleConfig``.

:class:`OmniModuleRuntimeConfig` is one module's slice of a composed Omni model:
the inherited :class:`~veomni.arguments.arguments_types.ModelArguments` fields
(``model_path``, ``ops_implementation``, ``accelerator``, ``optimizer``, freeze,
LoRA) plus the projection onto an HF :class:`OmniModuleConfig` descriptor.

It does **not** inherit :class:`~veomni.models.seed_omni.configuration_omni.OmniConfig`
or :class:`~veomni.models.seed_omni.modules.module_configuration_base.OmniModuleConfig`:
those are HuggingFace checkpoint shapes. This dataclass is the launcher/runtime
view; :meth:`to_hf_config` is the one-way projection.

``OmniModuleRuntimeArguments`` is the public alias kept in
:mod:`veomni.arguments.omni_arguments_types` so existing launcher imports keep
working. YAML resolution (``resolve_omni_model``, ``build_module_runtime_args``)
stays in ``arguments/`` to avoid an arguments ↔ accelerated import cycle.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .....arguments.arguments_types import ModelArguments


def hf_module_model_config(model_config: dict | None) -> dict:
    """Drop launcher layout keys before merging or exporting per-module ``model_config``."""
    from .....arguments.omni_arguments_types import LAUNCHER_CONFIG_KEYS

    if not model_config:
        return {}
    return {key: value for key, value in model_config.items() if key not in LAUNCHER_CONFIG_KEYS}


@dataclass
class OmniModuleRuntimeConfig(ModelArguments):
    """Per-module runtime — one module's slice of a composed Omni model.

    ``ModelArguments`` already is a complete training unit: the model
    fields plus this module's own ``accelerator`` and ``optimizer``, with
    ``model_path`` localized and ``fqn_to_index_mapping`` parsed lazily and cached
    per index path (several modules routinely share one checkpoint). All this adds
    is the projection onto an HF module descriptor.
    """

    def to_hf_config(self) -> dict:
        """Project onto this module's ``OmniConfig._module_entries`` entry.

        The module's name is the key that entry is filed under, so it is not
        repeated inside it.

        ``model_path`` is carried through explicitly: by the time this runs,
        ``build_module_runtime_args`` / ``_resolve_model_path`` has already
        resolved it to an absolute path — usually ``<checkpoint_root>/<name>``,
        but a launcher YAML module override may point it at a wholly different
        checkpoint (e.g. Qwen3 visual-instruction-tuning composing
        ``qwen3_llm``/``qwen3_text_encoder`` from one HF model with
        ``qwen3vl_vision`` from another). Dropping it and re-deriving
        ``checkpoint_root/name`` downstream (as
        :meth:`~veomni.models.seed_omni.modules.module_configuration_base.OmniModuleConfig.resolve_path`
        does for an entry without one) would silently resolve to the wrong path
        for that module.

        Kernels are not projected: the runtime builds each module with its own
        ``args.ops_implementation``, so an entry stating them too would be a
        second source of truth.
        """
        entry: dict = {}
        if self.model_path:
            entry["model_path"] = self.model_path
        if model_config := hf_module_model_config(self.model_config):
            entry["model_config"] = deepcopy(model_config)
        if self.processor_config:
            entry["processor_config"] = deepcopy(self.processor_config)
        return entry


__all__ = [
    "OmniModuleRuntimeConfig",
    "hf_module_model_config",
]
