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

"""VeOmni composite runtime config — the accelerated counterpart of ``OmniConfig``.

:class:`OmniModelRuntimeConfig` is one composed Omni model: the inherited
:class:`~veomni.arguments.arguments_types.ModelArguments` fields (which double as
the defaults each module is merged over) plus the modules it decomposes into
and every training / generation graph scenario.

It does **not** inherit :class:`~veomni.models.seed_omni.configuration_omni.OmniConfig`:
that is the HuggingFace checkpoint shape. This dataclass is the launcher/runtime
view; :meth:`to_hf_config` is the one-way projection.

``OmniModelRuntimeArguments`` is the public alias kept in
:mod:`veomni.arguments.omni_arguments_types` so existing launcher imports keep
working. YAML resolution (``resolve_omni_model``, ``build_omni_model_runtime``)
stays in ``arguments/`` to avoid an arguments ↔ accelerated import cycle.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .....arguments.arguments_types import ModelArguments
from ..omni_module.omni_module_config import OmniModuleRuntimeConfig


@dataclass
class OmniModelRuntimeConfig(ModelArguments):
    """One composed Omni model — a training unit plus the modules it decomposes into.

    YAML supplies the inherited ``model_path``, ``model_config``,
    ``ops_implementation``, ``accelerator`` and ``optimizer``, which double as the
    defaults each module's own block is merged over. :func:`~veomni.arguments.omni_arguments_types.resolve_omni_model`
    fills ``modules``, the graph scenario maps, and the scenario keys.
    """

    modules: dict[str, OmniModuleRuntimeConfig] = field(default_factory=dict)
    training_graphs: dict[str, Any] = field(default_factory=dict)
    generation_graphs: dict[str, Any] = field(default_factory=dict)
    train_type: str | None = None
    infer_type: str | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)

    def launcher_config(self, key: str, default: Any = None) -> Any:
        """Read a launcher layout key from ``model_config`` (``modules``, graphs, …)."""
        return (self.model_config or {}).get(key, default)

    def set_launcher_config(self, key: str, value: Any) -> None:
        if self.model_config is None:
            self.model_config = {}
        self.model_config[key] = value

    @property
    def resolved_model_path(self) -> str:
        path = self.model_path
        if not path:
            raise ValueError("`model.model_path` (split-checkpoint root) is required for OmniModel V2.")
        return path

    @property
    def module_names(self) -> list[str]:
        return list(self.modules)

    @property
    def train_types(self) -> list[str]:
        return list(self.training_graphs)

    @property
    def infer_types(self) -> list[str]:
        return list(self.generation_graphs)

    @property
    def training_graph(self) -> list[dict]:
        from ...configuration_omni import select_graph

        graph = select_graph(
            self.training_graphs,
            self.train_type,
            empty_hint="Populate `model.model_config.train_graph` with at least one scenario.",
            unknown_hint="train_type",
        )
        return list(graph)

    @property
    def generation_graph(self) -> dict:
        from ...configuration_omni import select_graph

        return select_graph(
            self.generation_graphs,
            self.infer_type,
            empty_hint="Populate `model.model_config.infer_graph` with at least one scenario.",
            unknown_hint="infer_type",
        )

    def module_checkpoint_subfolder(self, name: str) -> str:
        if name not in self.modules:
            known = ", ".join(self.modules) or "(none)"
            raise KeyError(f"Module {name!r} not found in model runtime; known modules: {known}.")
        return name

    def to_hf_config(self):
        """Project onto the checkpoint-shaped HF :class:`~veomni.models.seed_omni.configuration_omni.OmniConfig`.

        Both scenario maps are carried whole, with the keys that select them:
        ``OmniConfig.training_graph`` / ``generation_graph`` are read-only views
        of ``training_graphs[train_type]`` / ``generation_graphs[infer_type]``,
        so handing over only the active graph would lose every other scenario a
        launcher declared.
        """
        from ...configuration_omni import OmniConfig

        module_entries = {name: mod.to_hf_config() for name, mod in self.modules.items()}
        return OmniConfig.from_dict(
            {
                "_module_entries": module_entries,
                "training_graphs": deepcopy(self.training_graphs),
                "generation_graphs": deepcopy(self.generation_graphs),
                "train_type": self.train_type,
                "infer_type": self.infer_type,
                "generation_kwargs": dict(self.generation_kwargs),
            }
        )


__all__ = ["OmniModelRuntimeConfig"]
