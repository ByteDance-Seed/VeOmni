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

# ── SeedOmni V2 public API ──────────────────────────────────────────────────
# Exports cluster around three concerns:
#
#   1. Core graph / runtime types (:class:`OmniConfig`, :class:`OmniModel`,
#      :class:`ModuleMixin`, :class:`TrainingGraph`, :class:`GenerationGraph`).
#   2. Module registries — :data:`OMNI_CONFIG_REGISTRY`,
#      :data:`OMNI_MODEL_REGISTRY`, :data:`OMNI_PROCESSOR_REGISTRY` — resolve
#      ``model_type → class`` lazily at runtime.
from .configuration_omni import OmniConfig
from .conversation import build_conversation
from .generation_graph import GenerationGraph
from .graph import END, EdgeDef, NodeDef
from .modeling_omni import OmniModel
from .module import ModuleMixin
from .modules import (
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
    read_model_type,
)
from .training_graph import TrainingGraph


__all__ = [
    # Core
    "OmniConfig",
    "OmniModel",
    "ModuleMixin",
    "TrainingGraph",
    "GenerationGraph",
    "NodeDef",
    "EdgeDef",
    "END",
    "build_conversation",
    # Module registry
    "OMNI_CONFIG_REGISTRY",
    "OMNI_MODEL_REGISTRY",
    "OMNI_PROCESSOR_REGISTRY",
    "read_model_type",
]
