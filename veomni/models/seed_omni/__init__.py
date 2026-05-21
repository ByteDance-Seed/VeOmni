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
#      :class:`OmniModule`, :class:`TrainingGraph`, :class:`GenerationGraph`).
#   2. Module mixin registry — :data:`MODULE_MIXIN_REGISTRY`,
#      :data:`MODULE_PROCESSOR_REGISTRY` — used by :class:`OmniTrainer`
#      (Step 2 wiring) to resolve ``model_type → mixin class``.
#   3. Per-module checkpoint callback for Step-2 trainer integration.
from .checkpoint_callback import OmniModuleCheckpointCallback
from .configuration_seed_omni import OmniConfig
from .generation_graph import GenerationGraph
from .graph import END, EdgeDef, NodeDef
from .modeling_omni import OmniModel
from .module import OmniModule
from .modules import MODULE_MIXIN_REGISTRY, MODULE_PROCESSOR_REGISTRY
from .training_graph import TrainingGraph


__all__ = [
    # Core
    "OmniConfig",
    "OmniModel",
    "OmniModule",
    "TrainingGraph",
    "GenerationGraph",
    "NodeDef",
    "EdgeDef",
    "END",
    # Module registry
    "MODULE_MIXIN_REGISTRY",
    "MODULE_PROCESSOR_REGISTRY",
    # Lifecycle
    "OmniModuleCheckpointCallback",
]
