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
#      :class:`BaseMixin`, :class:`TrainingGraph`, :class:`GenerationGraph`).
#   2. Module registries — :data:`OMNI_CONFIG_REGISTRY`,
#      :data:`OMNI_MODEL_REGISTRY`, :data:`OMNI_PROCESSOR_REGISTRY` — resolve
#      ``model_type → class`` lazily at runtime.
from .configuration_omni import OmniConfig
from .graphs.base import END, EdgeDef, NodeDef
from .graphs.generation_graph import GenerationGraph
from .graphs.training_graph import TrainingGraph
from .mixins.base_mixin import BaseMixin
from .mixins.inference_module_mixin import InferenceModuleMixin
from .mixins.metric_meter_mixin import MetricMeterMixin
from .mixins.offline_encoding_mixin import OfflineEncodingConfigMixin, OfflineEncodingMixin
from .mixins.training_module_mixin import TrainingModuleMixin
from .modeling_omni import OmniModel
from .modules import (
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
    read_hf_model_type,
    read_model_type,
)
from .processing_omni import OmniProcessor
from .utils.conversation import build_conversation


__all__ = [
    # Core
    "OmniConfig",
    "OmniModel",
    "OmniProcessor",
    "BaseMixin",
    "TrainingModuleMixin",
    "InferenceModuleMixin",
    "MetricMeterMixin",
    "OfflineEncodingConfigMixin",
    "OfflineEncodingMixin",
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
    "read_hf_model_type",
    "read_model_type",
]
