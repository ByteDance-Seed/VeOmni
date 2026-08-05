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

"""SeedOmni V2 training / inference orchestrators (independent from V1 ``BaseTrainer``)."""

from ...models.seed_omni.accelerator import OmniModelRuntime
from ...models.seed_omni.accelerator.checkpoint import OmniModuleCheckpointManager
from ...models.seed_omni.accelerator.module_runtime import ModuleRuntime
from ...omni_arguments.model_runtime import build_module_runtime_args
from ..callbacks.omni_callbacks import (
    OmniGlobalStateCallback,
    OmniModuleDcpCallback,
    OmniModuleHfCallback,
    OmniRootAssetsCallback,
)
from .omni_inferencer import InferenceRequest, OmniInferencer
from .omni_trainer import MultiLRScheduler, MultiOptimizer, OmniTrainer


OmniModuleTrainer = ModuleRuntime

__all__ = [
    "OmniTrainer",
    "ModuleRuntime",
    "OmniModuleTrainer",
    "OmniModuleCheckpointManager",
    "OmniModelRuntime",
    "OmniInferencer",
    "InferenceRequest",
    "build_module_runtime_args",
    "MultiOptimizer",
    "MultiLRScheduler",
    "OmniModuleDcpCallback",
    "OmniModuleHfCallback",
    "OmniGlobalStateCallback",
    "OmniRootAssetsCallback",
]
