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

"""CLI argument parsing — V1 ``VeOmniArguments`` and V2 ``OmniArguments``."""

from __future__ import annotations

from .arguments_types import (
    AcceleratorConfig,
    BaseModelArguments,
    CheckpointConfig,
    ChunkMBSConfig,
    DataArguments,
    DataloaderConfig,
    FSDPConfig,
    GradientCheckpointingConfig,
    InferArguments,
    MixedPrecisionConfig,
    ModelArguments,
    ModelRuntimeArguments,
    OffloadConfig,
    OpsImplementationConfig,
    OptimizerConfig,
    ProfileConfig,
    TorchCompileConfig,
    TrainingArguments,
    VeOmniArguments,
    WandbConfig,
)
from .omni_arguments_types import (
    DEFAULT_SCENARIO,
    OMNI_TRAIN_WORKFLOWS,
    OmniArguments,
    OmniDataArguments,
    OmniGraphProfileArguments,
    OmniInferArguments,
    OmniModelRuntimeArguments,
    OmniModuleRuntimeArguments,
    OmniTrainingArguments,
    build_module_args,
    build_module_runtime_args,
    build_omni_model_runtime,
    resolve_omni_model,
)
from .omni_parser import load_yaml_with_inherit, parse_omni_args
from .parser import parse_args, save_args


__all__ = [
    "AcceleratorConfig",
    "BaseModelArguments",
    "ModelRuntimeArguments",
    "CheckpointConfig",
    "ChunkMBSConfig",
    "DataArguments",
    "DataloaderConfig",
    "FSDPConfig",
    "GradientCheckpointingConfig",
    "InferArguments",
    "MixedPrecisionConfig",
    "ModelArguments",
    "OffloadConfig",
    "OpsImplementationConfig",
    "OptimizerConfig",
    "ProfileConfig",
    "TorchCompileConfig",
    "TrainingArguments",
    "VeOmniArguments",
    "WandbConfig",
    "OmniArguments",
    "OmniDataArguments",
    "OmniGraphProfileArguments",
    "OmniInferArguments",
    "OmniModuleRuntimeArguments",
    "OmniModelRuntimeArguments",
    "OmniTrainingArguments",
    "OMNI_TRAIN_WORKFLOWS",
    "DEFAULT_SCENARIO",
    "build_module_args",
    "build_module_runtime_args",
    "build_omni_model_runtime",
    "load_yaml_with_inherit",
    "parse_args",
    "parse_omni_args",
    "resolve_omni_model",
    "save_args",
]
