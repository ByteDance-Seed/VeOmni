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

"""CLI argument parsing — V1 ``VeOmniArguments`` and V2 ``OmniArguments``.

Omni types are lazy: a V1 ``from veomni.arguments import parse_args`` must not
load ``seed_omni``. The runtime config classes live under
``veomni.models.seed_omni.accelerated``.
"""

from __future__ import annotations

from importlib import import_module

from .arguments_types import (
    AcceleratorConfig,
    BaseModelArguments,
    CheckpointConfig,
    DataArguments,
    DataloaderConfig,
    FSDPConfig,
    GradientCheckpointingConfig,
    InferArguments,
    MixedPrecisionConfig,
    ModelArguments,
    OffloadConfig,
    OpsImplementationConfig,
    OptimizerConfig,
    ProfileConfig,
    TorchCompileConfig,
    TrainingArguments,
    VeOmniArguments,
    WandbConfig,
)
from .parser import parse_args, save_args


_OMNI_LAZY_ATTRS = {
    "DEFAULT_SCENARIO": ".omni_arguments_types",
    "OMNI_TRAIN_WORKFLOWS": ".omni_arguments_types",
    "OmniArguments": ".omni_arguments_types",
    "OmniDataArguments": ".omni_arguments_types",
    "OmniGraphProfileArguments": ".omni_arguments_types",
    "OmniInferArguments": ".omni_arguments_types",
    "OmniModelRuntimeArguments": ".omni_arguments_types",
    "OmniModelRuntimeConfig": ".omni_arguments_types",
    "OmniModuleRuntimeArguments": ".omni_arguments_types",
    "OmniModuleRuntimeConfig": ".omni_arguments_types",
    "OmniTrainingArguments": ".omni_arguments_types",
    "build_module_args": ".omni_arguments_types",
    "build_module_runtime_args": ".omni_arguments_types",
    "build_omni_model_runtime": ".omni_arguments_types",
    "resolve_omni_model": ".omni_arguments_types",
    "load_yaml_with_inherit": ".omni_parser",
    "parse_omni_args": ".omni_parser",
}


def __getattr__(name: str):
    module_name = _OMNI_LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, __package__), name)


__all__ = [
    "AcceleratorConfig",
    "BaseModelArguments",
    "CheckpointConfig",
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
    "OmniModuleRuntimeConfig",
    "OmniModelRuntimeArguments",
    "OmniModelRuntimeConfig",
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
