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

"""SeedOmni V2 launcher arguments — isolated from V1 ``veomni.arguments``."""

from .arguments_types import (
    DEFAULT_SCENARIO,
    OMNI_TRAIN_WORKFLOWS,
    BaseOmniModelArguments,
    OmniArguments,
    OmniDataArguments,
    OmniGraphProfileArguments,
    OmniInferArguments,
    OmniModelRuntimeArguments,
    OmniModuleRuntimeArguments,
    OmniTrainingArguments,
    _is_omni_checkpoint_root,
    build_module_args,
    build_module_runtime_args,
    build_omni_model_runtime,
    resolve_omni_model,
)
from .parser import load_yaml_with_inherit, parse_omni_args


__all__ = [
    "DEFAULT_SCENARIO",
    "OMNI_TRAIN_WORKFLOWS",
    "BaseOmniModelArguments",
    "OmniArguments",
    "OmniDataArguments",
    "OmniGraphProfileArguments",
    "OmniInferArguments",
    "OmniModelRuntimeArguments",
    "OmniModuleRuntimeArguments",
    "OmniTrainingArguments",
    "_is_omni_checkpoint_root",
    "build_module_args",
    "build_module_runtime_args",
    "build_omni_model_runtime",
    "load_yaml_with_inherit",
    "parse_omni_args",
    "resolve_omni_model",
]
