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

from typing import TYPE_CHECKING, Any

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
from .parser import parse_args, save_args


if TYPE_CHECKING:
    from ..omni_arguments import (
        OMNI_TRAIN_WORKFLOWS,
        OmniArguments,
        OmniDataArguments,
        OmniGraphProfileArguments,
        OmniInferArguments,
        OmniModelRuntimeArguments,
        OmniModuleRuntimeArguments,
        OmniTrainingArguments,
        parse_omni_args,
    )

_OMNI_EXPORTS = frozenset(
    {
        "OMNI_TRAIN_WORKFLOWS",
        "OmniArguments",
        "OmniDataArguments",
        "OmniGraphProfileArguments",
        "OmniInferArguments",
        "OmniModelRuntimeArguments",
        "OmniModuleRuntimeArguments",
        "OmniTrainingArguments",
        "parse_omni_args",
    }
)


def __getattr__(name: str) -> Any:
    """Re-export the V2 argument types lazily.

    ``veomni.omni_arguments`` builds its dataclasses on top of this package's V1
    config objects, so importing it eagerly here would break whichever of the two
    packages a process happens to import first.
    """
    if name in _OMNI_EXPORTS:
        from .. import omni_arguments

        return getattr(omni_arguments, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AcceleratorConfig",
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
    "parse_args",
    "parse_omni_args",
    "save_args",
]
