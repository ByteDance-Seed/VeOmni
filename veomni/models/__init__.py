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

"""Public model construction, registration, and checkpoint interfaces.

Family packages are imported by later commits. This commit exposes the
shared construction and checkpoint surface.
"""

from .auto import (
    build_config,
    build_foundation_model,
    build_processor,
    build_tokenizer,
    check_context_parallel_supported,
    check_model_build_prerequisites,
)
from .checkpoint import ModelCheckpointManager
from .checkpoint.weights import (
    init_empty_weights,
    load_model_weights,
    load_model_weights_ep_sharded,
    rank0_load_and_broadcast_weights,
    save_model_assets,
    save_model_weights,
)
from .registry import (
    MODEL_CONFIG_REGISTRY,
    MODEL_PROCESSOR_REGISTRY,
    MODELING_REGISTRY,
    get_model_class,
)


__all__ = [
    "MODEL_CONFIG_REGISTRY",
    "MODEL_PROCESSOR_REGISTRY",
    "MODELING_REGISTRY",
    "ModelCheckpointManager",
    "build_config",
    "build_foundation_model",
    "build_processor",
    "build_tokenizer",
    "check_context_parallel_supported",
    "check_model_build_prerequisites",
    "get_model_class",
    "init_empty_weights",
    "load_model_weights",
    "load_model_weights_ep_sharded",
    "rank0_load_and_broadcast_weights",
    "save_model_assets",
    "save_model_weights",
]
