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

"""Infrastructure for kernel selection.

Modules:
- ``singleton``: stores the resolved ``OpsImplementationConfig``.
- ``registry``: declarative op/backend registry + dispatch engine.
"""

from .registry import (
    BackendSpec,
    OpScope,
    OpSpec,
    apply_per_model_patches,
    get_op,
    list_ops,
    register_op,
)
from .singleton import get_ops_config, set_ops_config


__all__ = [
    "BackendSpec",
    "OpScope",
    "OpSpec",
    "apply_per_model_patches",
    "get_op",
    "get_ops_config",
    "list_ops",
    "register_op",
    "set_ops_config",
]
