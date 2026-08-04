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

"""VeOmni accelerator layer over :class:`~veomni.models.seed_omni.modeling_omni.OmniModel`.

Lightweight exports only — ``module_runtime`` is a submodule; import it explicitly
(``from veomni.models.seed_omni.accelerator.module_runtime import …``) to avoid
pulling trainer/distributed setup into every ``seed_omni`` import.
"""

from .executor import execute_generation_node, execute_train_node
from .omni_model_runtime import OmniModelRuntime


__all__ = [
    "OmniModelRuntime",
    "execute_train_node",
    "execute_generation_node",
]
