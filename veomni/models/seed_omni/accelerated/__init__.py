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

"""VeOmni accelerated layer over :class:`~veomni.models.seed_omni.modeling_omni.OmniModel`.

Layout:

* :mod:`.omni_module` — per-module config + runtime
* :mod:`.omni_model` — composite config + runtime
* :mod:`.utils` — dispatch, executors, module save/iter helpers

``OmniModuleRuntimeConfig`` / ``OmniModelRuntimeConfig`` are lightweight
dataclasses. ``omni_module_runtime`` / ``dispatch`` are heavier submodules —
import them explicitly
(``from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import …``)
to avoid pulling trainer/distributed setup into every ``seed_omni`` import.
Per-module checkpoint I/O lives in ``veomni.models.seed_omni.utils.checkpoint``.
"""

from .omni_model.omni_model_config import OmniModelRuntimeConfig
from .omni_model.omni_model_runtime import OmniModelRuntime
from .omni_module.omni_module_config import OmniModuleRuntimeConfig
from .utils.executor import TrainNodeRunner, execute_generation_node


__all__ = [
    "OmniModelRuntime",
    "OmniModelRuntimeConfig",
    "OmniModuleRuntimeConfig",
    "TrainNodeRunner",
    "execute_generation_node",
]
