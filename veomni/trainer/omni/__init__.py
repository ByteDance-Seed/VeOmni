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

"""SeedOmni V2 trainer / inferencer package.

Split into four modules:

* :mod:`~veomni.trainer.omni.omni_module_trainer` — per-module training unit
  (:class:`OmniModuleTrainer`) + its per-module checkpoint callbacks.
* :mod:`~veomni.trainer.omni.omni_trainer` — orchestrator (:class:`OmniTrainer`)
  + multi-optimizer / multi-scheduler proxies + the global-state callback.
* :mod:`~veomni.trainer.omni.omni_module_inferencer` — per-module inference
  builder (:class:`OmniModuleInferencer`).
* :mod:`~veomni.trainer.omni.omni_inferencer` — inference driver
  (:class:`OmniInferencer`) + :class:`InferenceRequest`.
"""

from .omni_inferencer import InferenceRequest, OmniInferencer
from .omni_module_inferencer import OmniModuleInferencer
from .omni_module_trainer import OmniModuleTrainer
from .omni_trainer import OmniTrainer


__all__ = [
    "OmniTrainer",
    "OmniModuleTrainer",
    "OmniInferencer",
    "OmniModuleInferencer",
    "InferenceRequest",
]
