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

"""SeedOmni V2 training / inference orchestrators (independent from V1 ``BaseTrainer``).

Only :class:`OmniTrainer` / :class:`OmniInferencer` are re-exported here — the entry
points (``tasks/omni/train_omni.py`` / ``tasks/omni/infer_omni.py``) are the only real
consumers of this package-level surface. Everything else (``ModuleRuntime``,
``OmniModelRuntime``, the per-module callbacks, ``MultiOptimizer``/``MultiLRScheduler``,
``build_module_runtime_args``, …) is always imported directly from its owning submodule
(``veomni.models.seed_omni.accelerator.*``, ``veomni.omni_arguments.*``,
``veomni.trainer.callbacks.omni_callbacks``, ``veomni.trainer.omni.omni_trainer``) —
import it from there instead of adding it back here.
"""

from .omni_inferencer import OmniInferencer
from .omni_trainer import OmniTrainer


__all__ = [
    "OmniTrainer",
    "OmniInferencer",
]
