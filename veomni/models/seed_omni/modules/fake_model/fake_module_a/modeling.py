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

from typing import Any

import torch.nn as nn

from ...module_modeling_base import PretrainedOmniModule
from .configuration import FakeModuleAConfig


class FakeModuleA(PretrainedOmniModule):
    """Identity-sized linear map: ``hidden → hidden``. No preprocessor / conversation."""

    config_class = FakeModuleAConfig
    base_model_prefix = "fake_module_a"
    supports_gradient_checkpointing = False
    _no_split_modules = ["FakeModuleA"]
    # No attention here at all, so every implementation is equally vacuous. The
    # flag is what opens HF's `_sdpa_can_dispatch` gate, without which the omni
    # load path cannot be exercised end to end with anything but `eager` --
    # `attn_implementation` would raise before reaching this module's __init__.
    _supports_sdpa = True

    def __init__(self, config: FakeModuleAConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.ones_(module.weight)

    def forward(self, hidden=None, **kwargs: Any) -> dict[str, Any]:
        outputs = dict(kwargs)
        if hidden is not None:
            outputs["hidden"] = self.proj(hidden)
        return outputs
