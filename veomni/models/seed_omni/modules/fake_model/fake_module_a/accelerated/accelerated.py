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

"""VeOmni-side FakeModuleA — the head of the stand-in chain."""

from typing import Any

import torch

from .....mixins.base_mixin import BaseMixin
from .....mixins.inference_module_mixin import InferenceModuleMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, pre_forward
from ..modeling import FakeModuleA


class TrainingMixin(TrainingModuleMixin):
    """Turn a collated batch into the ``hidden`` the chain carries.

    The module has no preprocessor, so the batch reaches it as the collator
    left it: one ``conversation_list`` per sample. One ``hidden`` row per sample
    is all the chain needs to produce a loss that backpropagates through both
    modules.
    """

    @pre_forward("forward")
    def forward_pre(self, conversation_list: list[Any], hidden: Any = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        if hidden is None:
            weight = self.proj.weight
            hidden = torch.ones(
                len(conversation_list), self.config.hidden_size, device=weight.device, dtype=weight.dtype
            )
        return {"hidden": hidden}


class VeOmniMixin(BaseMixin, TrainingMixin, InferenceModuleMixin):
    pass


class FakeModuleAAccelerated(VeOmniMixin, FakeModuleA):
    pass


__all__ = ["FakeModuleAAccelerated"]
