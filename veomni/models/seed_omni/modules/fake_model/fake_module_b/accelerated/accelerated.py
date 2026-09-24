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

"""VeOmni-side FakeModuleB — the tail of the stand-in chain, where the loss is taken."""

from typing import Any

from .....mixins.base_mixin import BaseMixin
from .....mixins.inference_module_mixin import InferenceModuleMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....modeling_omni import LOSS_KEY
from ..modeling import FakeModuleB


class TrainingMixin(TrainingModuleMixin):
    """Take ``hidden`` off the batch and emit the chain's loss.

    A squared-magnitude loss has a nonzero gradient for any nonzero ``hidden``,
    so both modules' weights receive one on every step.
    """

    @pre_forward("forward")
    def forward_pre(self, hidden: Any, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"hidden": hidden}

    @post_forward("forward")
    def forward_post(self, hidden: Any = None, **outputs: Any) -> dict[str, Any]:
        if hidden is not None:
            outputs["hidden"] = hidden
            outputs[LOSS_KEY] = hidden.float().pow(2).mean()
        return outputs


class VeOmniMixin(BaseMixin, TrainingMixin, InferenceModuleMixin):
    pass


class FakeModuleBAccelerated(VeOmniMixin, FakeModuleB):
    pass


__all__ = ["FakeModuleBAccelerated"]
