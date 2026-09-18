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

"""VeOmni-accelerated FakeModuleB — identity training / generation graph hooks."""

from typing import Any

from .....mixins.base_mixin import BaseMixin
from .....mixins.inference_module_mixin import InferenceModuleMixin, post_generate, pre_generate
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ..modeling import FakeModuleB


class TrainingMixin(TrainingModuleMixin):
    """Passthrough ``pre_forward`` / ``post_forward`` for the training DAG."""

    @pre_forward("forward")
    def forward_pre(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    @post_forward("forward")
    def forward_post(self, **outputs: Any) -> dict[str, Any]:
        return outputs


class InferenceMixin(InferenceModuleMixin):
    """Passthrough generation hooks; ``generate`` reuses native ``forward``."""

    @pre_generate("generate")
    def generate_pre(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    @post_generate("generate")
    def generate_post(self, **outputs: Any) -> dict[str, Any]:
        return outputs

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("generation_kwargs", None)
        return self.forward(**kwargs)


class VeOmniMixin(BaseMixin, TrainingMixin, InferenceMixin):
    pass


class FakeModuleBAccelerated(VeOmniMixin, FakeModuleB):
    pass


__all__ = ["FakeModuleBAccelerated"]
