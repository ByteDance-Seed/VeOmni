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

"""VeOmni-accelerated ``LladaUniDecoder``."""

from typing import List

from .....mixins.base_mixin import BaseMixin
from .....mixins.metric_meter_mixin import MetricMeterMixin
from .....mixins.training_module_mixin import TrainingModuleMixin
from ..configuration import LladaUniDecoderConfig
from ..modeling import LladaUniDecoder


class TrainingMixin(TrainingModuleMixin):
    config: LladaUniDecoderConfig


class MeterMixin(MetricMeterMixin):
    config: LladaUniDecoderConfig

    def estimate_flops(self, seqlens: List[int]) -> float:
        del seqlens
        return 0.0


class VeOmniMixin(BaseMixin, TrainingMixin, MeterMixin):
    config: LladaUniDecoderConfig


class LladaUniDecoderAccelerated(VeOmniMixin, LladaUniDecoder):
    pass


__all__ = ["LladaUniDecoderAccelerated", "TrainingMixin", "VeOmniMixin"]
