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

"""Model boundary for opting native HF configs into VeOmni without global patches."""

from transformers import PreTrainedModel

from .configuration_base import VeOmniPretrainedConfig, as_veomni_config, get_veomni_config_class


class VeOmniPreTrainedModel(PreTrainedModel):
    """Adapt model-owned configs, retaining HF's model initialization behavior."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.config_class is not None:
            cls.config_class = get_veomni_config_class(cls.config_class)

    def __init__(self, config, *args, **kwargs):
        super().__init__(
            as_veomni_config(config, validate=not isinstance(config, VeOmniPretrainedConfig)), *args, **kwargs
        )
