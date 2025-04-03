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

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from .configuration_deepseek import DeepseekV3Config
from .modeling_deepseek import (
    DeepseekV3ForCausalLM,
    DeepseekV3ForSequenceClassification,
    DeepseekV3Model,
)


AutoConfig.register("deepseek_v3", DeepseekV3Config)
AutoModel.register(DeepseekV3Config, DeepseekV3Model)
AutoModelForCausalLM.register(DeepseekV3Config, DeepseekV3ForCausalLM)
AutoModelForSequenceClassification.register(DeepseekV3Config, DeepseekV3ForSequenceClassification)
