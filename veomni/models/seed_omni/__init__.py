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


from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

from .auto import SeedOmniConfig, SeedOmniModel, SeedOmniProcessor, build_omni_model, build_omni_processor
from .decoder import movqgan
from .encoder import qwen2_vl_vision_model
from .foundation import qwen2_vl_foundation


AutoConfig.register("seed_omni", SeedOmniConfig)
AutoModelForVision2Seq.register(SeedOmniConfig, SeedOmniModel)
AutoProcessor.register(SeedOmniConfig, SeedOmniProcessor)


__all__ = [
    "build_omni_model",
    "build_omni_processor",
    "SeedOmniModel",
    "SeedOmniConfig",
    "SeedOmniProcessor",
    "qwen2_vl_vision_model",
    "movqgan",
    "qwen2_vl_foundation",
]
