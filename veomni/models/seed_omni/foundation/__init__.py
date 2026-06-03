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

import transformers

from . import (
    bagel_foundation,
    janus_foundation,
    llama_foundation,
    qwen2_5_omni_foundation,
    qwen2_5_vl_foundation,
    qwen2_vl_foundation,
    qwen3_moe_foundation,
)
from .base import BaseFoundationConfigMixin, BaseFoundationModelMixin


__all__ = [
    "qwen2_vl_foundation",
    "janus_foundation",
    "llama_foundation",
    "qwen2_5_vl_foundation",
    "qwen2_5_omni_foundation",
    "BaseFoundationModelMixin",
    "BaseFoundationConfigMixin",
    "qwen3_moe_foundation",
    "bagel_foundation",
]

if transformers.__version__ >= "4.56.0":
    from . import seed_oss_foundation

    __all__ += ["seed_oss_foundation"]
