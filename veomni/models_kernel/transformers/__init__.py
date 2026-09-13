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
# See the License for the specific language governing limitations
# under the License.

"""Transformer modeling that calls local ``VeomniOp`` handles.

Importing this package registers the models listed below on the
``models_kernel`` registries. Add the next model here when it is ready.
"""

from . import deepseek_v4, llama, qwen3, qwen3_5, qwen3_5_moe, qwen3_moe, qwen3_vl, qwen3_vl_moe


__all__ = [
    "deepseek_v4",
    "llama",
    "qwen3",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
]
