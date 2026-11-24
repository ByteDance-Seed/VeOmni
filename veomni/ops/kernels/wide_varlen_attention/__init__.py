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
"""Varlen causal attention for heads too wide for flash (Triton).

Library op, like ``dllm_hole_attention``: DiffusionGemma's encoder calls this
directly on its five ``global_head_dim=512`` layers when the row is packed. No
OpSlot / KERNEL_REGISTRY entry.
"""

from .triton_kernel import wide_varlen_attention, wide_varlen_attention_reference


__all__ = ["wide_varlen_attention", "wide_varlen_attention_reference"]
