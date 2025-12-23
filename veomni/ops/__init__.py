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

from .attention import flash_attention_forward
from .fused_moe import fused_moe_forward
from .loss import causallm_loss_function


__all__ = [
    "flash_attention_forward",
    "fused_moe_forward",
    "causallm_loss_function",
]


def apply_ops_patch():
    from .attention import apply_veomni_attention_patch

    apply_veomni_attention_patch()
