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

"""VeOmni-native LoRA stack (PEFT-free, PEFT-format compatible).

Public API:
    * :class:`VeOmniLoraConfig` — LoRA config, reads/writes PEFT ``adapter_config.json``.
    * :class:`VeOmniLoraModel`  — PEFT-free ``PeftModel`` replacement.
    * :class:`LoraLinear`       — dense ``nn.Linear`` LoRA layer.
"""

from .config import VeOmniLoraConfig
from .layers import DEFAULT_ADAPTER, LoraLinear, is_lora_linear
from .model import LoraModel, VeOmniLoraModel
from .target_mapping import convert_fused_moe_lora_targets, resolve_fused_moe_lora_targets


def is_veomni_lora_model(model) -> bool:
    """True iff ``model`` is a VeOmni-native LoRA wrapper (not a PEFT model)."""
    return isinstance(model, VeOmniLoraModel)


__all__ = [
    "DEFAULT_ADAPTER",
    "LoraLinear",
    "LoraModel",
    "VeOmniLoraConfig",
    "VeOmniLoraModel",
    "convert_fused_moe_lora_targets",
    "is_lora_linear",
    "is_veomni_lora_model",
    "resolve_fused_moe_lora_targets",
]
