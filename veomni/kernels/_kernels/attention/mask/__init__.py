# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Mask builders. Not registered on ``KERNEL_REGISTRY``."""

from .flash import flash_attention_mask_builder
from .flex import flex_attention_mask_builder
from .magi import MagiAttentionMask, magi_attention_mask_builder
from .sdpa import sdpa_attention_mask_builder
from .shape import causal_mask, packed_causal_mask, sliding_window_mask


__all__ = [
    "MagiAttentionMask",
    "causal_mask",
    "magi_attention_mask_builder",
    "flash_attention_mask_builder",
    "flex_attention_mask_builder",
    "packed_causal_mask",
    "sdpa_attention_mask_builder",
    "sliding_window_mask",
]
