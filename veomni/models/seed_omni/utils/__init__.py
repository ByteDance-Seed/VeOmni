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

"""SeedOmni V2 utilities: conversation carrier + HF→split checkpoint conversion."""

from .conversation import (
    ConversationItem,
    build_conversation,
    collect_desired_values,
    is_dummy,
    iter_desired_items,
    maybe_merge_outputs,
    seal_outputs,
)
from .convert_registry import OMNI_CONVERT_REGISTRY, convert_checkpoint


__all__ = [
    "ConversationItem",
    "build_conversation",
    "is_dummy",
    "maybe_merge_outputs",
    "seal_outputs",
    "iter_desired_items",
    "collect_desired_values",
    "OMNI_CONVERT_REGISTRY",
    "convert_checkpoint",
]
