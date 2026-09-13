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

"""Per-module CPU preprocessing contracts and model asset binding."""

from .base import ModulePreprocessorBase
from .binding import MODULE_ASSET_ATTRS, bind_module_assets


__all__ = [
    "MODULE_ASSET_ATTRS",
    "ModulePreprocessorBase",
    "bind_module_assets",
]
