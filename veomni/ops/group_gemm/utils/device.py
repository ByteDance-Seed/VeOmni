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

from functools import lru_cache

from ....utils.device import get_device_name


@lru_cache
def get_device_key() -> str:
    import torch

    if torch.cuda.get_device_capability() == (8, 0):
        return "A100"  # A30 is treated the same way as A100 for the moment.

    if torch.cuda.get_device_capability() == (9, 0):
        return "H100"

    name = get_device_name()
    if name.startswith("NVIDIA "):
        name = name[len("NVIDIA ") :]

    return name
