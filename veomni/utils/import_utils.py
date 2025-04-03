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


"""Import utils"""

import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING

from packaging import version


if TYPE_CHECKING:
    from packaging.version import Version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> "Version":
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")


def is_flash_attn_2_available() -> bool:
    return _is_package_available("flash_attn")


def is_liger_kernel_available() -> bool:
    return _is_package_available("liger_kernel")


def is_fused_moe_available():
    import torch

    if torch.cuda.is_available() and not _is_package_available("torch_npu") and _is_package_available("triton"):
        return True

    return False


def is_torch_npu_available() -> bool:
    return _is_package_available("torch_npu")


def is_bytecheckpoint_available() -> bool:
    return _is_package_available("bytecheckpoint")


def is_vescale_available() -> bool:
    return _is_package_available("vescale")


@lru_cache
def is_torch_version_greater_than(value: str) -> bool:
    return _get_package_version("torch") >= version.parse(value)
