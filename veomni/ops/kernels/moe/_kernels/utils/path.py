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

import inspect
import os

import triton
from packaging import version

from .device import get_device_key
from .kernel import qualified_name


def _get_relative_dir_of_triton_kernel(kernel) -> str:
    path = os.path.relpath(inspect.getfile(kernel), get_bpex_root())
    return path


def get_bpex_root() -> str:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return path


def get_config_path_prefix_for(kernel) -> str:
    v = version.parse(triton.__version__)
    return (
        f"{get_bpex_root()}/config/{v.major}.{v.minor}/{get_device_key()}/"
        f"{_get_relative_dir_of_triton_kernel(kernel)}/{qualified_name(kernel)}"
    )


def get_config_dedicated_file_for(kernel, algo_key) -> str:
    # The only reason the extension is used is to avoid JSON lint..
    return f"{get_config_path_prefix_for(kernel)}/{algo_key}.bpex"
