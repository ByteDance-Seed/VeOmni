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

from .ops import apply_ops_patch, format_kernel_functions
from .utils.env import format_envs
from .utils.logging import get_logger


logger = get_logger(__name__)


def _apply_patches():
    apply_ops_patch()
    logger.info_rank0(format_envs())
    logger.info_rank0(format_kernel_functions())


_apply_patches()

__version__ = "v0.1.0"
