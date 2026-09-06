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

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils import logging
from .config.singleton import set_ops_config
from .dispatch import OpSlot


if TYPE_CHECKING:
    from ..arguments.arguments_types import OpsImplementationConfig

__all__ = ["OpSlot"]

logger = logging.get_logger(__name__)


def apply_ops_patch():
    """Leftover import-time hook. Attention now installs via ``apply_kernel_patch``."""
    from ..kernels import apply_kernel_patch

    apply_kernel_patch()


def apply_ops_config(ops_config: OpsImplementationConfig) -> None:
    """Populate the legacy ops-config singleton for old model integrations.

    Tensor-native kernels, including cross-entropy, are selected by
    ``models_kernel`` through instance-local ``VeomniKernel`` handles. This
    function intentionally does not mutate Transformers' process-global
    ``LOSS_MAPPING``.
    """
    set_ops_config(ops_config)
    logger.info_rank0("✅ VeOmni legacy ops config applied.")
