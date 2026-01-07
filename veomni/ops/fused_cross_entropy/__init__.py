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

import os
from typing import Optional

import torch
import torch.nn as nn
from transformers.loss.loss_utils import LOSS_MAPPING

from ...distributed.parallel_state import get_parallel_state
from ...distributed.sequence_parallel import reduce_sequence_parallel_loss
from ...utils import logging
from ...utils.env import get_env
from ...utils.import_utils import is_liger_kernel_available, is_torch_npu_available
from .eager import eager_cross_entropy
from .loss import ForCausalLMLoss, ForSequenceClassificationLoss


logger = logging.get_logger(__name__)


_cross_entropy = None


def apply_veomni_loss_patch():
    LOSS_MAPPING["ForCausalLM"] = ForCausalLMLoss
    LOSS_MAPPING["ForConditionalGeneration"] = ForCausalLMLoss
    LOSS_MAPPING["ForSequenceClassification"] = ForSequenceClassificationLoss
    global _cross_entropy
    if is_torch_npu_available():
        _cross_entropy = eager_cross_entropy
    elif is_liger_kernel_available() and get_env("USE_LIGER_KERNEL") == "1":
        from .liger_kernel import fused_liger_kernel_cross_entropy

        _cross_entropy = fused_liger_kernel_cross_entropy
    else:
        _cross_entropy = eager_cross_entropy
