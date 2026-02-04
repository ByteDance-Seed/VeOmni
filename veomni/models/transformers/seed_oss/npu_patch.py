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

import transformers.models.seed_oss.modeling_seed_oss as hf_seed_oss

from ....ops.npu_patch import npu_fused_operator
from ....utils import logging


logger = logging.get_logger(__name__)


def apply_seed_oss_npu_patch():
    # Patches for SeedOss Model
    hf_seed_oss.apply_rotary_pos_emb = npu_fused_operator.apply_rotary_pos_emb_npu
    hf_seed_oss.SeedOssRMSNorm.forward = npu_fused_operator.rms_norm_forward_npu
