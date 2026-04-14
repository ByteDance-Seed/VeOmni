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

from ....ops.ops_config import get_ops_config
from ....utils import logging


logger = logging.get_logger(__name__)


def apply_veomni_seed_oss_gpu_patch():
    ops_config = get_ops_config()
    if ops_config is None:
        return

    applied = []

    if ops_config.rotary_pos_emb_implementation == "liger_kernel":
        from liger_kernel.transformers.rope import liger_rotary_pos_emb

        hf_seed_oss.apply_rotary_pos_emb = liger_rotary_pos_emb
        applied.append("RoPE")

    if ops_config.rms_norm_implementation == "liger_kernel":
        from liger_kernel.transformers.rms_norm import LigerRMSNorm

        hf_seed_oss.SeedOssRMSNorm = LigerRMSNorm
        applied.append("RMSNorm")

    if ops_config.swiglu_mlp_implementation == "liger_kernel":
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        hf_seed_oss.SeedOssMLP = LigerSwiGLUMLP
        applied.append("SwiGLU")

    if applied:
        logger.info_rank0(f"Apply liger kernel to SeedOss: {', '.join(applied)}.")
