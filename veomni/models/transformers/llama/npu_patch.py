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

import transformers.models.llama.modeling_llama as hf_llama

from ....ops.npu_patch import npu_fused_operator
from ....utils import logging
from ....utils.import_utils import is_liger_kernel_available


logger = logging.get_logger(__name__)


def apply_llama_npu_patch():
    # Patches for Llama Model
    hf_llama.apply_rotary_pos_emb = npu_fused_operator.apply_rotary_pos_emb_npu
    hf_llama.LlamaRMSNorm.forward = npu_fused_operator.rms_norm_forward_npu

    # ================================================================
    # PATCH: apply_rotary_pos_emb, LlamaRMSNorm, LlamaMLP
    # Patch with Liger Kernel
    # ================================================================
    if is_liger_kernel_available():
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.rope import liger_rotary_pos_emb
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        hf_llama.apply_rotary_pos_emb = liger_rotary_pos_emb
        hf_llama.LlamaRMSNorm = LigerRMSNorm
        hf_llama.LlamaMLP = LigerSwiGLUMLP

        logger.info_rank0("Apply liger kernel to Llama.")
