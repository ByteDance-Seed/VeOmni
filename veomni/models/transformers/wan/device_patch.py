# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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

from ....ops.device_patch_utils import ImplSpec, apply_device_patches, rms_norm_patch, rope_patch
from ....utils import logging


logger = logging.get_logger(__name__)

PATCHES = [
    rms_norm_patch(
        "RMSNorm",
        {
            "liger_kernel": ImplSpec("liger_kernel.transformers.rms_norm", "LigerRMSNorm"),
            "npu": ImplSpec("veomni.models.transformers.wan.npu_patch", "rms_norm_forward_npu", replace_forward=True),
        },
    ),
    rope_patch(
        "rope_apply",
        {
            "npu": ImplSpec("veomni.models.transformers.wan.npu_patch", "rope_apply_fused"),
        },
    ),
]


def _custom_wan(ops_config, applied):
    if ops_config.rotary_pos_emb_implementation == "triton":
        try:
            from veomni.ops.dit.rope_wan.rotary import apply_rotary_emb

            from . import modeling_wan

            modeling_wan.rope_apply = apply_rotary_emb
            applied.append("RoPE (triton)")
        except ImportError:
            logger.warning_rank0("Triton RoPE for Wan requested but not available, using eager.")


def apply_veomni_wan_device_patch():
    """Apply ops patches to Wan model based on OpsImplementationConfig.

    Unlike HF-based models that monkey-patch ``transformers.models.*.modeling_*``
    symbols, Wan defines its own ``RMSNorm`` and ``rope_apply`` at module level.
    We import and replace them in ``modeling_wan`` accordingly.
    """
    from . import modeling_wan

    apply_device_patches(modeling_wan, PATCHES, "Wan", custom_patches=_custom_wan)
