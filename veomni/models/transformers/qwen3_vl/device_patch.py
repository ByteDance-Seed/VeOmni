# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

import transformers.models.qwen3_vl.modeling_qwen3_vl as hf_qwen3vl

from ....ops.config.registry import apply_per_model_patches


def _custom_qwen3vl(ops_config, applied):
    # Qwen3-VL additionally patches the vision-tower RoPE when running on NPU.
    if ops_config.rotary_pos_emb_implementation == "npu":
        from veomni.ops.kernels.rotary.npu import apply_rotary_pos_emb_vision_npu

        hf_qwen3vl.apply_rotary_pos_emb_vision = apply_rotary_pos_emb_vision_npu


def apply_veomni_qwen3vl_device_patch():
    apply_per_model_patches(
        hf_module=hf_qwen3vl,
        model_name="Qwen3-VL",
        targets={
            "rotary_pos_emb": "apply_rotary_pos_emb",
            "rms_norm": "Qwen3VLTextRMSNorm",
        },
        # Qwen3-VL's RMSNorm/RoPE follow standard HF shapes (T5-form
        # ``weight * x / rms``; ``apply_rotary_pos_emb(q, k, cos, sin, ...)``),
        # so the registry's default Liger backends drop in cleanly — no
        # per-model overrides needed.
        custom_patches=_custom_qwen3vl,
    )
