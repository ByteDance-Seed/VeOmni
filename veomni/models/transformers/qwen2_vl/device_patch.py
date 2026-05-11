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

import transformers.models.qwen2_vl.modeling_qwen2_vl as hf_qwen2_vl

from ....ops.config.registry import BackendSpec, apply_per_model_patches


def _custom_qwen2vl(ops_config, applied):
    # Qwen2-VL's liger RMSNorm backend also patches the vision-tower LayerNorm.
    if ops_config.rms_norm_implementation == "liger_kernel":
        from liger_kernel.transformers.layer_norm import LigerLayerNorm

        hf_qwen2_vl.LayerNorm = LigerLayerNorm


def apply_veomni_qwen2vl_device_patch():
    apply_per_model_patches(
        hf_module=hf_qwen2_vl,
        model_name="Qwen2-VL",
        targets={
            # Multimodal RoPE uses a model-specific symbol and a different
            # liger entry; we override the registry default via extra_backends.
            "rotary_pos_emb": "apply_multimodal_rotary_pos_emb",
            "rms_norm": "Qwen2RMSNorm",
            "swiglu_mlp": "Qwen2MLP",
        },
        extra_backends={
            "rotary_pos_emb": {
                "liger_kernel": BackendSpec(
                    entry="liger_kernel.transformers.qwen2vl_mrope:liger_multimodal_rotary_pos_emb",
                    requires=("liger_kernel",),
                ),
                # Registry default ``apply_rotary_pos_emb_npu`` doesn't fit
                # Qwen2-VL's multimodal RoPE layout. ``None`` = explicitly
                # disabled — raises cleanly instead of binding a wrong-shape
                # kernel. NPU users must pin ``rotary_pos_emb_implementation: eager``.
                "npu": None,  # type: ignore[dict-item]
            },
        },
        custom_patches=_custom_qwen2vl,
    )
