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

from ....ops.config.registry import apply_per_model_patches


def apply_veomni_seed_oss_device_patch():
    apply_per_model_patches(
        hf_module=hf_seed_oss,
        model_name="SeedOss",
        targets={
            "rotary_pos_emb": "apply_rotary_pos_emb",
            "rms_norm": "SeedOssRMSNorm",
            "swiglu_mlp": "SeedOssMLP",
        },
    )
