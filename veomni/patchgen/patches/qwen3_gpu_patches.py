# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Patch configuration for Qwen3 GPU LigerKernel replacements.

This mirrors the runtime GPU patch in
veomni/models/transformers/qwen3/gpu_patch.py.
"""

from typing import Optional

import torch

from ..patch_spec import PatchConfig, create_patch_from_external


config = PatchConfig(
    source_module="transformers.models.qwen3.modeling_qwen3",
    target_file="patched_modeling_qwen3_gpu.py",
    description="Qwen3 with LigerKernel GPU replacements",
)

config.patches.append(
    create_patch_from_external(
        target="Qwen3RMSNorm",
        source_module="liger_kernel.transformers.rms_norm",
        source_name="LigerRMSNorm",
        description="Use LigerKernel RMSNorm",
    )
)

config.patches.append(
    create_patch_from_external(
        target="Qwen3MLP",
        source_module="liger_kernel.transformers.swiglu",
        source_name="LigerSwiGLUMLP",
        description="Use LigerKernel SwiGLU MLP",
    )
)


@config.replace_function("apply_rotary_pos_emb", description="Use LigerKernel rotary embedding")
def apply_rotary_pos_emb_liger(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

    return liger_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
        position_ids=position_ids,
        unsqueeze_dim=unsqueeze_dim,
    )
