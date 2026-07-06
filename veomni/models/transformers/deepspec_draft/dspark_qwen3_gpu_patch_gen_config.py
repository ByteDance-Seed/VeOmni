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
"""Patch config for DeepSpec Qwen3 DSpark draft modeling on GPU.

The DeepSpec draft model imports Qwen3RMSNorm and Qwen3MLP from HuggingFace's
Qwen3 modeling module, then composes them in its own DSpark attention/layer
classes. This patchgen target keeps DeepSpec's model structure unchanged while
wrapping those imported classes with VeOmni OpSlot guards for Liger RMSNorm and
SwiGLU kernels.

RoPE is deliberately left eager here: DSpark applies RoPE to a short draft
query block and a longer context+draft key block, while the existing Liger RoPE
slot is tested for equal q/k sequence lengths.
"""

from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="deepspec.modeling.dspark.qwen3.modeling",
    target_file="patched_modeling_dspark_qwen3_gpu.py",
    description="DeepSpec Qwen3 DSpark draft model with VeOmni OpSlot guards",
)

config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # These are bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_rms_norm = OpSlot("rms_norm", "standard")
    veomni_swiglu_mlp = OpSlot("swiglu_mlp", "standard")

    _DeepSpecQwen3RMSNorm = Qwen3RMSNorm
    _DeepSpecQwen3MLP = Qwen3MLP

    class Qwen3RMSNorm(_DeepSpecQwen3RMSNorm):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            if veomni_rms_norm.use_non_eager_impl:
                return veomni_rms_norm(hidden_states, self.weight, self.variance_epsilon)
            return super().forward(hidden_states)

    class Qwen3MLP(_DeepSpecQwen3MLP):
        def forward(self, x):
            if veomni_swiglu_mlp.use_non_eager_impl:
                return veomni_swiglu_mlp(self, x)
            return super().forward(x)
    """
)
