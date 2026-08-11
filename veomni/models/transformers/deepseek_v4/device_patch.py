# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from ....ops.config.registry import BackendSpec, apply_per_model_patches


_TRITON_ROPE_BACKEND = BackendSpec(
    entry="veomni.ops.kernels.rotary.triton_deepseek_v4:apply_rotary_pos_emb_triton",
    requires=("triton",),
)


def apply_veomni_deepseek_v4_device_patch(gen_module):
    """Backend selection for the patchgen-generated module.

    Only ``rotary_pos_emb`` is wired here. Replacing the module-level
    ``apply_rotary_pos_emb`` covers every call site at once — Q, MQA KV, the
    inverse rotation on the attention output, the indexer Q, and the three
    compressors all resolve it through the module global.

    Both registry defaults are explicitly disabled. V4 rotates only the trailing
    ``qk_rope_head_dim`` slice with an interleaved cos/sin layout, which neither
    ``liger_rotary_pos_emb`` nor ``npu_rotary_mul`` implements (partial_rotary ->
    NaN), and both expect a ``(q, k, cos, sin)`` pair returning two tensors
    rather than V4's single tensor. ``None`` yields a clean "explicitly disabled"
    error instead of a wrong-signature crash or silent garbage, so the surviving
    choices are ``triton`` and ``eager`` — V4 YAMLs pin
    ``rotary_pos_emb_implementation`` because the registry default
    (``liger_kernel``) is among the disabled ones.

    This holds for the NPU sibling module as well, which reaches the same
    conclusion independently (see ``deepseek_v4_npu_patch_gen_config``). There
    ``triton`` is unavailable, so the pin resolves to ``eager``.
    """
    apply_per_model_patches(
        hf_module=gen_module,
        model_name="DeepSeek-V4",
        targets={"rotary_pos_emb": "apply_rotary_pos_emb"},
        extra_backends={
            "rotary_pos_emb": {
                "liger_kernel": None,
                "npu": None,
                "triton": _TRITON_ROPE_BACKEND,
            },
        },
    )
