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

from ....ops.config.registry import BackendSpec, apply_per_model_patches
from ....utils import logging


logger = logging.get_logger(__name__)


def _custom_wan(ops_config, applied):
    # Wan's Triton RoPE lives in ``veomni.ops.kernels.rotary.triton_wan`` and
    # depends on Triton being importable; fall back silently if unavailable.
    if ops_config.rotary_pos_emb_implementation == "triton":
        try:
            from veomni.ops.kernels.rotary.triton_wan import apply_rotary_emb

            from . import modeling_wan

            modeling_wan.rope_apply = apply_rotary_emb
            applied.append("RoPE (triton)")
        except ImportError:
            logger.warning_rank0("Triton RoPE for Wan requested but not available, using eager.")


def apply_veomni_wan_device_patch():
    """Apply ops patches to Wan model based on ``OpsImplementationConfig``.

    Unlike HF-based models that monkey-patch ``transformers.models.*.modeling_*``
    symbols, Wan defines its own ``RMSNorm`` and ``rope_apply`` at module level.
    We import and replace them in ``modeling_wan`` accordingly.
    """
    from . import modeling_wan

    apply_per_model_patches(
        hf_module=modeling_wan,
        model_name="Wan",
        targets={
            "rms_norm": "RMSNorm",
            "rotary_pos_emb": "rope_apply",
        },
        # Wan has a model-specific NPU RMSNorm wrapper and a model-specific
        # NPU RoPE.  The Triton RoPE is a module-level function replacement
        # handled in ``_custom_wan`` to preserve the import-fallback path.
        extra_backends={
            "rms_norm": {
                "liger_kernel": BackendSpec(
                    entry="liger_kernel.transformers.rms_norm:LigerRMSNorm",
                    requires=("liger_kernel",),
                ),
                "npu": BackendSpec(
                    entry="veomni.models.transformers.wan.npu_patch:rms_norm_forward_npu",
                    requires=("torch_npu",),
                    replace_forward=True,
                ),
            },
            "rotary_pos_emb": {
                # Wan's RoPE is a module-level ``rope_apply(x, **kwargs)`` with
                # a non-standard signature (single tensor + freqs/head_dim
                # kwargs), so the registry-default ``liger_kernel`` backend
                # — which expects ``(q, k, cos, sin)`` — cannot drop in.
                # ``None`` here marks the backend as *explicitly disabled* so
                # ``apply_per_model_patches`` raises with a model-specific
                # "explicitly disabled for Wan" message rather than silently
                # binding the wrong-signature kernel and crashing at runtime.
                # Wan YAMLs pin ``rotary_pos_emb_implementation: eager`` to
                # avoid the raise.
                "liger_kernel": None,
                "npu": BackendSpec(
                    entry="veomni.models.transformers.wan.npu_patch:rope_apply_fused",
                    requires=("torch_npu",),
                ),
            },
        },
        custom_patches=_custom_wan,
    )
