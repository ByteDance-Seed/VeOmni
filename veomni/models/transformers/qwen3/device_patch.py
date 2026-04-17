import transformers.models.qwen3.modeling_qwen3 as hf_qwen3

from ....ops.device_patch_utils import ImplSpec, apply_device_patches, rms_norm_patch, rope_patch, swiglu_patch


PATCHES = [
    rope_patch(
        "apply_rotary_pos_emb",
        {
            "liger_kernel": ImplSpec("liger_kernel.transformers.rope", "liger_rotary_pos_emb"),
            "npu": ImplSpec("veomni.ops.npu_patch.npu_fused_operator", "apply_rotary_pos_emb_npu"),
        },
    ),
    rms_norm_patch(
        "Qwen3RMSNorm",
        {
            "liger_kernel": ImplSpec("liger_kernel.transformers.rms_norm", "LigerRMSNorm"),
            "npu": ImplSpec("veomni.ops.npu_patch.npu_fused_operator", "rms_norm_forward_npu", replace_forward=True),
        },
    ),
    swiglu_patch(
        "Qwen3MLP",
        {
            "liger_kernel": ImplSpec("liger_kernel.transformers.swiglu", "LigerSwiGLUMLP"),
        },
    ),
]


def apply_veomni_qwen3_device_patch():
    apply_device_patches(hf_qwen3, PATCHES, "Qwen3")
