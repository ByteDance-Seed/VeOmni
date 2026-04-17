import transformers.models.qwen2_vl.modeling_qwen2_vl as hf_qwen2_vl

from ....ops.device_patch_utils import ImplSpec, OpsPatch, apply_device_patches, rms_norm_patch, swiglu_patch


PATCHES = [
    OpsPatch(
        "rotary_pos_emb_implementation",
        "M-RoPE",
        "apply_multimodal_rotary_pos_emb",
        {
            "liger_kernel": ImplSpec("liger_kernel.transformers.qwen2vl_mrope", "liger_multimodal_rotary_pos_emb"),
        },
    ),
    rms_norm_patch(
        "Qwen2RMSNorm",
        {
            "liger_kernel": ImplSpec("liger_kernel.transformers.rms_norm", "LigerRMSNorm"),
        },
    ),
    swiglu_patch(
        "Qwen2MLP",
        {
            "liger_kernel": ImplSpec("liger_kernel.transformers.swiglu", "LigerSwiGLUMLP"),
        },
    ),
]


def _custom_qwen2vl(ops_config, applied):
    if ops_config.rms_norm_implementation == "liger_kernel":
        from liger_kernel.transformers.layer_norm import LigerLayerNorm

        hf_qwen2_vl.LayerNorm = LigerLayerNorm


def apply_veomni_qwen2vl_device_patch():
    apply_device_patches(hf_qwen2_vl, PATCHES, "Qwen2-VL", custom_patches=_custom_qwen2vl)
