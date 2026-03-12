from ...loader import LIGER_KERNEL_MAPPING_REGISTRY


KERNEL_MAPPING = {
    "RMSNorm": "liger_kernel.transformers.rms_norm.LigerRMSNorm",
    # "LayerNorm": "liger_kernel.transformers.layer_norm.LigerLayerNorm",
    # "SwiGLUMLP": "liger_kernel.transformers.swiglu.LigerSwiGLUMLP",
    # "MultimodalRoPE": "liger_kernel.transformers.qwen2vl_mrope.liger_multimodal_rotary_pos_emb",
}


@LIGER_KERNEL_MAPPING_REGISTRY.register("qwen2_vl")
def get_liger_kernel_mapping():
    return KERNEL_MAPPING
