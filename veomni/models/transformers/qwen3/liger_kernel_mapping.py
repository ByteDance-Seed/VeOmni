from ...loader import LIGER_KERNEL_MAPPING_REGISTRY


KERNEL_MAPPING = {
    "RMSNorm": {
        "type": "layer",
        "module": "veomni.models.transformers.qwen3.liger_kernels_hub",
        "name": "LigerRMSNorm",
    },
    "rotary_pos_emb": {
        "type": "func",
        "module": "veomni.models.transformers.qwen3.liger_kernels_hub",
        "name": "liger_rotary_pos_emb",
    },
}


@LIGER_KERNEL_MAPPING_REGISTRY.register("qwen3")
def get_liger_kernel_mapping():
    return KERNEL_MAPPING
