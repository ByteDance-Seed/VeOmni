import transformers.models.qwen2_5_vl.configuration_qwen2_5_vl as hf_qwen25vl
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import PretrainedConfig, Qwen2_5_VLConfig


# https://github.com/huggingface/transformers/pull/41758
def Qwen2_5_VLConfig___getattribute__(self: Qwen2_5_VLConfig, key: str):
    if "text_config" in PretrainedConfig.__getattribute__(self, "__dict__") and key not in [
        "dtype",
        "_attn_implementation_internal",
        "_name_or_path",
        "model_type",
    ]:
        text_config = PretrainedConfig.__getattribute__(self, "text_config")
        if key in text_config.__dict__:
            return getattr(text_config, key)

    return PretrainedConfig.__getattribute__(self, key)


def apply_veomni_qwen25_vl_patch():
    hf_qwen25vl.Qwen2_5_VLConfig.__getattribute__ = Qwen2_5_VLConfig___getattribute__
