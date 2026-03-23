import transformers.models.qwen2_vl.configuration_qwen2_vl as hf_qwen2vl
from transformers.models.qwen2_vl.configuration_qwen2_vl import PretrainedConfig, Qwen2VLConfig


# https://github.com/huggingface/transformers/pull/41758
def Qwen2VLConfig___getattribute__(self: Qwen2VLConfig, key: str):
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


def apply_veomni_qwen2vl_patch():
    hf_qwen2vl.Qwen2VLConfig.__getattribute__ = Qwen2VLConfig___getattribute__
