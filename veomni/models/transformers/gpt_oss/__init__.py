import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)

from .configuration_gpt_oss import GptOssConfig
from .modeling_gpt_oss import (
    GptOssForCausalLM,
    GptOssModel,
)


if transformers.__version__ < "4.55.0":
    AutoConfig.register("gpt_oss", GptOssConfig)
    AutoModel.register(GptOssConfig, GptOssModel)
    AutoModelForCausalLM.register(GptOssConfig, GptOssForCausalLM)

__all__ = ["GptOssConfig", "GptOssModel", "GptOssForCausalLM"]
