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
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("gpt_oss")
def register_gpt_oss_modeling(architecture: str):
    architecture = architecture or "GptOssForCausalLM"

    try:
        import transformers.models.gpt_oss
    except ImportError as e:
        raise RuntimeError(
            "GPT-OSS support requires a Transformers build that provides `transformers.models.gpt_oss`."
        ) from e
    from .generated.patched_modeling_gpt_oss_gpu import (
        GptOssForCausalLM,
        GptOssForSequenceClassification,
        GptOssForTokenClassification,
        GptOssModel,
    )

    if "ForSequenceClassification" in architecture:
        return GptOssForSequenceClassification
    elif "ForTokenClassification" in architecture:
        return GptOssForTokenClassification
    elif "Model" in architecture:
        return GptOssModel
    else:
        return GptOssForCausalLM
