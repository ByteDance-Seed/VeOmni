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
from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("llada")
def register_llada_config():
    from .configuration_llada import LLaDAConfig

    return LLaDAConfig


@MODELING_REGISTRY.register("llada")
def register_llada_modeling(architecture: str):
    # LLaDA is vendored rather than patchgen-generated: it is a bidirectional mask
    # predictor with real ``LLaDA*`` class names (seeded from Llama, maintained
    # directly). HuggingFace does not ship this architecture.
    from .modeling_llada import LLaDAForCausalLM, LLaDAModel

    # Unconverted LLaDA-MoE checkpoints also say `model_type: llada`; building the dense
    # model from one would leave every MLP randomly initialized without an error.
    if "MoE" in architecture:
        raise ValueError(
            f"Architecture {architecture!r} is LLaDA-MoE with its upstream `model_type: llada`. Convert it first: "
            "python scripts/convert_lladamoe_to_veomni.py --src <ckpt> --dst <ckpt>-veomni"
        )
    if architecture.endswith("Model"):
        return LLaDAModel
    # LLaDA's upstream architecture is "LLaDAModelLM" — it is used as a
    # causal-LM-style mask predictor with a separate lm_head, so default to
    # ForCausalLM.
    return LLaDAForCausalLM
