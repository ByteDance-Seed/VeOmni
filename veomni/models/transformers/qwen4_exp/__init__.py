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

from ....utils.device import IS_NPU_AVAILABLE
from ...loader import MODELING_REGISTRY
from .checkpoint_tensor_converter import (
    convert_qwen4_exp_fqn_to_index_mapping,
    create_qwen4_exp_checkpoint_tensor_converter,
)


@MODELING_REGISTRY.register("qwen4_exp")
def register_qwen4_exp_modeling(architecture: str):
    """Register the Qwen4-Exp VLM SFT implementation for the active device."""
    if architecture != "Qwen4ExpForConditionalGeneration":
        raise NotImplementedError(
            "The initial Qwen4-Exp integration supports only Qwen4ExpForConditionalGeneration for VLM SFT."
        )

    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen4_exp_npu import Qwen4ExpForConditionalGeneration
    else:
        from .generated.patched_modeling_qwen4_exp_gpu import Qwen4ExpForConditionalGeneration

    Qwen4ExpForConditionalGeneration._create_checkpoint_tensor_converter = staticmethod(
        create_qwen4_exp_checkpoint_tensor_converter
    )
    Qwen4ExpForConditionalGeneration._convert_fqn_to_index_mapping = staticmethod(
        convert_qwen4_exp_fqn_to_index_mapping
    )
    return Qwen4ExpForConditionalGeneration


@MODELING_REGISTRY.register("qwen4_exp_text")
def register_qwen4_exp_text_modeling(architecture: str):
    """Reject the standalone text model, which is outside the initial VLM SFT scope."""
    raise NotImplementedError(
        "The initial Qwen4-Exp integration does not support standalone text architectures; use VLM SFT instead."
    )
