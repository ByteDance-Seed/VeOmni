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
# See the License for the specific language governing limitations
# under the License.

"""Qwen2 modeling that calls local ``VeomniOp`` handles."""

from veomni.models.registry import MODELING_REGISTRY


@MODELING_REGISTRY.register("qwen2")
def register_qwen2_modeling(architecture: str | None):
    from .generated.patched_modeling_qwen2_gpu import (
        Qwen2ForCausalLM,
        Qwen2ForQuestionAnswering,
        Qwen2ForSequenceClassification,
        Qwen2ForTokenClassification,
        Qwen2Model,
    )

    architecture = architecture or ""
    if "ForCausalLM" in architecture:
        return Qwen2ForCausalLM
    if "ForTokenClassification" in architecture:
        return Qwen2ForTokenClassification
    if "ForSequenceClassification" in architecture:
        return Qwen2ForSequenceClassification
    if "ForQuestionAnswering" in architecture:
        return Qwen2ForQuestionAnswering
    if "Model" in architecture:
        return Qwen2Model
    return Qwen2ForCausalLM
