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

"""Local validation for the optional NPU expert weight-gradient precision."""

import torch


def validate_expert_precision_config(mixed, fsdp_mode, low_precision_comm, tp_size, pp_size, compile_enabled):
    """Reject locally unsupported precision and parallelization combinations."""
    if not mixed.npu_expert_fp32_wgrad:
        return
    if (
        not mixed.enable
        or mixed.param_dtype != "bfloat16"
        or mixed.reduce_dtype != "float32"
        or fsdp_mode != "fsdp2"
        or low_precision_comm
        or tp_size != 1
        or pp_size != 1
        or compile_enabled
    ):
        raise ValueError(
            "NPU expert FP32 wgrad requires FSDP2, BF16 compute, FP32 reduction, native communication, "
            "TP=PP=1, fused_npu MoE, and torch.compile disabled"
        )


def validate_qwen4_experts(model):
    """Check the local constructed expert modules before any FSDP wrapping."""
    from ..lora.moe_layers import has_lora_moe_experts

    if has_lora_moe_experts(model):
        raise ValueError("NPU expert FP32 wgrad does not support MoE LoRA expert wrappers")
    experts = [
        (name, module) for name, module in model.named_modules() if type(module).__name__ == "Qwen4ExpTextExperts"
    ]
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen4_exp" or not experts:
        raise ValueError("NPU expert FP32 wgrad currently requires Qwen4-Exp expert modules")
    if any(
        getattr(getattr(module, "veomni_moe", None), "impl", None) != "fused_npu"
        or not getattr(module, "use_swiglu_mlp", False)
        for _, module in experts
    ):
        raise ValueError("NPU expert FP32 wgrad requires Qwen4-Exp SiLU fused_npu experts")
    if any(param.dtype != torch.float32 for _, module in experts for param in module.parameters()):
        raise ValueError("NPU expert FP32 wgrad requires FP32 expert master parameters")
    return {id(module) for _, module in experts}
