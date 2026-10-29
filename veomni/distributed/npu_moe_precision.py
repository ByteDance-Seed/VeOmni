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

"""Job-wide agreement for the optional NPU expert weight-gradient precision."""

import torch.distributed as dist


def expert_precision_fingerprint(mixed, fsdp_mode, low_precision_comm, tp_size, pp_size, compile_enabled, moe_impl):
    return (
        mixed.npu_expert_fp32_wgrad,
        mixed.enable,
        mixed.param_dtype,
        mixed.reduce_dtype,
        mixed.output_dtype,
        mixed.cast_forward_inputs,
        fsdp_mode,
        low_precision_comm,
        tp_size,
        pp_size,
        compile_enabled,
        moe_impl,
    )


def validate_expert_precision_consensus(accelerator, moe_implementation):
    """Agree before mesh creation; every NPU training rank must call this even when disabled."""
    fsdp = accelerator.fsdp_config
    mixed = fsdp.mixed_precision
    local = expert_precision_fingerprint(
        mixed,
        fsdp.fsdp_mode,
        fsdp.low_precision_reduce_scatter_comm,
        accelerator.tp_size,
        accelerator.pp_size,
        accelerator.torch_compile.enable,
        moe_implementation or "fused_npu",
    )
    peers = [None] * dist.get_world_size()
    dist.all_gather_object(peers, local)
    if not any(peer[0] for peer in peers):
        return
    if any(peer != local for peer in peers):
        raise ValueError("NPU expert FP32 wgrad requires identical precision configuration on all ranks")
    if (
        not mixed.enable
        or mixed.param_dtype != "bfloat16"
        or mixed.reduce_dtype != "float32"
        or fsdp.fsdp_mode != "fsdp2"
        or fsdp.low_precision_reduce_scatter_comm
        or accelerator.tp_size != 1
        or accelerator.pp_size != 1
        or accelerator.torch_compile.enable
        or local[-1] != "fused_npu"
    ):
        raise ValueError(
            "NPU expert FP32 wgrad requires FSDP2, BF16 compute, FP32 reduction, native communication, "
            "TP=PP=1, fused_npu MoE, and torch.compile disabled"
        )
    mixed._npu_expert_fp32_wgrad_fingerprint = local


def require_expert_precision_consensus(mixed, fsdp_mode, low_precision_comm, tp_size, pp_size, compile_enabled):
    """Refuse unsupported callers or a config mutated after the job-wide check."""
    if not mixed.npu_expert_fp32_wgrad:
        return
    original = getattr(mixed, "_npu_expert_fp32_wgrad_fingerprint", None)
    current = expert_precision_fingerprint(
        mixed, fsdp_mode, low_precision_comm, tp_size, pp_size, compile_enabled, "fused_npu"
    )
    if original != current:
        raise ValueError("NPU expert FP32 wgrad requires the unchanged job-wide precision consensus")


def validate_qwen4_experts(model):
    """Check the constructed expert modules on every rank before any FSDP wrapping."""
    experts = [
        (name, module) for name, module in model.named_modules() if type(module).__name__ == "Qwen4ExpTextExperts"
    ]
    report = (
        getattr(getattr(model, "config", None), "model_type", None),
        tuple(
            (
                name,
                getattr(getattr(module, "veomni_moe", None), "impl", None),
                getattr(module, "use_swiglu_mlp", False),
                tuple(
                    (param_name, str(param.dtype), tuple(param.shape))
                    for param_name, param in module.named_parameters()
                ),
            )
            for name, module in experts
        ),
    )
    peers = [None] * dist.get_world_size()
    dist.all_gather_object(peers, report)
    if any(peer != report for peer in peers):
        raise ValueError("NPU expert FP32 wgrad requires identical Qwen4-Exp expert modules on all ranks")
    if report[0] != "qwen4_exp" or not experts:
        raise ValueError("NPU expert FP32 wgrad currently requires Qwen4-Exp expert modules")
    if any(impl != "fused_npu" or not swiglu for _, impl, swiglu, _ in report[1]):
        raise ValueError("NPU expert FP32 wgrad requires Qwen4-Exp SiLU fused_npu experts")
    if any(dtype != "torch.float32" for _, _, _, params in report[1] for _, dtype, _ in params):
        raise ValueError("NPU expert FP32 wgrad requires FP32 expert master parameters")
    return {id(module) for _, module in experts}
