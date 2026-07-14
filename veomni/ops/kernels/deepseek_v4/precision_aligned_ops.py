# Copyright 2026 the Miles contributors and ByteDance Ltd. and/or its affiliates
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
#
# Adapted from radixark/miles; modified for VeOmni.

import torch


class _BFloat16LinearFP32Func(torch.autograd.Function):
    # Forward matches SGLang's default DeepSeek-V4 compressor path
    # (`sglang.jit_kernel.deepseek_v4.linear_bf16_fp32`, cublas backend):
    # BF16 activation x BF16 weight -> FP32 output. This keeps Megatron's
    # compressor log-prob computation aligned with SGLang rollout. Backward is
    # only needed for training, so keep its gradient matmuls in FP32.
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(torch.bfloat16)
        weight_bf16 = weight.to(torch.bfloat16)
        ctx.save_for_backward(x_bf16, weight_bf16)
        ctx.input_shape = x.shape
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype

        x_2d = x_bf16.reshape(-1, x_bf16.shape[-1])
        if torch.version.hip is not None or not x_bf16.is_cuda:
            # ROCm and CPU lack bf16-in/fp32-out torch.mm, so upcast
            # bf16-rounded inputs to fp32 to match CUDA/cuBLAS accumulation.
            out = torch.mm(x_2d.float(), weight_bf16.t().float())
        else:
            out = torch.mm(x_2d, weight_bf16.t(), out_dtype=torch.float32)
        return out.view(*x.shape[:-1], weight_bf16.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x_bf16, weight_bf16 = ctx.saved_tensors
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).float()

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output_2d.matmul(weight_bf16.float())
            grad_x = grad_x.view(ctx.input_shape).to(ctx.input_dtype)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            x_2d = x_bf16.reshape(-1, x_bf16.shape[-1])
            grad_weight = grad_output_2d.t().matmul(x_2d.float()).to(ctx.weight_dtype)

        return grad_x, grad_weight


def linear_bf16_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _BFloat16LinearFP32Func.apply(x, weight)
