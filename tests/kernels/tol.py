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

"""Shared numerical tolerances for kernel tests."""

EAGER_ATOL = 1e-6
EAGER_RTOL = 1e-6
EAGER_GRAD_ATOL = 1e-5
EAGER_GRAD_RTOL = 1e-5

RMS_FUSED_ATOL = 2e-3
RMS_FUSED_RTOL = 2e-3
RMS_FUSED_QWEN35_ATOL = 1e-2
RMS_FUSED_QWEN35_RTOL = 1e-2
RMS_FUSED_GRAD_ATOL = 2e-2
RMS_FUSED_GRAD_RTOL = 2e-2
RMS_TRITON_ATOL = 1e-2
RMS_TRITON_RTOL = 1e-2
RMS_TRITON_GRAD_ATOL = 3e-2
RMS_TRITON_GRAD_RTOL = 2e-2
RMS_NPU_ATOL = 1e-2
RMS_NPU_RTOL = 1e-2
RMS_UNWEIGHTED_ATOL = 1e-2
RMS_UNWEIGHTED_RTOL = 1e-2

ROPE_FUSED_ATOL = 1e-2
ROPE_FUSED_RTOL = 1e-2
ROPE_FUSED_GRAD_ATOL = 2e-2
ROPE_FUSED_GRAD_RTOL = 2e-2
ROPE_NPU_ATOL = 2e-2
ROPE_NPU_RTOL = 2e-2

SWIGLU_FUSED_ATOL = 2e-3
SWIGLU_FUSED_RTOL = 2e-3
SWIGLU_FUSED_GRAD_ATOL = 2e-2
SWIGLU_FUSED_GRAD_RTOL = 2e-2

LB_FUSED_ATOL = 1e-4
LB_FUSED_RTOL = 1e-4
LB_FUSED_GRAD_ATOL = 1e-4
LB_FUSED_GRAD_RTOL = 1e-4

CE_FUSED_ATOL = 2e-3
CE_FUSED_RTOL = 2e-3
CE_FUSED_GRAD_ATOL = 2e-2
CE_FUSED_GRAD_RTOL = 2e-2

MOE_FUSED_ATOL = 2e-2
MOE_FUSED_RTOL = 2e-2
MOE_FUSED_GRAD_ATOL = 5e-2
MOE_FUSED_GRAD_RTOL = 5e-2

GDN_FUSED_ATOL = 2e-2
GDN_FUSED_RTOL = 2e-2
GDN_FUSED_GRAD_ATOL = 5e-2
GDN_FUSED_GRAD_RTOL = 5e-2
GDN_CHUNK_ATOL = 5e-2
GDN_CHUNK_RTOL = 5e-2
GDN_CHUNK_GRAD_ATOL = 8e-2
GDN_CHUNK_GRAD_RTOL = 8e-2
