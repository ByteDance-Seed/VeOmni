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

import triton
import triton.language as tl


# FIXME: Maybe we should allow different `GROUP_SIZE` along `M` and `N`. Needs more investigation
# on PTX produced.
@triton.jit
def get_pid_mn(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_SIZE: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def make_blocked(t: tl.tensor, intermediate_type: tl.dtype) -> tl.tensor:
    """Forcibly convert tensor (from "mma" layout) into "blocked" layout.

    `intermediate_type` affects performance. Usually `tl.bfloat16` or `tl.float16` should be used.
    INTERNALLY `t` IS CONVERTED TO `intermediate_type` AND BACK  SO THE PRECISION CAN DROP.

    ATM Triton does such conversion prior to storing tensor into global memory. This usually doesn't
    matter as we usually only store the accumulator once. However, if we'd like to perform some
    element-wise operation on the accumulator and save both pre-op and post-op results, Triton will
    do the conversion twice, and hence hurt performance.

    In such cases, forcibly convert tensor eagerly can help performance. This is not guaranteed, so
    be sure to benchmark before applying this "optimization".

    NOTE: Once Triton can optimize away multiple layout conversions, this hack should be removed.
    """
    # This really relies on Triton's internal implementation.. See implementation of `expand_dims`
    # op, it triggers emission of `triton_gpu.convert_layout`.
    return t.to(intermediate_type).expand_dims(0).reshape(t.shape)
