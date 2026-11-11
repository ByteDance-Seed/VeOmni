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

from functools import partial
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.distributed.fsdp import FSDPModule


def _divide_owned_output(output: torch.Tensor, *, factor: float) -> None:
    output.div_(factor)


def configure_singleton_gradient_scaling(
    module: "FSDPModule", *, mesh_size: int, reduce_dtype: torch.dtype | None, gradient_divide_factor: float
) -> bool:
    """Scale a fresh singleton FSDP group's owned output without a division temporary.

    Call immediately after fully_shard, before setting a native divide factor.
    The public hook receives FSDP's allocated reduction output, not borrowed
    autograd gradients. Nontrivial reductions and overflow-sensitive FP16 keep
    native normalization. The caller owns backend selection and fallback.
    """
    if (
        mesh_size != 1
        or reduce_dtype not in (torch.float32, torch.bfloat16)
        or not callable(getattr(module, "set_all_reduce_hook", None))
    ):
        return False
    if gradient_divide_factor != 1:
        module.set_all_reduce_hook(partial(_divide_owned_output, factor=gradient_divide_factor))
    return True
