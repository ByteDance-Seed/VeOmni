# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Shared helpers for attention adapters and mask builders."""

import torch


def require_all(condition: torch.Tensor, message: str) -> None:
    """Require every element to be true without synchronizing CUDA to Python.

    ``torch._assert_async`` is private but supported by the pinned PyTorch 2.10
    and 2.11 releases. On CUDA, a failed assertion surfaces at a later kernel
    launch and invalidates the process's CUDA context instead of raising the
    catchable ``ValueError`` used by the CPU branch.
    """
    all_true = condition.all()
    if all_true.device.type == "cpu":
        if not bool(all_true):
            raise ValueError(message)
        return

    torch._assert_async(all_true, message)
