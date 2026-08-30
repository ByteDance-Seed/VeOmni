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

"""Async Ulysses compound kernels.

``async_ulysses_qkv`` / ``async_ulysses_o`` each have ``standard`` and ``dit``
eager rows. Nested RMSNorm uses a raw kernel pair. LayerNorm still calls the
fused CUDA extension in ``shared/norm`` / ``shared/backward``.
"""

from ...registry import register_kernel
from .o_proj.dit import eager as o_dit
from .o_proj.standard import eager as o_standard
from .qkv_proj.dit import eager as qkv_dit
from .qkv_proj.standard import eager as qkv_standard


register_kernel("async_ulysses_qkv", "standard", "eager", qkv_standard.forward, qkv_standard.backward)
register_kernel("async_ulysses_qkv", "dit", "eager", qkv_dit.forward, qkv_dit.backward)
register_kernel("async_ulysses_o", "standard", "eager", o_standard.forward, o_standard.backward)
register_kernel("async_ulysses_o", "dit", "eager", o_dit.forward, o_dit.backward)
