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
"""Newer-generation Triton (``triton-ascend``) kernels for gated delta-rule
(vendored via MindSpeed-MM's ``triton_core``, derived from fla_npu commit c2e3d83f —
https://github.com/flashserve/flash-linear-attention-npu).

This is a distinct, newer kernel generation from the sibling ``triton/`` package
(e.g. a rewritten ``solve_tril``, an autotuned ``chunk_scaled_dot_kkt``) — the
name ``triton_core`` refers to that kernel generation, not to the ``fla_npu``
runtime package. These kernels are used by the ``flash_gated_delta_rule`` chunk
path as glue *around* the heavy ``fla_npu`` ``torch.ops.npu.*`` fused ops:
``chunk_scaled_dot_kkt``, ``l2norm``, ``solve_tril`` (arch35 only), and ``utils``.
"""
