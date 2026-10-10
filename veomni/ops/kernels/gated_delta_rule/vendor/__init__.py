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
"""Vendored NPU Triton for Qwen3.5 gated delta-rule.

Copied from MindSpeed-MM. The verbatim kernels keep their upstream headers.
Treat them as a drop-in vendor blob so they stay diff-able against upstream —
do not hand-edit kernel logic.

- ``triton/`` — original ``triton-ascend`` kernels for ``npu`` (chunk and conv)
- ``triton_core/`` — newer generation used as glue around ``fla_npu``
  ``torch.ops.npu.*`` on ``npu_ascendc``

Imported lazily by the impls. Importing this package on a host without
``triton`` / ``fla_npu`` does not pull the kernels in.
"""
