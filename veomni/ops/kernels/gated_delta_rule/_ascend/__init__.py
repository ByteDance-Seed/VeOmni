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
"""Vendored NPU Triton kernels for Qwen3.5's gated delta-rule linear attention.

Copied verbatim from MindSpeed-MM (https://gitcode.com/Ascend/MindSpeed-MM),
which in turn ports flash-linear-attention (FLA) to Ascend NPU; all files keep
their original Apache-2.0 headers. Do not hand-edit the kernel logic here —
treat this as a drop-in vendor blob so it stays diff-able against upstream. VeOmni's registry-facing wrappers live one level up
(``npu_causal_conv1d.py`` and the ``chunk_gated_delta_rule`` factory in the
package ``__init__``); those are the only entry points other code should call.

These modules require ``triton`` (``triton-ascend`` on NPU) and are imported
lazily by the kernel factories, so importing the parent package on a host
without triton-ascend does not pull them in.
"""
