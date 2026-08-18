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
"""Vendored NPU kernels for Qwen3.5's gated delta-rule linear attention.

Copied from MindSpeed-MM (https://gitcode.com/Ascend/MindSpeed-MM), which ports
flash-linear-attention (FLA) to Ascend NPU; the newer ``triton_core`` kernels
derive from fla_npu commit c2e3d83f as redistributed by MindSpeed-MM. The verbatim
kernels keep their upstream headers; VeOmni-authored glue (the ``triton_core``
package ``__init__`` and the adapted ``flash_gated_delta_rule.py``) carries
VeOmni's own header. Treat the kernels as a drop-in vendor blob so they stay
diff-able against upstream — do not hand-edit kernel logic. VeOmni's registry-facing
wrappers live one level up (``npu_causal_conv1d.py`` and the
``chunk_gated_delta_rule`` factories in the package ``__init__``); those are the
only entry points other code should call.

Two generations of the gated delta-rule stack live here:

- ``triton/`` — the original pure-Triton (``triton-ascend``) kernels, backing
  the ``npu`` backend (``chunk_gated_delta_rule_mm.py``).
- ``triton_core/`` + ``flash_gated_delta_rule.py`` — the newer AscendC path
  (``npu_ascendc`` backend). The heavy GDN compute is delegated to the external
  ``fla_npu`` package (``torch.ops.npu.*`` fused ops, installed manually on NPU);
  ``triton_core`` (a newer *generation* of Triton kernels) supplies the glue
  around the fused ops on the chunk path. ``flash_gated_delta_rule.py`` is
  adapted from upstream — see its module docstring for the exact diffs.

These modules require ``triton`` (``triton-ascend`` on NPU) — and ``flash_gated_delta_rule``
additionally requires ``fla_npu`` — and are imported lazily by the kernel
factories, so importing the parent package on a host without those does not
pull them in.
"""
