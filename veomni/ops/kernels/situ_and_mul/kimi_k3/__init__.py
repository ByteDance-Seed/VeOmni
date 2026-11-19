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

"""situ_and_mul ``kimi_k3`` variant: SituGLU activation for Kimi-K3 MLPs.

* ``eager`` is the fp32-compute PyTorch reference; ``eager.situ_eager`` is
  the single definition of the SituGLU math on pre-split (gate, up) halves
  and is also called by the Kimi K3 MoE expert loops, so their numerics are
  identical to this family by construction.
* ``triton`` is the fused Triton-Ascend kernel vendored from MindSpeed-MM
  (fused forward + backward, fp32 compute inside the kernel).
"""
