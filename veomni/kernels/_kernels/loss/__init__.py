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

"""Loss kernels.

``load_balancing_loss`` has ``standard`` eager / triton rows. Cross-entropy
lands here later. Callers stack per-layer gate logits to
``[num_layers, tokens, num_experts]``. An empty ``attention_mask`` means
every token counts.
"""

from ...registry import register_kernel
from ...requirement import CudaKernelRequirement
from .load_balancing_loss.standard import eager as lb_eager
from .load_balancing_loss.standard import triton as lb_triton


register_kernel("load_balancing_loss", "standard", "eager", lb_eager.forward, lb_eager.backward)

register_kernel(
    "load_balancing_loss",
    "standard",
    "triton",
    lb_triton.forward,
    lb_triton.backward,
    requirement=CudaKernelRequirement(),
)
