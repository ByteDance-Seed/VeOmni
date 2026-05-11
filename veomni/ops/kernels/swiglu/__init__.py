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

"""SwiGLU MLP kernel registry entry.

Default per-model backend:
    - ``liger_kernel``: ``liger_kernel.transformers.swiglu.LigerSwiGLUMLP``
"""

from ...config.registry import BackendSpec, OpScope, OpSpec, register_op


register_op(
    OpSpec(
        name="swiglu_mlp",
        config_field="swiglu_mlp_implementation",
        label="SwiGLU",
        scope=OpScope.PER_MODEL,
        default="liger_kernel",
        backends={
            "liger_kernel": BackendSpec(
                entry="liger_kernel.transformers.swiglu:LigerSwiGLUMLP",
                requires=("liger_kernel",),
            ),
        },
    )
)
