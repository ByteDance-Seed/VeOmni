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

"""RMSNorm kernel registry entry.

Default per-model backends:
    - ``liger_kernel``: ``liger_kernel.transformers.rms_norm.LigerRMSNorm``
    - ``npu``: ``torch_npu.npu_rms_norm`` via ``veomni.ops.kernels.rms_norm.npu``
Models can register a ``triton`` backend via ``extra_backends`` in their
``device_patch.py`` (e.g. DeepSeek V3 batch-invariant kernel).
"""

from ...config.registry import BackendSpec, OpScope, OpSpec, register_op


register_op(
    OpSpec(
        name="rms_norm",
        config_field="rms_norm_implementation",
        label="RMSNorm",
        scope=OpScope.PER_MODEL,
        default="eager",
        backends={
            "liger_kernel": BackendSpec(
                entry="liger_kernel.transformers.rms_norm:LigerRMSNorm",
                requires=("liger_kernel",),
            ),
            "npu": BackendSpec(
                entry="veomni.ops.kernels.rms_norm.npu:rms_norm_forward_npu",
                requires=("torch_npu",),
                replace_forward=True,
            ),
        },
    )
)
