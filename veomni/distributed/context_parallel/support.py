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

"""Platform and operator limits for context parallelism.

Kept separate from the collectives so model construction can ask the question
without importing the exchange helpers.
"""

from __future__ import annotations

from veomni.utils.device import IS_NPU_AVAILABLE


# ``eager`` and ``cann`` both serve CP's local-query/global-KV layout. The CANN
# adapters narrow the replicated buffers to the prefix the kernels assume.
_NPU_CP_DSA_IMPLEMENTATIONS = {
    "dsa_indexer_implementation": frozenset({"eager", "cann"}),
    "dsa_attention_implementation": frozenset({"eager", "cann"}),
}


def check_cp_platform_support(config: object) -> None:
    """Refuse context-parallel setups this release cannot run.

    Ascend implements context parallelism for Qwen4-Exp and DeepSeek-V4.
    DeepSeek-V4 CP supports the ``eager`` and ``cann`` DSA selections.
    """
    if not IS_NPU_AVAILABLE:
        return

    model_type = getattr(config, "model_type", None)
    if model_type == "qwen4_exp":
        return
    if model_type != "deepseek_v4":
        raise NotImplementedError(
            "Context parallelism on Ascend/NPU is currently implemented for Qwen4-Exp and deepseek_v4 only; "
            "set cp_size=1 on Ascend/NPU runs."
        )

    # Fail closed: a missing tree cannot be judged, so it is not admitted.
    ops_config = getattr(config, "ops_config", None)
    if ops_config is None:
        raise NotImplementedError(
            "Context parallelism on Ascend/NPU requires DeepSeek-V4 DSA operator selections, but ops_config is missing."
        )

    for field, implementations in _NPU_CP_DSA_IMPLEMENTATIONS.items():
        implementation = getattr(ops_config, field, None)
        if implementation not in implementations:
            allowed = " or ".join(sorted(implementations))
            raise NotImplementedError(
                f"Context parallelism on Ascend/NPU requires {field} to be {allowed}; got {implementation!r}. "
                "The CANN implementations serve CP's local-query/global-KV layout; set it in "
                "configs/text/deepseek_v4_npu.yaml, or set cp_size=1."
            )
