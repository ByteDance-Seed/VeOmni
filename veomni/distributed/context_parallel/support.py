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


_CP_DSA_FIELDS = ("dsa_indexer_implementation", "dsa_attention_implementation")


def check_cp_platform_support(config: object) -> None:
    """Refuse context-parallel setups this release cannot run.

    Ascend implements context parallelism for DeepSeek-V4 only. The CANN
    DeepSeek-V4 indexer and sparse attention reject CP's local-query/global-KV
    layout, so a ``cann`` selection is refused on every device. Other DSA
    implementations are admitted.
    """
    model_type = getattr(config, "model_type", None)
    if IS_NPU_AVAILABLE and model_type != "deepseek_v4":
        raise NotImplementedError(
            "Context parallelism is GPU-only in this release except for deepseek_v4; set cp_size=1 on Ascend/NPU runs."
        )

    if model_type != "deepseek_v4":
        return

    ops_config = getattr(config, "ops_config", None)
    if ops_config is None:
        return

    for field in _CP_DSA_FIELDS:
        # ``ops_config`` resolves dataclass defaults, so an unset field reads
        # back as its default rather than ``None``.
        implementation = getattr(ops_config, field, None)
        if implementation == "cann":
            raise NotImplementedError(
                f"Context parallelism does not support {field}='cann'. "
                "The CANN DeepSeek-V4 sparse attention rejects CP's local-query/global-KV geometry and "
                "the CANN indexer rejects CP's explicit cu_seqlen ranges, so training fails in the first "
                "compressed layer. Select another implementation for both DSA operators, or set cp_size=1."
            )
