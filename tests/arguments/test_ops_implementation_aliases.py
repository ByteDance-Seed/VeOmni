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

"""Renamed ``ops_implementation`` values keep parsing with a warning; non-implementation values fail at parse time."""

from __future__ import annotations

import pytest

import veomni.utils.import_utils as import_utils
from veomni.arguments import OpsImplementationConfig, arguments_types


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    messages: list[str] = []
    monkeypatch.setattr(arguments_types.logger, "warning_rank0", lambda msg, *a, **k: messages.append(msg))
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    # Ascend NPU rejects Hub attention and re-resolves GPU defaults, which these name tests do not cover.
    monkeypatch.setattr(import_utils, "is_torch_npu_available", lambda: False)
    return messages


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("veomni_flash_attention_2_with_sp", "veomni_flash_attention_2"),
        ("veomni_flash_attention_2_hub_with_sp", "veomni_flash_attention_2_hub"),
        ("veomni_flash_attention_3_with_sp", "veomni_flash_attention_3"),
        ("veomni_flash_attention_3_hub_with_sp", "veomni_flash_attention_3_hub"),
        ("veomni_flash_attention_4_with_sp", "veomni_flash_attention_4"),
        ("veomni_flex_attention_with_sp", "veomni_flex_attention"),
        ("veomni_magi_attention_with_sp", "veomni_magi_attention"),
    ],
)
def test_with_sp_attention_names_resolve_to_the_unsuffixed_name(old: str, new: str, warnings: list[str]):
    assert OpsImplementationConfig(attn_implementation=old).attn_implementation == new
    assert len(warnings) == 1 and old in warnings[0] and new in warnings[0]


def test_fused_mlu_triton_resolves_to_fused_triton(warnings: list[str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(import_utils, "is_torch_mlu_available", lambda: True)
    assert OpsImplementationConfig(moe_implementation="fused_mlu_triton").moe_implementation == "fused_triton"
    assert len(warnings) == 1 and "fused_mlu_triton" in warnings[0]


@pytest.mark.parametrize(
    ("mlu", "apex", "expected"),
    [(False, False, "fused_quack"), (True, True, "fused_mlu"), (True, False, "fused_triton")],
)
def test_legacy_fused_moe_resolves_per_host(
    mlu: bool, apex: bool, expected: str, warnings: list[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(import_utils, "is_torch_mlu_available", lambda: mlu)
    monkeypatch.setattr(import_utils, "is_apex_mlu_available", lambda: apex)
    assert OpsImplementationConfig(moe_implementation="fused").moe_implementation == expected
    assert len(warnings) == 1 and expected in warnings[0]


def test_qwen3_5_rms_norm_value_is_rejected_at_parse_time(warnings: list[str]):
    with pytest.raises(ValueError, match="variant name .*'offset'"):
        OpsImplementationConfig(rms_norm_implementation="qwen3_5")


def test_current_names_do_not_warn(warnings: list[str]):
    config = OpsImplementationConfig(attn_implementation="veomni_flash_attention_2", moe_implementation="fused_triton")
    assert (config.attn_implementation, config.moe_implementation) == ("veomni_flash_attention_2", "fused_triton")
    assert warnings == []
