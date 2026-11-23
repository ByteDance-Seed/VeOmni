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

"""Keep device-specific math out of model files."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


REPO_ROOT = Path(__file__).parents[2]
_BRANCH = re.compile(
    r"IS_NPU_AVAILABLE"
    r"|import\s+torch_npu\b"
    r"|get_device_type\(\)\s*==\s*[\"']npu[\"']"
    r"|(?:device_type|\.type)\s*[!=]=\s*[\"']npu[\"']"
)
# Allowlist keys are the exact source lines. A new statement is a new key.
_ALLOWED = {
    (
        "veomni/models/diffusers/qwen_image/qwen_image_transformer/modeling_qwen_image_transformer.py",
        'if device.type == "npu":',
    ),
    (
        "veomni/models/diffusers/qwen_image/qwen_image_transformer/modeling_qwen_image_transformer.py",
        'if attention_mask is None or device_type != "npu":',
    ),
    (
        "veomni/models/diffusers/hunyuanvideo15/hunyuanvideo15_transformer/native/modules/attention.py",
        'if q.device.type == "npu":',
    ),
}


def _model_files() -> list[Path]:
    root = REPO_ROOT / "veomni" / "models"
    files = list(root.rglob("modeling_*.py"))
    files.append(root / "diffusers/hunyuanvideo15/hunyuanvideo15_transformer/native/modules/attention.py")
    return sorted(set(files))


def test_modeling_device_branches_stay_on_the_allowlist() -> None:
    """New NPU branches in model files must move into ops or an explicit allowlist entry."""
    found: Counter[tuple[str, str]] = Counter()
    unexpected: list[str] = []
    for path in _model_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(source.splitlines(), start=1):
            if _BRANCH.search(line) is None:
                continue
            key = (relative, line.strip())
            found[key] += 1
            if key not in _ALLOWED:
                unexpected.append(f"{relative}:{lineno} {line.strip()}")

    assert unexpected == []
    assert found == Counter(dict.fromkeys(_ALLOWED, 1))
