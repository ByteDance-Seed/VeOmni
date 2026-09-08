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

"""Check that in-tree YAML op names resolve to registered implementations."""

from __future__ import annotations

from pathlib import Path

import yaml

from veomni.ops import OP_REGISTRY


REPO_ROOT = Path(__file__).parents[3]

_FIELD_TO_OP = {
    "attn_implementation": "attention",
    "causal_conv1d_implementation": "causal_conv1d",
    "chunk_gated_delta_rule_implementation": "chunk_gated_delta_rule",
    "cross_entropy_loss_implementation": "cross_entropy_loss",
    "dsa_attention_implementation": "dsa_attention",
    "dsa_indexer_implementation": "dsa_indexer",
    "load_balancing_loss_implementation": "load_balancing_loss",
    "mhc_implementation": "mhc",
    "moe_implementation": "moe_experts",
    "rms_norm_gated_implementation": "rms_norm_gated",
    "rms_norm_implementation": "rms_norm",
    "rotary_pos_emb_implementation": "rope",
    "rotary_pos_emb_vision_implementation": "rope_vision",
    "swiglu_mlp_implementation": "swiglu_mlp",
}


def _registered_implementations(op: str) -> set[str]:
    """Return implementation names registered across every op variant and device."""
    return {key[2] for key in OP_REGISTRY._entries if key[0] == op}


def test_yaml_op_implementation_names_are_registered():
    """Keep repository configs on canonical registry names instead of private aliases."""
    invalid: list[str] = []
    for root in (REPO_ROOT / "configs", REPO_ROOT / "tests"):
        for path in sorted(root.rglob("*.yaml")):
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                continue
            model = document.get("model")
            if not isinstance(model, dict):
                continue
            implementations = model.get("ops_implementation")
            if not isinstance(implementations, dict):
                continue

            for field, implementation in implementations.items():
                op = _FIELD_TO_OP.get(field)
                if op is None:
                    invalid.append(f"{path.relative_to(REPO_ROOT)}: unmapped implementation field {field!r}")
                    continue
                if implementation in _registered_implementations(op):
                    continue
                invalid.append(f"{path.relative_to(REPO_ROOT)}: {field}={implementation!r} ({op})")

    assert not invalid, "YAML op implementation names are not registered:\n" + "\n".join(invalid)
