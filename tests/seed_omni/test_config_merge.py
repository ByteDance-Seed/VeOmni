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

"""The deep merge every layered omni config rests on.

`_deep_update` is what makes `__inherit__` bases, the launcher's global `model:`
block, a checkpoint's per-module entries and a `modules:` YAML compose instead of
shadow each other. Its edge cases are silent — a wrong answer does not raise, it
quietly runs the wrong kernel or the wrong parallelism — so they are pinned here.
"""

import yaml

from veomni.arguments.omni_parser import load_yaml_with_inherit
from veomni.arguments.parser import _deep_update


def test_a_key_named_with_nothing_under_it_keeps_what_is_there():
    """An empty mapping is a merge of nothing, not an erasure.

    This is how a layer says "I have nothing to say about this": a bare module
    name under `modules:`, an `accelerator:` block that only sets a sibling
    field. Reading it as an erasure let such an entry silently take down every
    layer beneath it — the module's exported kernels, the inference eager
    default — and nothing failed loudly when it did.
    """
    base = {"accelerator": {"fsdp_config": {"fsdp_mode": "eager"}}, "model_path": "vision"}

    assert _deep_update(dict(base), {"accelerator": {}}) == base
    assert _deep_update({"modules": base}, {"modules": {}}) == {"modules": base}


def test_an_explicit_none_still_clears():
    """`None` is the one way to unset, and stays distinct from `{}`."""
    merged = _deep_update({"accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2"}}}, {"accelerator": None})

    assert merged == {"accelerator": None}


def test_nested_mappings_merge_while_lists_and_scalars_replace():
    merged = _deep_update(
        {"accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2", "reshard": True}, "sizes": [1, 2]}},
        {"accelerator": {"fsdp_config": {"fsdp_mode": "ddp"}, "sizes": [4]}},
    )

    # The sibling the override never named survives the merge...
    assert merged["accelerator"]["fsdp_config"] == {"fsdp_mode": "ddp", "reshard": True}
    # ...while a list replaces rather than concatenates.
    assert merged["accelerator"]["sizes"] == [4]


def test_an_inheriting_config_does_not_erase_a_base_by_naming_a_key(tmp_path):
    """The same rule, reached the way configs actually reach it."""
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({"janus_vqvae": {"model_path": "janus_vqvae", "ops_implementation": {"attn": "eager"}}}),
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({"__inherit__": "base.yaml", "janus_vqvae": {}}), encoding="utf-8")

    assert load_yaml_with_inherit(str(child)) == {
        "janus_vqvae": {"model_path": "janus_vqvae", "ops_implementation": {"attn": "eager"}}
    }
