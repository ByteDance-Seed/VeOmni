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

"""Smoke tests for ``format_kernel_selection_summary``.

These cover the formatter's branching, not the underlying kernel selection
itself: the three ``_attn_implementation`` shapes (flat / HF v5 dict / per
sub-config) and the OpSlot section's labelling for unbound / eager / kernel
states.
"""

from types import ModuleType, SimpleNamespace

from veomni.ops import format_kernel_selection_summary
from veomni.ops.dispatch import OpSlot


def _model(**cfg_kwargs):
    return SimpleNamespace(config=SimpleNamespace(**cfg_kwargs))


def test_no_args_emits_header_footer_and_core_rows():
    out = format_kernel_selection_summary()
    assert "============= Kernel selection summary =============" in out
    assert out.rstrip().endswith("=====================================================")
    # GLOBAL ops + cross-entropy are always present
    for alias in ("cross_entropy", "_fused_moe_forward", "_flash_attention_forward", "_load_balancing_loss"):
        assert alias in out
    # No model section / OpSlot section when called without args
    assert "model " not in out
    assert "OpSlot bindings" not in out


def test_flat_attn_implementation_and_model_label():
    out = format_kernel_selection_summary(model=_model(model_type="qwen3", _attn_implementation="flash_attention_2"))
    assert "(model_type=qwen3)" in out
    # Single attn line with the flat string (no [sub_config] suffix).
    attn_lines = [line for line in out.splitlines() if "attn_implementation" in line]
    assert len(attn_lines) == 1
    assert "[" not in attn_lines[0]
    assert "flash_attention_2" in attn_lines[0]


def test_dict_form_attn_implementation_expands_per_key():
    out = format_kernel_selection_summary(
        model=_model(_attn_implementation={"text_config": "flash_attention_2", "vision_config": "sdpa"})
    )
    assert "attn_implementation[text_config]" in out
    assert "flash_attention_2" in out
    assert "attn_implementation[vision_config]" in out
    assert "sdpa" in out


def test_sub_config_attn_implementation_collected():
    out = format_kernel_selection_summary(
        model=SimpleNamespace(
            config=SimpleNamespace(
                text_config=SimpleNamespace(_attn_implementation="sdpa"),
                vision_config=SimpleNamespace(_attn_implementation="eager"),
            )
        )
    )
    assert "attn_implementation[text_config]" in out
    assert "attn_implementation[vision_config]" in out


def test_opslot_bindings_section_sorted_with_state_labels():
    mod = ModuleType("fake_modeling")

    unbound = OpSlot("rms_norm", "standard")  # never bound

    eager = OpSlot("rms_norm", "standard")  # bound, but resolved to eager (kernel=None)
    eager._impl_name = "eager"

    def my_kernel(x):
        return x

    bound = OpSlot("rms_norm", "standard")
    bound._impl_name = "my_impl"
    bound._kernel = my_kernel

    # Names chosen so alphabetical sort != insertion order, to verify sorting.
    mod.veomni_z_unbound = unbound
    mod.veomni_a_eager = eager
    mod.veomni_m_bound = bound
    mod.not_a_slot = "ignored"  # non-OpSlot attrs must be filtered out

    out = format_kernel_selection_summary(modeling_module=mod)

    assert "-- OpSlot bindings (fake_modeling) --" in out
    assert out.index("veomni_a_eager") < out.index("veomni_m_bound") < out.index("veomni_z_unbound")
    assert "<unbound> -> eager" in out
    assert "eager -> eager" in out
    assert "my_impl -> my_kernel" in out
    assert "not_a_slot" not in out
