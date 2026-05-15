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

"""Unit tests for veomni.utils.moe_router_replay.

These tests cover the module's full surface without requiring GPU or any
real MoE model weights:

* ``maybe_replay_indices`` is a bit-identical passthrough when no manager is
  active.
* ``set_active_replay`` / ``get_active_replay`` round-trip correctly.
* A duck-typed mock manager receives the documented call signature and its
  return value is honoured by ``maybe_replay_indices``.
* ``validate_model_for_replay`` accepts declared-supported ``model_type``
  values and fails fast with an actionable message otherwise.

It also includes a patchgen-drift guard: the generated ``patched_modeling_*``
files for every wired MoE family MUST import and call
``maybe_replay_indices``. The NPU variant is fragile under solo regeneration
(see the docstring in ``qwen3_5_moe_npu_patch_gen_config.py``), so a static
check here prevents the call site silently disappearing on a stale
``make patchgen``.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from veomni.utils import moe_router_replay
from veomni.utils.moe_router_replay import (
    SUPPORTED_MOE_MODEL_TYPES,
    get_active_replay,
    maybe_replay_indices,
    set_active_replay,
    validate_model_for_replay,
)


@pytest.fixture(autouse=True)
def _restore_active_manager():
    """Guarantee the module-level singleton is reset between tests."""
    prior = get_active_replay()
    try:
        set_active_replay(None)
        yield
    finally:
        set_active_replay(prior)


def _make_routing(num_tokens: int = 6, num_experts: int = 8, top_k: int = 2):
    """Return (module, routing_scores, top_indices) matching the hook contract."""
    torch.manual_seed(0)
    logits = torch.randn(num_tokens, num_experts)
    scores = torch.softmax(logits, dim=-1)
    _, top_indices = torch.topk(scores, k=top_k, dim=-1)
    return nn.Linear(4, num_experts), scores, top_indices


# ---------------------------------------------------------------- disabled path


def test_maybe_replay_indices_passthrough_when_no_manager():
    module, scores, indices = _make_routing()

    out = maybe_replay_indices(module, scores, indices)

    # Must be the SAME tensor object, not a copy — default path adds no overhead.
    assert out is indices


def test_get_and_set_active_replay_roundtrip():
    assert get_active_replay() is None
    sentinel = object()
    set_active_replay(sentinel)
    assert get_active_replay() is sentinel
    set_active_replay(None)
    assert get_active_replay() is None


# --------------------------------------------------------------- mock manager


class _RecordingManager:
    """Minimal duck-typed manager: records every call, optionally substitutes indices."""

    def __init__(self, substitute_indices: torch.Tensor | None = None):
        self.calls: list[dict] = []
        self.substitute_indices = substitute_indices

    def on_router_forward(self, module, routing_scores, top_indices):
        self.calls.append(
            dict(
                module=module,
                routing_scores=routing_scores,
                top_indices=top_indices,
            )
        )
        if self.substitute_indices is None:
            return top_indices
        return self.substitute_indices


def test_manager_receives_documented_arguments():
    module, scores, indices = _make_routing()
    mgr = _RecordingManager()
    set_active_replay(mgr)

    out = maybe_replay_indices(module, scores, indices)

    assert len(mgr.calls) == 1
    call = mgr.calls[0]
    assert call["module"] is module
    assert call["routing_scores"] is scores
    assert call["top_indices"] is indices
    # Passthrough manager returns the input unchanged.
    assert out is indices


def test_manager_substitution_is_honoured():
    module, scores, indices = _make_routing()
    substitute = torch.zeros_like(indices)
    mgr = _RecordingManager(substitute_indices=substitute)
    set_active_replay(mgr)

    out = maybe_replay_indices(module, scores, indices)

    assert out is substitute
    assert torch.equal(out, torch.zeros_like(indices))


def test_manager_reentrancy_across_layers():
    """Two routers calling maybe_replay_indices in sequence both reach the manager."""
    m1, s1, i1 = _make_routing(num_tokens=4, num_experts=8, top_k=2)
    m2, s2, i2 = _make_routing(num_tokens=4, num_experts=8, top_k=2)
    mgr = _RecordingManager()
    set_active_replay(mgr)

    maybe_replay_indices(m1, s1, i1)
    maybe_replay_indices(m2, s2, i2)

    assert [c["module"] for c in mgr.calls] == [m1, m2]


def test_clearing_manager_restores_passthrough():
    module, scores, indices = _make_routing()
    mgr = _RecordingManager(substitute_indices=torch.zeros_like(indices))
    set_active_replay(mgr)
    maybe_replay_indices(module, scores, indices)
    assert len(mgr.calls) == 1

    set_active_replay(None)
    out = maybe_replay_indices(module, scores, indices)
    assert out is indices
    # Manager did NOT receive the second call.
    assert len(mgr.calls) == 1


# ---------------------------------------------------------- validator behaviour


@pytest.mark.parametrize("model_type", sorted(SUPPORTED_MOE_MODEL_TYPES))
def test_validate_accepts_supported_model_types(model_type):
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    validate_model_for_replay(model)  # must not raise


def test_validate_rejects_unknown_model_type():
    model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    with pytest.raises(RuntimeError, match="router replay is not wired"):
        validate_model_for_replay(model)


def test_validate_error_points_at_extension_surface():
    model = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v3"))
    with pytest.raises(RuntimeError) as exc:
        validate_model_for_replay(model)
    msg = str(exc.value)
    # Error must name both the registry and the patch surface the maintainer touches.
    assert "SUPPORTED_MOE_MODEL_TYPES" in msg
    assert "maybe_replay_indices" in msg


def test_validate_rejects_missing_config():
    model = SimpleNamespace()  # no .config
    with pytest.raises(RuntimeError, match="cannot determine model.config.model_type"):
        validate_model_for_replay(model)


def test_validate_rejects_config_without_model_type():
    model = SimpleNamespace(config=SimpleNamespace())
    with pytest.raises(RuntimeError, match="cannot determine model.config.model_type"):
        validate_model_for_replay(model)


def test_validate_accepts_ddp_style_module_wrapper():
    """DDP / PeftModel-style wrappers hide the real model under ``.module``.

    The validator's fallback walk must see through one level of wrapper.
    """
    inner = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"))
    wrapper = SimpleNamespace(module=inner)  # no .config on wrapper
    validate_model_for_replay(wrapper)  # must not raise


# --------------------------------------------------- patchgen drift guard
#
# The router-replay wiring lives inside generated patched_modeling_*.py files.
# If `make patchgen` is ever run with a broken shared-state cache (see the
# NPU gen_config docstring), the `maybe_replay_indices` call site can silently
# drop out while the import stays — or vice versa. These checks surface such a
# regression at test time instead of at the first RL step.


_REPO_ROOT = Path(moe_router_replay.__file__).resolve().parents[2]
_GENERATED_FILES = [
    _REPO_ROOT / "veomni/models/transformers/qwen3_moe/generated/patched_modeling_qwen3_moe_gpu.py",
    _REPO_ROOT / "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_gpu.py",
    _REPO_ROOT / "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_npu.py",
]


@pytest.mark.parametrize("path", _GENERATED_FILES, ids=lambda p: p.name)
def test_generated_file_wires_maybe_replay_indices(path: Path):
    assert path.is_file(), f"generated file missing: {path} — run `make patchgen`"
    src = path.read_text()
    # Import must be present. The patch site imports `maybe_replay_indices`
    # alongside `get_active_replay` (to skip the guarded block on the default
    # path), so match module path + imported name rather than a fixed line.
    assert re.search(
        r"from\s+veomni\.utils\.moe_router_replay\s+import\s+[^\n]*\bmaybe_replay_indices\b",
        src,
    ), f"{path.name} dropped the maybe_replay_indices import — patchgen state likely stale"
    # Call site must be present (match arbitrary whitespace/newlines after the name).
    assert re.search(r"\bmaybe_replay_indices\s*\(", src), (
        f"{path.name} imports maybe_replay_indices but never calls it — patchgen regressed"
    )
