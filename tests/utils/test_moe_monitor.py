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

"""Unit tests for veomni.utils.moe_monitor.

Single-process tests covering the public surface:
* extractor registry resolves router classes by name and returns indices
* attach_moe_router_monitor wires hooks; record() accumulates correctly
* pause()/resume() gate accumulation
* get_load_matrix() normalizes rows and resets counts
* compute_vio() numerical correctness on hand-built inputs
* compute_metrics() returns the documented key set
* unknown router class is silently skipped at attach time
* extractor returning None triggers a single warning per class
"""

from __future__ import annotations

import torch
import torch.nn as nn

from veomni.utils import moe_monitor
from veomni.utils.moe_monitor import (
    EXTERNAL_RECORD_ROUTERS,
    ROUTER_EXTRACTORS,
    MoERouterMonitor,
    attach_moe_router_monitor,
    get_active_monitor,
    record_router_indices,
    register_external_record_router,
    register_router_extractor,
    set_active_monitor,
)


# ---------------------------------------------------------------------------
# Fake router modules — one whose forward returns the qwen3-style tuple, one
# that returns nothing useful (to exercise the None-extractor path).
# ---------------------------------------------------------------------------


class FakeQwenRouter(nn.Module):
    """Mimics Qwen3MoeTopKRouter's patched output shape: (logits, scores, indices)."""

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Deterministic "indices" we return on every call; tests parameterize them.
        self._next_indices: torch.Tensor | None = None

    def set_next_indices(self, indices: torch.Tensor) -> None:
        self._next_indices = indices

    def forward(self, hidden_states: torch.Tensor):
        # Shape conventions match the patched Qwen3 router.
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        if self._next_indices is None:
            indices = torch.zeros(num_tokens, self.top_k, dtype=torch.long, device=device)
        else:
            indices = self._next_indices.to(device)
        logits = torch.zeros(num_tokens, self.num_experts, device=device)
        scores = torch.zeros(num_tokens, self.top_k, device=device)
        return logits, scores, indices


# Register the fake router under its class name so the registry finds it.
register_router_extractor("FakeQwenRouter")(ROUTER_EXTRACTORS["Qwen3MoeTopKRouter"])


class TwoLayerModel(nn.Module):
    def __init__(self, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.router0 = FakeQwenRouter(num_experts, top_k)
        self.router1 = FakeQwenRouter(num_experts, top_k)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, hidden_states: torch.Tensor):
        self.router0(hidden_states)
        self.router1(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_monitor_and_attach(num_experts: int = 4, top_k: int = 2):
    """Fresh monitor + model with hooks wired in. Activates the singleton."""
    monitor = MoERouterMonitor(num_experts=num_experts)
    model = TwoLayerModel(num_experts=num_experts, top_k=top_k)
    attached = attach_moe_router_monitor(model, monitor)
    assert attached == 2, f"expected 2 routers attached, got {attached}"
    set_active_monitor(monitor)
    return monitor, model


def test_attach_and_record_basic():
    monitor, model = _make_monitor_and_attach()
    try:
        # 6 tokens, top_k=2, all routed to expert 0 in layer 0, expert 3 in layer 1.
        idx_layer0 = torch.zeros(6, 2, dtype=torch.long)
        idx_layer1 = torch.full((6, 2), 3, dtype=torch.long)

        model.router0.set_next_indices(idx_layer0)
        model.router1.set_next_indices(idx_layer1)
        model(torch.zeros(6, 8))

        # 6 tokens * top_k=2 = 12 selections per layer.
        # Layer 0: expert 0 got all 12; others 0.
        # Layer 1: expert 3 got all 12; others 0.
        # get_load_matrix normalizes, so each row sums to 1.
        load = monitor.get_load_matrix(current_step=0)
        assert load.shape == (2, 4)
        assert torch.allclose(load[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        assert torch.allclose(load[1], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    finally:
        set_active_monitor(None)


def test_disable_is_sticky_against_resume():
    """A subsequent ``resume()`` must not un-do an explicit ``disable()``."""
    monitor, model = _make_monitor_and_attach()
    try:
        model.router0.set_next_indices(torch.zeros(4, 2, dtype=torch.long))
        model.router1.set_next_indices(torch.zeros(4, 2, dtype=torch.long))
        monitor.disable()
        monitor.resume()  # phase-scoped callers (e.g. verl) call this every batch
        model(torch.zeros(4, 8))
        # Hook still fired but the paused check inside the hook short-circuited.
        assert monitor._counts == {}
    finally:
        set_active_monitor(None)


def test_pause_resume_gates_accumulation():
    monitor, model = _make_monitor_and_attach()
    try:
        model.router0.set_next_indices(torch.zeros(4, 2, dtype=torch.long))
        model.router1.set_next_indices(torch.zeros(4, 2, dtype=torch.long))

        # Run once normally — counts should accumulate.
        model(torch.zeros(4, 8))
        # Pause and run again — counts must NOT change.
        monitor.pause()
        before = {mid: c.clone() for mid, c in monitor._counts.items()}
        model(torch.zeros(4, 8))
        for mid, c in monitor._counts.items():
            assert torch.equal(c, before[mid]), "pause() should freeze accumulation"
        # Resume — counts grow again.
        monitor.resume()
        model(torch.zeros(4, 8))
        for mid, c in monitor._counts.items():
            assert (c > before[mid]).any(), "resume() should re-enable accumulation"
    finally:
        set_active_monitor(None)


def test_get_load_matrix_resets_counts():
    monitor, model = _make_monitor_and_attach()
    try:
        model.router0.set_next_indices(torch.zeros(4, 2, dtype=torch.long))
        model.router1.set_next_indices(torch.zeros(4, 2, dtype=torch.long))
        model(torch.zeros(4, 8))
        _ = monitor.get_load_matrix(current_step=5)
        for c in monitor._counts.values():
            assert c.sum().item() == 0, "counts must be zeroed after get_load_matrix"
        # Step range bookkeeping.
        assert monitor._last_step_range[1] == 5
        assert monitor._accumulate_start_step == 6
    finally:
        set_active_monitor(None)


def test_compute_vio_numerics():
    # 1 layer, 4 experts. Uniform: each expert gets 1/4. deviation = 0.
    uniform = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    vio_uniform = MoERouterMonitor.compute_vio(uniform)
    assert torch.allclose(vio_uniform["max_vio"], torch.tensor([0.0]))
    assert torch.allclose(vio_uniform["min_vio"], torch.tensor([0.0]))
    assert torch.allclose(vio_uniform["avg_vio"], torch.tensor([0.0]))

    # Fully collapsed: expert 0 gets all tokens.
    # deviation = [4*1 - 1, 4*0 - 1, 4*0 - 1, 4*0 - 1] = [3, -1, -1, -1].
    collapsed = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    vio_c = MoERouterMonitor.compute_vio(collapsed)
    assert torch.allclose(vio_c["max_vio"], torch.tensor([3.0]))
    assert torch.allclose(vio_c["min_vio"], torch.tensor([-1.0]))
    # |dev| mean = (3 + 1 + 1 + 1) / 4 = 1.5
    assert torch.allclose(vio_c["avg_vio"], torch.tensor([1.5]))


def test_compute_metrics_key_shape():
    monitor, model = _make_monitor_and_attach(num_experts=4, top_k=2)
    try:
        model.router0.set_next_indices(torch.zeros(4, 2, dtype=torch.long))
        model.router1.set_next_indices(torch.zeros(4, 2, dtype=torch.long))
        model(torch.zeros(4, 8))
        metrics = monitor.compute_metrics(current_step=10)
        expected_keys = (
            {f"moe/max_vio/layer_{i}" for i in range(2)}
            | {f"moe/min_vio/layer_{i}" for i in range(2)}
            | {f"moe/avg_vio/layer_{i}" for i in range(2)}
            | {
                "moe/max_vio/max",
                "moe/max_vio/avg",
                "moe/min_vio/max",
                "moe/min_vio/avg",
                "moe/avg_vio/max",
                "moe/avg_vio/avg",
                "moe/expert_load_heatmap",
            }
        )
        assert set(metrics.keys()) == expected_keys
        assert metrics["moe/expert_load_heatmap"].__class__.__module__.startswith("PIL.")
    finally:
        set_active_monitor(None)


def test_compute_metrics_empty_returns_empty():
    monitor = MoERouterMonitor(num_experts=4)
    # No record() ever called.
    assert monitor.compute_metrics(current_step=0) == {}


def test_active_monitor_singleton_roundtrip():
    assert get_active_monitor() is None
    m = MoERouterMonitor(num_experts=4)
    set_active_monitor(m)
    try:
        assert get_active_monitor() is m
    finally:
        set_active_monitor(None)
    assert get_active_monitor() is None


def test_hook_noop_when_inactive():
    """The forward hook should be a cheap no-op when no monitor is active."""
    monitor = MoERouterMonitor(num_experts=4)
    model = TwoLayerModel(num_experts=4, top_k=2)
    attach_moe_router_monitor(model, monitor)
    # Do NOT call set_active_monitor — monitor remains inactive.
    assert get_active_monitor() is None
    model.router0.set_next_indices(torch.zeros(4, 2, dtype=torch.long))
    model.router1.set_next_indices(torch.zeros(4, 2, dtype=torch.long))
    model(torch.zeros(4, 8))
    # Counts dict stays empty because record() was never invoked.
    assert monitor._counts == {}


def test_extractor_returning_none_fails_loud():
    """A registered extractor that returns None means the router shape drifted."""
    import pytest

    class _DriftRouter(nn.Module):
        def forward(self, x):
            return x  # not a 3-tuple, extractor returns None

    _DriftRouter.__name__ = "_DriftRouter"
    register_router_extractor("_DriftRouter")(lambda out: None)

    monitor = MoERouterMonitor(num_experts=4)
    model = nn.Sequential(_DriftRouter())
    attach_moe_router_monitor(model, monitor)
    set_active_monitor(monitor)
    try:
        with pytest.raises(AssertionError, match="returned None"):
            model(torch.zeros(2, 4))
    finally:
        set_active_monitor(None)


def test_attach_is_idempotent():
    """Re-attaching to the same model must not duplicate heatmap rows."""
    monitor = MoERouterMonitor(num_experts=4)
    model = TwoLayerModel(num_experts=4, top_k=2)
    attach_moe_router_monitor(model, monitor)
    attach_moe_router_monitor(model, monitor)  # second attach must be a no-op for _layer_order

    set_active_monitor(monitor)
    try:
        model.router0.set_next_indices(torch.zeros(4, 2, dtype=torch.long))
        model.router1.set_next_indices(torch.zeros(4, 2, dtype=torch.long))
        model(torch.zeros(4, 8))
        # Two routers in the model -> exactly 2 rows.
        load = monitor.get_load_matrix(current_step=0)
        assert load.shape == (2, 4), f"expected 2 rows, got {load.shape}"
    finally:
        set_active_monitor(None)


def test_unfired_layers_appear_as_zero_rows():
    """A router registered at attach time but never invoked must not crash.

    Some MoE families have conditionally-routed layers (e.g. capacity gating
    that skips a layer when no tokens are routed to it). The heatmap shape
    must stay stable; the skipped layer just shows up cold.
    """
    monitor = MoERouterMonitor(num_experts=4)
    model = TwoLayerModel(num_experts=4, top_k=2)
    attach_moe_router_monitor(model, monitor)

    set_active_monitor(monitor)
    try:
        # Only router0 fires; router1 stays cold this interval.
        model.router0.set_next_indices(torch.zeros(4, 2, dtype=torch.long))
        model.router0(torch.zeros(4, 8))

        load = monitor.get_load_matrix(current_step=1)
        assert load.shape == (2, 4)
        # Router0 routed every token to expert 0 -> first row [1, 0, 0, 0].
        assert torch.allclose(load[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        # Router1 never fired -> normalized row is all-zero (clamp(min=1.0) keeps it cold).
        assert torch.allclose(load[1], torch.zeros(4))
    finally:
        set_active_monitor(None)


def test_deepseek_v3_style_external_record_path():
    """DeepSeek-V3's top-k math lives in the MoE block, not the router.

    The router class is registered in EXTERNAL_RECORD_ROUTERS so attach
    pre-registers it (stable layer order); the patched MoE block then calls
    record_router_indices(self.gate, topk_indices) explicitly. This test
    simulates that pattern with a fake router class and asserts the monitor
    receives the indices correctly.
    """

    class FakeDeepSeekRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(8, 4))  # 8 experts, hidden=4

        def forward(self, x):
            # DeepSeek-V3 router returns only logits — the MoE block does the topk.
            return x @ self.weight.T

    # Register under the real class name so EXTERNAL_RECORD_ROUTERS picks it up.
    FakeDeepSeekRouter.__name__ = "DeepseekV3TopkRouter"
    assert "DeepseekV3TopkRouter" in EXTERNAL_RECORD_ROUTERS

    class FakeDeepSeekMoE(nn.Module):
        """Mimics DeepseekV3MoE.forward calling record_router_indices."""

        def __init__(self):
            super().__init__()
            self.gate = FakeDeepSeekRouter()

        def forward(self, hidden_states, topk_indices):
            self.gate(hidden_states)  # produces logits (unused here)
            # Patched DeepseekV3MoE.forward calls this after route_tokens_to_experts.
            record_router_indices(self.gate, topk_indices)
            return hidden_states

    monitor = MoERouterMonitor(num_experts=8)
    model = nn.ModuleList([FakeDeepSeekMoE(), FakeDeepSeekMoE()])
    attached = attach_moe_router_monitor(model, monitor)
    # Both router instances pre-registered, no hooks attached (no extractor for this class).
    assert attached == 2, f"expected 2 external-record routers, got {attached}"

    set_active_monitor(monitor)
    try:
        # Layer 0: 6 tokens, top_k=2, every token chooses expert 0.
        # Layer 1: 6 tokens, top_k=2, every token chooses expert 7.
        idx0 = torch.zeros(6, 2, dtype=torch.long)
        idx1 = torch.full((6, 2), 7, dtype=torch.long)
        model[0](torch.zeros(6, 4), idx0)
        model[1](torch.zeros(6, 4), idx1)

        load = monitor.get_load_matrix(current_step=0)
        assert load.shape == (2, 8)
        assert torch.allclose(load[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        assert torch.allclose(load[1], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    finally:
        set_active_monitor(None)


def test_record_router_indices_noop_when_paused_or_inactive():
    """record_router_indices must be a cheap no-op when the monitor is paused/off."""

    class R(nn.Module):
        def forward(self, x):
            return x

    R.__name__ = "DeepseekV3TopkRouter"
    monitor = MoERouterMonitor(num_experts=4)
    router = R()
    attach_moe_router_monitor(nn.ModuleList([router]), monitor)

    # Monitor not active — call must not crash and not record.
    record_router_indices(router, torch.zeros(2, 1, dtype=torch.long))
    assert monitor._counts == {}

    set_active_monitor(monitor)
    try:
        monitor.pause()
        record_router_indices(router, torch.zeros(2, 1, dtype=torch.long))
        assert monitor._counts == {}, "paused monitor must drop the record"
        monitor.resume()
        record_router_indices(router, torch.zeros(2, 1, dtype=torch.long))
        assert monitor._counts != {}, "resumed monitor must accept the record"
    finally:
        set_active_monitor(None)


def test_register_external_record_router_is_idempotent():
    """Registering the same class twice must not break anything."""
    before = len(EXTERNAL_RECORD_ROUTERS)
    register_external_record_router("DeepseekV3TopkRouter")  # already there
    register_external_record_router("DeepseekV3TopkRouter")
    assert len(EXTERNAL_RECORD_ROUTERS) == before


def test_attach_returns_zero_when_no_routers():
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin(x)

    monitor = MoERouterMonitor(num_experts=4)
    assert attach_moe_router_monitor(Plain(), monitor) == 0


def test_qwen3_extractor_handles_non_tensor_output():
    """Defensive: malformed router output must not crash the extractor."""
    extract = ROUTER_EXTRACTORS["Qwen3MoeTopKRouter"]
    assert extract(None) is None
    assert extract((torch.zeros(1),)) is None  # too short
    assert extract((torch.zeros(1), torch.zeros(1), torch.zeros(1, dtype=torch.float))) is None
    indices = torch.zeros(2, 2, dtype=torch.long)
    assert extract((torch.zeros(1), torch.zeros(1), indices)) is indices


def test_module_dunder_safety():
    # moe_monitor.__name__ sanity — guards against stray import-time errors
    # when the module is loaded by callbacks early in trainer init.
    assert moe_monitor.MoERouterMonitor.__name__ == "MoERouterMonitor"
