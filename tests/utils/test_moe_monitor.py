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

import sys
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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


def test_qwen3_5_router_reuses_qwen_tuple_extractor():
    assert ROUTER_EXTRACTORS["Qwen3_5MoeTopKRouter"] is ROUTER_EXTRACTORS["Qwen3MoeTopKRouter"]
    indices = torch.tensor([[1, 3]], dtype=torch.long)
    output = (torch.zeros(1, 4), torch.zeros(1, 2), indices)
    assert ROUTER_EXTRACTORS["Qwen3_5MoeTopKRouter"](output) is indices


def _register_router_counts(monitor: MoERouterMonitor, num_layers: int = 1) -> list[nn.Module]:
    routers = [nn.Identity() for _ in range(num_layers)]
    for layer_index, router in enumerate(routers):
        monitor._register_layer(router)
        monitor.record(router, torch.tensor([[layer_index % monitor.num_experts]], dtype=torch.long))
    return routers


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((-1, (1, 1), (1, 1), 0, 0), "layer_index"),
        ((1, (1, 1), (1, 1), 0, 0), "layer_index"),
        ((0, (1, 1), (2,), 0, 0), "shape"),
        ((0, (1, -1), (0, 0), 0, 0), "non-negative"),
        ((0, (2, 0), (1, 0), 0, 0), "conserve"),
        ((0, (1, 1), (1, 1), -1, 0), "active_replicas"),
        ((0, (1, 1), (1, 1), 0, -1), "moved_tokens"),
    ],
)
def test_record_ep_balance_validation(args, match):
    monitor = MoERouterMonitor(num_experts=4)
    _register_router_counts(monitor)
    with pytest.raises(ValueError, match=match):
        monitor.record_ep_balance(*args)


def test_record_ep_balance_requires_consistent_rank_width_across_layers():
    monitor = MoERouterMonitor(num_experts=4)
    _register_router_counts(monitor, num_layers=2)
    monitor.record_ep_balance(0, (4, 0), (2, 2), 1, 2)
    with pytest.raises(ValueError, match="rank.*size"):
        monitor.record_ep_balance(1, (3, 0, 0), (1, 1, 1), 1, 2)


def test_record_ep_balance_respects_pause_and_sticky_disable():
    monitor = MoERouterMonitor(num_experts=4)
    _register_router_counts(monitor)

    monitor.pause()
    monitor.record_ep_balance(-1, (-1,), (2, 3), -1, -1)
    assert monitor._ep_rank_loads_before == {}

    monitor.resume()
    monitor.disable()
    monitor.resume()
    monitor.record_ep_balance(-1, (-1,), (2, 3), -1, -1)
    assert monitor._ep_rank_loads_before == {}


def test_ep_balance_metrics_accumulate_exact_formulas_keys_and_reset(monkeypatch):
    monitor = MoERouterMonitor(num_experts=4)
    _register_router_counts(monitor, num_layers=2)
    monitor.record_ep_balance(0, (8, 0), (4, 4), 1, 4)
    monitor.record_ep_balance(0, (2, 2), (1, 3), 2, 1)
    monitor.record_ep_balance(1, (0, 0), (0, 0), 0, 0)

    heatmap_calls = []

    def fake_rank_heatmap(matrix, stage):
        heatmap_calls.append((matrix.clone(), stage))
        return f"{stage}-image"

    monkeypatch.setattr(monitor, "build_ep_rank_heatmap_image", fake_rank_heatmap, raising=False)
    metrics = monitor.compute_metrics(current_step=7)

    required = {
        "moe/ep_rank_load_before_heatmap",
        "moe/ep_rank_load_after_heatmap",
        "moe/ep_rank_imbalance_before/layer_0",
        "moe/ep_rank_imbalance_before/layer_1",
        "moe/ep_rank_imbalance_before/max",
        "moe/ep_rank_imbalance_before/avg",
        "moe/ep_rank_imbalance_after/layer_0",
        "moe/ep_rank_imbalance_after/layer_1",
        "moe/ep_rank_imbalance_after/max",
        "moe/ep_rank_imbalance_after/avg",
        "moe/ep_active_replicas/layer_0",
        "moe/ep_active_replicas/sum",
        "moe/ep_moved_tokens/layer_0",
        "moe/ep_moved_tokens/sum",
        "moe/ep_moved_token_fraction/layer_0",
        "moe/ep_moved_token_fraction/layer_1",
        "moe/ep_moved_token_fraction/max",
        "moe/ep_moved_token_fraction/avg",
    }
    assert required <= metrics.keys()
    assert metrics["moe/ep_rank_load_before_heatmap"] == "before-image"
    assert metrics["moe/ep_rank_load_after_heatmap"] == "after-image"
    assert [stage for _, stage in heatmap_calls] == ["before", "after"]
    assert torch.equal(heatmap_calls[0][0], torch.tensor([[10, 2], [0, 0]]))
    assert torch.equal(heatmap_calls[1][0], torch.tensor([[5, 7], [0, 0]]))

    assert metrics["moe/ep_rank_imbalance_before/layer_0"] == pytest.approx(2 / 3)
    assert metrics["moe/ep_rank_imbalance_before/layer_1"] == 0.0
    assert metrics["moe/ep_rank_imbalance_before/max"] == pytest.approx(2 / 3)
    assert metrics["moe/ep_rank_imbalance_before/avg"] == pytest.approx(1 / 3)
    assert metrics["moe/ep_rank_imbalance_after/layer_0"] == pytest.approx(1 / 6)
    assert metrics["moe/ep_rank_imbalance_after/layer_1"] == 0.0
    assert metrics["moe/ep_active_replicas/layer_0"] == 3
    assert metrics["moe/ep_active_replicas/sum"] == 3
    assert metrics["moe/ep_moved_tokens/layer_0"] == 5
    assert metrics["moe/ep_moved_tokens/sum"] == 5
    assert metrics["moe/ep_moved_token_fraction/layer_0"] == pytest.approx(5 / 12)
    assert metrics["moe/ep_moved_token_fraction/layer_1"] == 0.0
    assert metrics["moe/ep_moved_token_fraction/avg"] == pytest.approx(5 / 24)

    assert monitor._ep_rank_loads_before == {}
    assert monitor._ep_rank_loads_after == {}
    assert monitor._ep_stats == {}
    second_interval = monitor.compute_metrics(current_step=8)
    assert not any("/ep_" in key for key in second_interval)


def test_ep_balance_accumulates_on_corresponding_router_count_device():
    monitor = MoERouterMonitor(num_experts=4)
    routers = _register_router_counts(monitor)
    count_device = monitor._counts[id(routers[0])].device
    monitor.record_ep_balance(0, (4, 0), (2, 2), 1, 2)
    assert monitor._ep_rank_loads_before[0].device == count_device
    assert monitor._ep_rank_loads_after[0].device == count_device
    assert monitor._ep_stats[0].device == count_device


def _two_rank_ep_monitor_worker(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        monitor = MoERouterMonitor(num_experts=4, dp_group=dist.group.WORLD)
        _register_router_counts(monitor)
        if rank == 0:
            monitor.record_ep_balance(0, (4, 0), (2, 2), 1, 2)
        else:
            monitor.record_ep_balance(0, (0, 6), (3, 3), 2, 3)

        metrics = monitor.compute_metrics(current_step=3, format_only_on=rank == 0)
        if rank == 0:
            assert metrics["moe/ep_rank_imbalance_before/layer_0"] == pytest.approx(0.2)
            assert metrics["moe/ep_rank_imbalance_after/layer_0"] == 0.0
            assert metrics["moe/ep_active_replicas/sum"] == 3
            assert metrics["moe/ep_moved_tokens/sum"] == 5
            assert metrics["moe/ep_moved_token_fraction/avg"] == pytest.approx(0.5)
        else:
            assert metrics == {}
        assert monitor._ep_rank_loads_before == {}
        assert monitor._ep_rank_loads_after == {}
        assert monitor._ep_stats == {}
    finally:
        set_active_monitor(None)
        dist.destroy_process_group()


def test_ep_balance_real_world2_gloo_dp_reduction_and_reset(tmp_path):
    rendezvous = f"file://{tmp_path / 'moe-monitor-world2'}"
    mp.spawn(_two_rank_ep_monitor_worker, args=(2, rendezvous), nprocs=2, join=True)


def _asymmetric_ep_monitor_worker(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        monitor = MoERouterMonitor(num_experts=4, dp_group=dist.group.WORLD)
        _register_router_counts(monitor)
        if rank == 0:
            monitor.record_ep_balance(0, (4, 0), (2, 2), 1, 2)

        metrics = monitor.compute_metrics(current_step=3, format_only_on=rank == 0)
        if rank == 0:
            assert metrics["moe/ep_rank_imbalance_before/layer_0"] == 1.0
            assert metrics["moe/ep_rank_imbalance_after/layer_0"] == 0.0
            assert metrics["moe/ep_active_replicas/sum"] == 1
            assert metrics["moe/ep_moved_tokens/sum"] == 2
            assert metrics["moe/ep_moved_token_fraction/avg"] == pytest.approx(0.5)
        else:
            assert metrics == {}
        assert monitor._ep_rank_loads_before == {}
        assert monitor._ep_rank_loads_after == {}
        assert monitor._ep_stats == {}
    finally:
        set_active_monitor(None)
        dist.destroy_process_group()


def test_ep_balance_real_world2_asymmetric_record_uses_matching_collectives(tmp_path):
    rendezvous = f"file://{tmp_path / 'moe-monitor-asymmetric-world2'}"
    process_context = mp.spawn(
        _asymmetric_ep_monitor_worker,
        args=(2, rendezvous),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 30
    try:
        while not process_context.join(timeout=0.5):
            if time.monotonic() >= deadline:
                pytest.fail("Asymmetric physical telemetry caused mismatched collectives.")
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_metric_format_failure_preserves_interval_for_exact_retry(monkeypatch):
    monitor = MoERouterMonitor(num_experts=4)
    routers = _register_router_counts(monitor)
    monitor.record_ep_balance(0, (4, 0), (2, 2), 1, 2)
    logical_before = monitor._counts[id(routers[0])].clone()
    physical_before = monitor._ep_rank_loads_before[0].clone()
    physical_after = monitor._ep_rank_loads_after[0].clone()
    stats_before = monitor._ep_stats[0].clone()

    def fail_heatmap(*args, **kwargs):
        raise RuntimeError("injected format failure")

    monkeypatch.setattr(monitor, "build_heatmap_image", fail_heatmap)
    with pytest.raises(RuntimeError, match="injected format failure"):
        monitor.compute_metrics(current_step=4)

    assert torch.equal(monitor._counts[id(routers[0])], logical_before)
    assert torch.equal(monitor._ep_rank_loads_before[0], physical_before)
    assert torch.equal(monitor._ep_rank_loads_after[0], physical_after)
    assert torch.equal(monitor._ep_stats[0], stats_before)
    assert monitor._accumulate_start_step == 0
    assert monitor._last_step_range == (0, 0)

    monkeypatch.setattr(monitor, "build_heatmap_image", lambda *args, **kwargs: "logical-image")
    monkeypatch.setattr(
        monitor,
        "build_ep_rank_heatmap_image",
        lambda matrix, stage: f"{stage}-image",
    )
    metrics = monitor.compute_metrics(current_step=4)
    assert metrics["moe/ep_moved_tokens/sum"] == 2
    assert metrics["moe/ep_total_routed_tokens/sum"] == 4
    assert monitor._counts[id(routers[0])].sum().item() == 0
    assert monitor._ep_rank_loads_before == {}
    assert monitor._ep_rank_loads_after == {}
    assert monitor._ep_stats == {}
    assert monitor._accumulate_start_step == 5
    assert monitor._last_step_range == (0, 4)


def _coordinated_host_transfer_failure_worker(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    original_tensor_cpu = torch.Tensor.cpu
    try:
        monitor = MoERouterMonitor(num_experts=4, dp_group=dist.group.WORLD)
        routers = _register_router_counts(monitor)
        if rank == 0:
            monitor.record_ep_balance(0, (4, 0), (2, 2), 1, 2)
        logical_before = monitor._counts[id(routers[0])].clone()
        ep_before = {layer: value.clone() for layer, value in monitor._ep_rank_loads_before.items()}
        ep_after = {layer: value.clone() for layer, value in monitor._ep_rank_loads_after.items()}
        stats_before = {layer: value.clone() for layer, value in monitor._ep_stats.items()}

        if rank == 0:
            cpu_calls = 0

            def fail_second_host_transfer(tensor, *args, **kwargs):
                nonlocal cpu_calls
                cpu_calls += 1
                if cpu_calls == 2:
                    raise RuntimeError("injected formatter host transfer failure")
                return original_tensor_cpu(tensor, *args, **kwargs)

            torch.Tensor.cpu = fail_second_host_transfer

        expected_error = "injected formatter host transfer failure" if rank == 0 else r"another DP\+SP/FSDP"
        with pytest.raises(RuntimeError, match=expected_error):
            monitor.compute_metrics(current_step=6, format_only_on=rank == 0)
        torch.Tensor.cpu = original_tensor_cpu

        assert torch.equal(monitor._counts[id(routers[0])], logical_before)
        assert monitor._ep_rank_loads_before.keys() == ep_before.keys()
        assert monitor._ep_rank_loads_after.keys() == ep_after.keys()
        assert monitor._ep_stats.keys() == stats_before.keys()
        for layer, value in ep_before.items():
            assert torch.equal(monitor._ep_rank_loads_before[layer], value)
            assert torch.equal(monitor._ep_rank_loads_after[layer], ep_after[layer])
            assert torch.equal(monitor._ep_stats[layer], stats_before[layer])
        assert monitor._accumulate_start_step == 0
        assert monitor._last_step_range == (0, 0)

        metrics = monitor.compute_metrics(current_step=6, format_only_on=rank == 0)
        if rank == 0:
            assert metrics["moe/ep_moved_tokens/sum"] == 2
            assert metrics["moe/ep_total_routed_tokens/sum"] == 4
            assert metrics["moe/ep_moved_token_fraction/avg"] == pytest.approx(0.5)
        else:
            assert metrics == {}
        assert monitor._counts[id(routers[0])].sum().item() == 0
        assert monitor._ep_rank_loads_before == {}
        assert monitor._ep_rank_loads_after == {}
        assert monitor._ep_stats == {}
        assert monitor._accumulate_start_step == 7
        assert monitor._last_step_range == (0, 6)
    finally:
        torch.Tensor.cpu = original_tensor_cpu
        set_active_monitor(None)
        dist.destroy_process_group()


def test_real_world2_formatter_host_transfer_failure_is_coordinated_and_retryable(tmp_path):
    rendezvous = f"file://{tmp_path / 'moe-monitor-host-failure-world2'}"
    process_context = mp.spawn(
        _coordinated_host_transfer_failure_worker,
        args=(2, rendezvous),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 30
    try:
        while not process_context.join(timeout=0.5):
            if time.monotonic() >= deadline:
                pytest.fail("Formatter host-transfer failure was not coordinated with non-formatting peers.")
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def _patch_callback_base_init(monkeypatch):
    from veomni.trainer.callbacks.base import Callback

    def fake_init(callback, trainer):
        callback.trainer = trainer
        callback.parallel_state = trainer.parallel_state

    monkeypatch.setattr(Callback, "__init__", fake_init)


def _monitor_callback_trainer(*, config, wandb_enabled=False, global_rank=0):
    return SimpleNamespace(
        args=SimpleNamespace(
            train=SimpleNamespace(
                moe_load_balance_monitor_interval=2,
                global_rank=global_rank,
                wandb=SimpleNamespace(enable=wandb_enabled),
            )
        ),
        model_config=config,
        parallel_state=SimpleNamespace(ep_enabled=True, ep_size=2, fsdp_group=None),
        step_env_metrics={"environment/tokens": 10},
        model=nn.Identity(),
    )


def test_monitor_callback_resolves_nested_qwen3_5_text_config(monkeypatch):
    from veomni.trainer.callbacks.trace_callback import MoERouterMonitorCallback

    _patch_callback_base_init(monkeypatch)
    trainer = _monitor_callback_trainer(config=SimpleNamespace(text_config=SimpleNamespace(num_experts=8)))
    callback = MoERouterMonitorCallback(trainer)
    try:
        assert callback.monitor is not None
        assert callback.monitor.num_experts == 8
    finally:
        callback.on_train_end(SimpleNamespace())


def test_monitor_callback_collects_without_wandb_and_propagates_only_scalars(monkeypatch):
    from veomni.trainer.callbacks.trace_callback import MoERouterMonitorCallback

    trainer = _monitor_callback_trainer(config=SimpleNamespace(num_experts=4), wandb_enabled=False)
    callback = MoERouterMonitorCallback.__new__(MoERouterMonitorCallback)
    callback.trainer = trainer
    calls = []

    class FakeMonitor:
        _last_step_range = (0, 2)

        def compute_metrics(self, current_step, format_only_on):
            calls.append((current_step, format_only_on))
            return {
                "moe/ep_rank_imbalance_before/avg": 0.5,
                "moe/ep_rank_load_before_heatmap": object(),
            }

    callback.monitor = FakeMonitor()
    callback.on_step_end(SimpleNamespace(global_step=2))
    assert calls == [(2, True)]
    assert trainer.step_env_metrics == {
        "environment/tokens": 10,
        "moe/ep_rank_imbalance_before/avg": 0.5,
    }


def test_monitor_callback_nonzero_rank_collects_and_resets_without_formatting():
    from veomni.trainer.callbacks.trace_callback import MoERouterMonitorCallback

    trainer = _monitor_callback_trainer(
        config=SimpleNamespace(num_experts=4),
        wandb_enabled=False,
        global_rank=1,
    )
    callback = MoERouterMonitorCallback.__new__(MoERouterMonitorCallback)
    callback.trainer = trainer
    calls = []

    class FakeMonitor:
        def compute_metrics(self, current_step, format_only_on):
            calls.append((current_step, format_only_on))
            return {}

    callback.monitor = FakeMonitor()
    callback.on_step_end(SimpleNamespace(global_step=2))
    assert calls == [(2, False)]


def test_monitor_callback_wandb_directly_logs_only_heatmaps_and_handles_missing_logical_keys(monkeypatch):
    from veomni.trainer.callbacks.trace_callback import MoERouterMonitorCallback

    trainer = _monitor_callback_trainer(config=SimpleNamespace(num_experts=4), wandb_enabled=True)
    callback = MoERouterMonitorCallback.__new__(MoERouterMonitorCallback)
    callback.trainer = trainer
    image = object()

    class FakeMonitor:
        _last_step_range = (1, 2)

        def compute_metrics(self, current_step, format_only_on):
            return {
                "moe/ep_rank_imbalance_before/avg": 0.5,
                "moe/ep_rank_load_before_heatmap": image,
            }

    logged = []
    fake_wandb = SimpleNamespace(
        Image=lambda value, caption: (value, caption),
        log=lambda payload, step, commit: logged.append((payload, step, commit)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    callback.monitor = FakeMonitor()
    callback.on_step_end(SimpleNamespace(global_step=2))

    assert trainer.step_env_metrics["moe/ep_rank_imbalance_before/avg"] == 0.5
    assert logged == [
        (
            {"moe/ep_rank_load_before_heatmap": (image, "Steps 1-2")},
            2,
            False,
        )
    ]


def test_monitor_callback_on_train_end_always_clears_singleton():
    from veomni.trainer.callbacks.trace_callback import MoERouterMonitorCallback

    stale_monitor = MoERouterMonitor(num_experts=4)
    set_active_monitor(stale_monitor)
    callback = MoERouterMonitorCallback.__new__(MoERouterMonitorCallback)
    callback.monitor = None
    try:
        callback.on_train_end(SimpleNamespace())
        assert get_active_monitor() is None
    finally:
        set_active_monitor(None)


def test_base_trainer_orders_monitor_after_metric_producers_before_wandb(monkeypatch):
    import veomni.trainer.base as trainer_base

    class FakeCallback:
        def __init__(self, name):
            self.name = name

    callback_names = {
        "EnvironMeterCallback": "environment",
        "TqdmCallback": "tqdm",
        "WandbTraceCallback": "wandb",
        "ProfileTraceCallback": "profile",
        "CheckpointerCallback": "checkpointer",
        "HuggingfaceCkptCallback": "hf_checkpoint",
        "HFLoraCkptCallback": "hf_lora_checkpoint",
        "EvaluateCallback": "evaluate",
        "MoERouterMonitorCallback": "moe_monitor",
        "ChannelLossCallback": "channel_loss",
    }
    for symbol, name in callback_names.items():
        monkeypatch.setattr(trainer_base, symbol, lambda trainer, name=name: FakeCallback(name))

    trainer = trainer_base.BaseTrainer.__new__(trainer_base.BaseTrainer)
    trainer.args = SimpleNamespace(model=SimpleNamespace(lora_config={}))
    trainer._init_callbacks()
    order = [callback.name for callback in trainer._callbacks]
    assert order.index("environment") < order.index("moe_monitor")
    assert order.index("channel_loss") < order.index("moe_monitor")
    assert order.index("moe_monitor") < order.index("wandb")


def test_module_dunder_safety():
    # moe_monitor.__name__ sanity — guards against stray import-time errors
    # when the module is loaded by callbacks early in trainer init.
    assert moe_monitor.MoERouterMonitor.__name__ == "MoERouterMonitor"
