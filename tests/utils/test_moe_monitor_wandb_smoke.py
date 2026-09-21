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

"""End-to-end smoke test of the MoE monitor through ``wandb.log`` (offline mode).

Validates the full pipeline without needing a real MoE model or GPU:

    fake routers -> attach_moe_router_monitor -> hooks fire -> compute_metrics
        -> wandb.log -> run.summary -> assertions

Asserts that ``wandb.log`` accepts every entry the monitor produces (no
exceptions raised) and that the latest scalar values reflect the non-uniform
routing we drove through the fake routers.

Real-model e2e via mlx-worker is documented in ``tasks/smoke/moe_monitor_smoke.md``.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest
import torch
import torch.nn as nn

from veomni.utils.moe_monitor import (
    MoERouterMonitor,
    attach_moe_router_monitor,
    set_active_monitor,
)


wandb = pytest.importorskip("wandb")


class FakeQwen3Router(nn.Module):
    """Stand-in with the same name as ``Qwen3MoeTopKRouter`` so the registry finds it."""

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self._next_indices: torch.Tensor | None = None

    def set_next_indices(self, idx: torch.Tensor) -> None:
        self._next_indices = idx

    def forward(self, hidden_states: torch.Tensor):
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        indices = self._next_indices.to(device)
        logits = torch.zeros(num_tokens, self.num_experts, device=device)
        scores = torch.zeros(num_tokens, self.top_k, device=device)
        return logits, scores, indices


FakeQwen3Router.__name__ = "Qwen3MoeTopKRouter"


class TwoLayer(nn.Module):
    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.r0 = FakeQwen3Router(num_experts, top_k)
        self.r1 = FakeQwen3Router(num_experts, top_k)

    def forward(self, x):
        self.r0(x)
        self.r1(x)
        return x


def test_full_pipeline_through_wandb_offline():
    tmpdir = tempfile.mkdtemp(prefix="moe_monitor_smoke_")
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = tmpdir
    os.environ["WANDB_SILENT"] = "true"

    run = wandb.init(project="moe-monitor-smoke", name="smoke", reinit=True)
    try:
        num_experts, top_k = 8, 2
        model = TwoLayer(num_experts, top_k)
        monitor = MoERouterMonitor(num_experts=num_experts)
        attached = attach_moe_router_monitor(model, monitor)
        assert attached == 2
        set_active_monitor(monitor)

        # Drive 4 steps. Layer 1 collapses to expert num_experts-1 every step,
        # so max_vio on that layer must be (num_experts - 1) after normalization.
        tokens = 8
        for step in range(1, 5):
            idx0 = torch.zeros(tokens, top_k, dtype=torch.long)
            idx0[: tokens // 4, 0] = 1
            idx1 = torch.full((tokens, top_k), num_experts - 1, dtype=torch.long)
            model.r0.set_next_indices(idx0)
            model.r1.set_next_indices(idx1)
            model(torch.zeros(tokens, 4))

            metrics = monitor.compute_metrics(current_step=step)
            wandb_metrics = {
                k: (wandb.Image(v) if k.endswith("expert_load_heatmap") else v) for k, v in metrics.items()
            }
            wandb.log(wandb_metrics, step=step)

        # ``wandb.log`` accepted every entry without exception (we got here).
        # The latest scalar values land in run.summary in real time.
        summary = dict(run.summary)
        required_scalars = {
            "moe/max_vio/max",
            "moe/max_vio/avg",
            "moe/min_vio/max",
            "moe/avg_vio/max",
            "moe/max_vio/layer_0",
            "moe/max_vio/layer_1",
        }
        missing = required_scalars - summary.keys()
        assert not missing, f"missing scalar keys in run.summary: {missing}"

        # Layer 1 was fully collapsed -> max_vio == num_experts - 1 == 7.
        assert summary["moe/max_vio/layer_1"] == pytest.approx(num_experts - 1)
        # max_vio/max aggregates across layers, so it also equals num_experts - 1.
        assert summary["moe/max_vio/max"] == pytest.approx(num_experts - 1)
        # Heatmap key is recorded in summary as a media descriptor.
        assert "moe/expert_load_heatmap" in summary
    finally:
        wandb.finish()
        set_active_monitor(None)
        shutil.rmtree(tmpdir, ignore_errors=True)
        os.environ.pop("WANDB_MODE", None)
        os.environ.pop("WANDB_DIR", None)
        os.environ.pop("WANDB_SILENT", None)
