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

"""MoE Router Load Balance Monitor.

Provides tools to monitor expert load distribution across MoE layers during training.
Logs a [num_moe_layers, num_experts] heatmap and per-layer violation metrics to wandb.

Architecture:
    1. Router modules (e.g. PatchQwen3MoeTopKRouter) register ``router_forward_hook``
       in their ``__init__``. This happens at model construction time.
    2. The hook is a no-op (single ``if`` check) until a ``MoERouterMonitor`` is
       created and activated via ``set_active_monitor()``. This is done by the
       trainer when ``moe_load_balance_monitor_interval > 0``.
    3. Once active, each router forward accumulates token-to-expert counts on device
       (no CPU sync). At the configured interval, ``get_load_matrix()`` moves counts
       to CPU (single sync) and produces a normalized frequency matrix.
    4. ``compute_vio()`` derives max/min/avg violation metrics from the matrix.

This design avoids FSDP compatibility issues — hooks are on the original router modules,
not discovered through the FSDP wrapper at runtime.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Global active monitor singleton.
# Router forward hooks check this; when None, the hook is a no-op.
# Activated by the trainer/callback via set_active_monitor().
# ---------------------------------------------------------------------------
_active_monitor: Optional["MoERouterMonitor"] = None


def get_active_monitor() -> Optional["MoERouterMonitor"]:
    """Return the currently active MoE router monitor, or None if disabled."""
    return _active_monitor


def set_active_monitor(monitor: Optional["MoERouterMonitor"]) -> None:
    """Activate or deactivate the global MoE router monitor.

    Args:
        monitor: A ``MoERouterMonitor`` instance to activate, or ``None`` to deactivate.
    """
    global _active_monitor
    _active_monitor = monitor


def router_forward_hook(module: nn.Module, input, output):
    """PyTorch forward hook registered on MoE router modules at construction time.

    When no monitor is active (``_active_monitor is None``), this is effectively
    a no-op — just a single ``if`` check per router forward, with negligible overhead.

    Expected ``output`` format: ``(router_logits, router_scores, router_indices)``
    where ``router_indices`` has shape ``[num_tokens, top_k]``.
    """
    if _active_monitor is None:
        return
    # router_indices: [num_tokens, top_k] — the selected expert indices per token
    _active_monitor.record(module, output[2])


class MoERouterMonitor:
    """Monitors MoE expert load distribution via router forward hooks.

    Router modules register ``router_forward_hook`` at construction time (in the
    model patch, e.g. ``PatchQwen3MoeTopKRouter.__init__``). The hook is a no-op
    until a monitor is activated via ``set_active_monitor()``. This avoids FSDP
    compatibility issues and module-walking entirely.

    Typical usage (via ``MoERouterMonitorCallback``)::

        # At train begin:
        monitor = MoERouterMonitor(num_experts=128)
        set_active_monitor(monitor)

        # ... training runs, hooks auto-accumulate counts ...

        # At logging interval:
        load_matrix = monitor.get_load_matrix(current_step=step)
        image = monitor.create_wandb_image(load_matrix)
        vio = MoERouterMonitor.compute_vio(load_matrix)

        # At train end:
        set_active_monitor(None)

    Attributes:
        num_experts: Total number of experts in the MoE model.
    """

    def __init__(self, num_experts: int):
        """Initialize the monitor.

        Args:
            num_experts: Number of experts per MoE layer (e.g. 128 for Qwen3-30B-A3B).
        """
        self.num_experts = num_experts
        # Maps module id -> accumulated token counts tensor (on device, shape [num_experts]).
        # Using id(module) as key so we can track each router instance independently.
        self._counts: Dict[int, torch.Tensor] = {}
        # Ordered list of module ids, preserving layer discovery order (layer 0 first).
        self._layer_order: list = []
        # Step range tracking for heatmap captions.
        self._accumulate_start_step: int = 0
        self._accumulate_end_step: int = 0
        self._last_step_range: tuple = (0, 0)

    def record(self, module: nn.Module, router_indices: torch.Tensor):
        """Record expert selections from a single router forward pass.

        Called by ``router_forward_hook``. Accumulates on device (no CPU sync).

        Args:
            module: The router module instance (used as key via ``id(module)``).
            router_indices: Expert indices of shape ``[num_tokens, top_k]``.
        """
        mid = id(module)
        # Lazily initialize the counts tensor for new router modules.
        # The first forward pass through each router auto-registers it.
        if mid not in self._counts:
            self._layer_order.append(mid)
            device = router_indices.device
            self._counts[mid] = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        # Count how many tokens were routed to each expert.
        counts = torch.bincount(
            router_indices.reshape(-1).to(torch.long),
            minlength=self.num_experts,
        )
        # Accumulate on device — detach to avoid graph retention.
        self._counts[mid] += counts.detach()

    def get_load_matrix(self, current_step: int = 0) -> torch.Tensor:
        """Return the normalized load matrix and reset accumulated counts.

        This is the **only** method that moves data from device to CPU, causing a
        single CUDA sync. Should be called only at the logging interval (not every step).

        Args:
            current_step: The current global training step, used to record
                the accumulation range for heatmap captions.

        Returns:
            A float tensor of shape ``[num_moe_layers, num_experts]`` where each
            row sums to 1.0, representing the fraction of tokens routed to each
            expert in that layer.
        """
        if not self._counts:
            return torch.zeros(0, self.num_experts)
        self._accumulate_end_step = current_step
        # Stack counts in layer order and move to CPU (single sync point).
        matrix = torch.stack([self._counts[mid] for mid in self._layer_order]).float().cpu()
        # Normalize each row (layer) to sum to 1.0.
        row_sums = matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
        matrix = matrix / row_sums
        # Save step range for caption, then reset for next interval.
        self._last_step_range = (self._accumulate_start_step, self._accumulate_end_step)
        self._reset_counts()
        self._accumulate_start_step = current_step + 1
        return matrix

    def create_wandb_image(self, load_matrix: torch.Tensor, caption: str = None):
        """Create a wandb.Image heatmap from the normalized load matrix.

        The heatmap has expert index on the X axis, MoE layer index on the Y axis,
        and color intensity representing normalized token frequency.

        Args:
            load_matrix: Normalized ``[num_moe_layers, num_experts]`` tensor from
                ``get_load_matrix()``.
            caption: Optional caption override. If ``None``, auto-generates from
                the accumulated step range (e.g. "Steps 11-20").

        Returns:
            A ``wandb.Image`` object ready to be logged via ``wandb.log()``.
        """
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        if caption is None:
            start, end = self._last_step_range
            caption = f"Steps {start}-{end}"

        fig, ax = plt.subplots(figsize=(max(8, load_matrix.shape[1] * 0.1), max(4, load_matrix.shape[0] * 0.2)))
        im = ax.imshow(load_matrix.numpy(), aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("Expert Index")
        ax.set_ylabel("MoE Layer Index")
        ax.set_title(f"MoE Expert Load Distribution ({caption})")
        fig.colorbar(im, ax=ax, label="Normalized Token Frequency")
        fig.tight_layout()

        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)

            from PIL import Image

            image = wandb.Image(Image.open(buf), caption=caption)
            return image
        finally:
            buf.close()

    @staticmethod
    def compute_vio(load_matrix: torch.Tensor) -> dict:
        """Compute per-layer load balance violation metrics.

        Given a normalized load matrix where each row sums to 1.0, computes the
        deviation from uniform distribution for each layer:

            deviation = normalized_freq * num_experts - 1

        Under a perfectly uniform distribution, every expert gets ``1/num_experts``
        of the tokens, so ``deviation = 0`` everywhere. Metrics:

        - **max_vio**: ``deviation.max()`` per layer. Measures the most *overloaded*
          expert. Range: ``[0, num_experts - 1]``. Closer to 0 = more balanced.
        - **min_vio**: ``deviation.min()`` per layer. Measures the most *underloaded*
          expert. Range: ``[-1, 0]``. Closer to 0 = more balanced.
        - **avg_vio**: ``|deviation|.mean()`` per layer. Measures the average absolute
          deviation from uniform. Range: ``[0, ...]``. Closer to 0 = more balanced.

        Args:
            load_matrix: Normalized ``[num_moe_layers, num_experts]`` tensor from
                ``get_load_matrix()``.

        Returns:
            Dict with keys ``"max_vio"``, ``"min_vio"``, ``"avg_vio"``, each a
            tensor of shape ``[num_moe_layers]``.
        """
        num_experts = load_matrix.shape[1]
        # deviation from uniform: 0 means perfectly balanced
        deviation = load_matrix * num_experts - 1.0
        return {
            "max_vio": deviation.max(dim=1).values,
            "min_vio": deviation.min(dim=1).values,
            "avg_vio": deviation.abs().mean(dim=1),
        }

    def _reset_counts(self):
        """Zero out all accumulated counts (on device) for the next interval."""
        for mid in self._counts:
            self._counts[mid].zero_()
