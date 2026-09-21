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

Monitors expert load distribution across MoE layers during training. Produces a
``[num_moe_layers, num_experts]`` heatmap and per-layer violation metrics.

Architecture
------------
The monitor is **driver-attached**, not patch-registered. The trainer (VeOmni
``MoERouterMonitorCallback`` or verl ``VeOmniEngine``) constructs a
:class:`MoERouterMonitor`, then calls :func:`attach_moe_router_monitor` once on
the fully-constructed model. That function walks the model, finds every
recognized router/gate module via :data:`ROUTER_EXTRACTORS`, and registers a
forward hook on each. No model-patch code needs to know about the monitor.

Each registered hook is gated by :func:`get_active_monitor` so the cost when
disabled is one ``if`` per router forward.

At logging cadence the caller invokes :meth:`MoERouterMonitor.compute_metrics`
to get a plain dict of scalars + a PIL heatmap; the caller wraps it for its
logging backend (wandb / tensorboard / mlflow / verl ``Tracking``).

Adding a new model family
-------------------------
**Case A — router forward output exposes top-k indices** (Qwen3 family).
Register an extractor::

    @register_router_extractor("MyNewRouter")
    def _extract(output):
        return output["indices"]  # or output[2], etc.

**Case B — top-k math lives downstream of the router** (DeepSeek-V3 family).
The router only produces logits; the actual top-k is computed inside the
patched MoE block (with sigmoid + bias correction + group routing for
DeepSeek-V3). For these families:

1. Call :func:`register_external_record_router` for the router class so
   :func:`attach_moe_router_monitor` pre-registers the layer (stable order
   in the heatmap).
2. Insert one line into the patched MoE block's ``forward`` right after the
   indices are computed::

       record_router_indices(self.gate, topk_indices)

   Symmetric to :func:`maybe_replay_indices` in ``moe_router_replay.py``.

Do not try to recompute the top-k inside this module — the gating math is
family-specific and prone to drift.
"""

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from .logging import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Global active monitor singleton.
# Router forward hooks check this; when None, the hook is a no-op.
# ---------------------------------------------------------------------------
_active_monitor: Optional["MoERouterMonitor"] = None


def get_active_monitor() -> Optional["MoERouterMonitor"]:
    """Return the currently active MoE router monitor, or None if disabled."""
    return _active_monitor


def set_active_monitor(monitor: Optional["MoERouterMonitor"]) -> None:
    """Activate or deactivate the global MoE router monitor."""
    global _active_monitor
    _active_monitor = monitor


# ---------------------------------------------------------------------------
# Router class registry. Maps class name (string, to avoid importing patched
# classes at module load time) -> extractor returning router indices.
# ---------------------------------------------------------------------------
RouterExtractor = Callable[[Any], Optional[torch.Tensor]]
ROUTER_EXTRACTORS: Dict[str, RouterExtractor] = {}


def register_router_extractor(class_name: str) -> Callable[[RouterExtractor], RouterExtractor]:
    """Decorator: register an extractor for a router module class by name.

    The extractor receives the router module's forward output and must return
    a tensor of expert indices with shape ``[num_tokens, top_k]`` (int), or
    ``None`` if it can't recover them (the forward will then be skipped).
    """

    def deco(fn: RouterExtractor) -> RouterExtractor:
        ROUTER_EXTRACTORS[class_name] = fn
        return fn

    return deco


@register_router_extractor("Qwen3MoeTopKRouter")
@register_router_extractor("Qwen3VLMoeTopKRouter")
@register_router_extractor("Qwen3OmniMoeTopKRouter")
def _extract_qwen3_topk(output: Any) -> Optional[torch.Tensor]:
    """Qwen3-family patched router returns ``(logits, top_value, indices)``."""
    if isinstance(output, (tuple, list)) and len(output) >= 3:
        cand = output[2]
        if isinstance(cand, torch.Tensor) and cand.dtype in (torch.int32, torch.int64, torch.long):
            return cand
    return None


# ---------------------------------------------------------------------------
# External-record routers. Families whose router forward doesn't surface
# indices (DeepSeek-V3) record by calling :func:`record_router_indices`
# explicitly from the patched MoE block. We still want
# :func:`attach_moe_router_monitor` to count and pre-register these modules
# so the heatmap layer order is stable across resumes.
# ---------------------------------------------------------------------------
EXTERNAL_RECORD_ROUTERS: set[str] = set()


def register_external_record_router(class_name: str) -> None:
    """Mark a router class as recording via explicit ``record_router_indices()``
    calls rather than a forward hook."""
    EXTERNAL_RECORD_ROUTERS.add(class_name)


register_external_record_router("DeepseekV3TopkRouter")


def record_router_indices(router_module: nn.Module, indices: torch.Tensor) -> None:
    """Record expert selections from a family-patched MoE block.

    Called from inside the patched ``DeepseekV3MoE.forward`` (and any other
    family whose top-k math lives downstream of the router). No-op when no
    monitor is active or the monitor is paused. Symmetric to
    :func:`veomni.utils.moe_router_replay.maybe_replay_indices`.
    """
    monitor = _active_monitor
    if monitor is None or monitor._paused:
        return
    monitor.record(router_module, indices)


# ---------------------------------------------------------------------------
# Hook builder.
# ---------------------------------------------------------------------------


def _make_router_hook(extractor: RouterExtractor):
    def _hook(module: nn.Module, inputs, output):  # noqa: ANN001
        monitor = _active_monitor
        if monitor is None or monitor._paused:
            return
        indices = extractor(output)
        # A registered extractor that returns None means its router class's
        # forward output shape changed. Fail loud — silently skipping would
        # produce empty heatmaps that look like a balanced model.
        assert indices is not None, (
            f"MoE router extractor for {type(module).__name__} returned None. "
            "Update the extractor in veomni/utils/moe_monitor.py to match the "
            "router's current forward output."
        )
        monitor.record(module, indices)

    return _hook


def attach_moe_router_monitor(model: nn.Module, monitor: "MoERouterMonitor") -> int:
    """Walk ``model`` and wire up every recognized router module.

    Two recognition paths:

    * :data:`ROUTER_EXTRACTORS` — routers whose forward output exposes top-k
      indices. A forward hook is registered.
    * :data:`EXTERNAL_RECORD_ROUTERS` — routers whose patched MoE block calls
      :func:`record_router_indices` directly. No hook is registered, but the
      layer is pre-registered so the heatmap row order is stable.

    Each router's order is captured at attach time so logs are consistent
    across resumes. Returns the number of routers wired up. The caller should
    treat 0 as an error — the monitor is enabled but will never accumulate data.
    """
    attached = 0
    for mod in model.modules():
        cls_name = type(mod).__name__
        extractor = ROUTER_EXTRACTORS.get(cls_name)
        if extractor is not None:
            mod.register_forward_hook(_make_router_hook(extractor))
            monitor._register_layer(mod)
            attached += 1
        elif cls_name in EXTERNAL_RECORD_ROUTERS:
            monitor._register_layer(mod)
            attached += 1
    monitor._attached_count = attached
    return attached


# ---------------------------------------------------------------------------
# Monitor.
# ---------------------------------------------------------------------------


class MoERouterMonitor:
    """Accumulates per-layer per-expert token counts and produces summary metrics.

    Counts accumulate on device. The only CPU-sync points are inside
    :meth:`compute_metrics` (one all-reduce + one host transfer per interval).
    """

    def __init__(
        self,
        num_experts: int,
        dp_group: Optional["dist.ProcessGroup"] = None,
    ):
        """
        Args:
            num_experts: Total experts per MoE layer (global, not per-EP-rank).
            dp_group: The process group to all-reduce expert counts across.
                Should span every rank that holds a *distinct* token slice
                (data-parallel × sequence-parallel). Do **not** include EP
                siblings: the router gate is replicated across EP, so they
                produce identical indices and summing them inflates counts
                by ``ep_size``. In VeOmni this is ``parallel_state.fsdp_group``
                (which is the ``dp_sp`` mesh dim).
        """
        self.num_experts = num_experts
        self.dp_group = dp_group
        # Sticky disable, separate from pause/resume so callers using
        # pause/resume for phase scoping can't clobber a hard disable.
        self._disabled: bool = False

        # Layer order captured at attach time (stable across resumes).
        self._layer_order: List[int] = []
        # Per-module accumulated counts, lazily allocated on first record.
        self._counts: Dict[int, torch.Tensor] = {}

        # Step range tracking for heatmap captions.
        self._accumulate_start_step: int = 0
        self._last_step_range: tuple = (0, 0)

        # Pause/resume support (used by verl during rollout phase).
        self._paused: bool = False

        # Diagnostics.
        self._attached_count: int = 0

    # ---------------------- Lifecycle ----------------------

    def pause(self) -> None:
        """Stop accumulating counts. Hooks become no-ops until :meth:`resume`."""
        self._paused = True

    def resume(self) -> None:
        """Resume count accumulation. No-op if the monitor was permanently disabled."""
        if not self._disabled:
            self._paused = False

    def disable(self) -> None:
        """Permanently disable accumulation. Survives subsequent ``resume()`` calls."""
        self._disabled = True
        self._paused = True

    # ---------------------- Internal ----------------------

    def _register_layer(self, module: nn.Module) -> None:
        """Capture the layer's stable order at attach time.

        Idempotent: calling :func:`attach_moe_router_monitor` more than once
        on the same model (or otherwise re-registering a router) must not
        produce duplicate rows in the heatmap.
        """
        mid = id(module)
        if mid not in self._layer_order:
            self._layer_order.append(mid)

    def _reset_counts(self) -> None:
        for mid in self._counts:
            self._counts[mid].zero_()

    # ---------------------- Recording ----------------------

    def record(self, module: nn.Module, router_indices: torch.Tensor) -> None:
        """Record expert selections from one router forward.

        Called by the forward hook. Pure on-device accumulation.
        """
        mid = id(module)
        if mid not in self._counts:
            # First-seen forward for this router — lazily allocate counts.
            # If the layer wasn't pre-registered at attach time (e.g.
            # dynamically-added router), append to layer order now.
            if mid not in self._layer_order:
                self._layer_order.append(mid)
            self._counts[mid] = torch.zeros(self.num_experts, dtype=torch.long, device=router_indices.device)
        counts = torch.bincount(
            router_indices.reshape(-1).to(torch.long),
            minlength=self.num_experts,
        )
        self._counts[mid] += counts.detach()

    # ---------------------- Reduction & metrics ----------------------

    def _stack_and_reduce(self) -> torch.Tensor:
        """Stack per-layer counts and all-reduce across the configured DP+SP group.

        Returns an on-device ``[num_moe_layers, num_experts]`` long tensor.
        Layers that were registered at attach time but did not fire during
        the interval (e.g. routing-gated layers, partial-network warmup) are
        included as zero rows so the heatmap shape stays stable across
        intervals.
        """
        if not self._counts:
            # No data recorded yet — return an empty tensor on CPU.
            return torch.zeros(0, self.num_experts, dtype=torch.long)

        # Device hint from any allocated counts tensor — we need it to
        # synthesize zero rows for layers that haven't fired yet.
        device = next(iter(self._counts.values())).device
        zero_row = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        matrix = torch.stack([self._counts.get(mid, zero_row) for mid in self._layer_order])

        # All-reduce across the DP+SP group so the heatmap aggregates every
        # distinct token slice. EP siblings hold the replicated gate and
        # produce identical counts, so we deliberately do not reduce across
        # EP — that would inflate by ``ep_size``.
        if self.dp_group is not None and dist.is_initialized():
            dist.all_reduce(matrix, op=dist.ReduceOp.SUM, group=self.dp_group)
        return matrix

    def get_load_matrix(self, current_step: int = 0) -> torch.Tensor:
        """Return normalized ``[num_moe_layers, num_experts]`` load matrix and reset.

        Rows sum to 1.0. Issues one CUDA sync via the host transfer.
        """
        matrix = self._stack_and_reduce()
        if matrix.numel() == 0:
            return matrix.float()
        matrix = matrix.float().cpu()
        row_sums = matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
        matrix = matrix / row_sums
        self._last_step_range = (self._accumulate_start_step, current_step)
        self._reset_counts()
        self._accumulate_start_step = current_step + 1
        return matrix

    @staticmethod
    def compute_vio(load_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-layer load-balance violation metrics.

        With ``deviation = load_matrix * num_experts - 1``:

        - ``max_vio``: most-overloaded expert per layer, in ``[0, num_experts - 1]``.
        - ``min_vio``: most-underloaded expert per layer, in ``[-1, 0]``.
        - ``avg_vio``: mean absolute deviation per layer, ``[0, +inf)``.

        All three are 0 under perfect uniform routing.
        """
        num_experts = load_matrix.shape[1]
        deviation = load_matrix * num_experts - 1.0
        return {
            "max_vio": deviation.max(dim=1).values,
            "min_vio": deviation.min(dim=1).values,
            "avg_vio": deviation.abs().mean(dim=1),
        }

    def compute_metrics(
        self,
        current_step: int,
        prefix: str = "moe",
        format_only_on: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Produce a backend-agnostic metrics dict for the current interval.

        **Collective**: under EP/DP this calls ``all_reduce``. Every rank in
        the EP/DP groups must call this method even if only one rank logs the
        result. Pass ``format_only_on=False`` on non-logging ranks to skip the
        scalar + heatmap build (still does the collective + reset).

        Returns a dict with:

        - ``{prefix}/expert_load_heatmap``: PIL ``Image`` (when matplotlib is available).
        - ``{prefix}/{max,min,avg}_vio/layer_{i}``: per-layer scalars.
        - ``{prefix}/{max,min,avg}_vio/{max,avg}``: across-layer aggregates.

        Returns an empty dict if no data was recorded or ``format_only_on`` is False.
        """
        load_matrix = self.get_load_matrix(current_step=current_step)
        num_layers = load_matrix.shape[0]
        if num_layers == 0 or format_only_on is False:
            return {}

        vio = self.compute_vio(load_matrix)
        max_vio, min_vio, avg_vio = vio["max_vio"], vio["min_vio"], vio["avg_vio"]

        metrics: Dict[str, Any] = {}
        metrics[f"{prefix}/expert_load_heatmap"] = self.build_heatmap_image(load_matrix)
        for i in range(num_layers):
            metrics[f"{prefix}/max_vio/layer_{i}"] = max_vio[i].item()
            metrics[f"{prefix}/min_vio/layer_{i}"] = min_vio[i].item()
            metrics[f"{prefix}/avg_vio/layer_{i}"] = avg_vio[i].item()
        metrics[f"{prefix}/max_vio/max"] = max_vio.max().item()
        metrics[f"{prefix}/max_vio/avg"] = max_vio.mean().item()
        metrics[f"{prefix}/min_vio/max"] = min_vio.max().item()
        metrics[f"{prefix}/min_vio/avg"] = min_vio.mean().item()
        metrics[f"{prefix}/avg_vio/max"] = avg_vio.max().item()
        metrics[f"{prefix}/avg_vio/avg"] = avg_vio.mean().item()
        return metrics

    def build_heatmap_image(self, load_matrix: torch.Tensor, caption: Optional[str] = None):
        """Build a PIL ``Image`` of the load matrix.

        The caller wraps it for its backend (e.g. ``wandb.Image(img, caption=...)``).
        Requires matplotlib (declared in ``pyproject.toml``).
        """
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

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
            return Image.open(buf).copy()
        finally:
            buf.close()
