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

from numbers import Integral
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
@register_router_extractor("Qwen3_5MoeTopKRouter")
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
    :meth:`compute_metrics` (DP+SP/FSDP reductions + host transfer per interval).
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

        # Physical EP telemetry is keyed by the stable router-layer index used
        # by the Qwen3.5 load-balancer attachment. Tensors live on the same
        # device as the corresponding logical router counts.
        self._ep_rank_loads_before: Dict[int, torch.Tensor] = {}
        self._ep_rank_loads_after: Dict[int, torch.Tensor] = {}
        # Per-layer [active_replicas, moved_tokens, total_routed_tokens].
        self._ep_stats: Dict[int, torch.Tensor] = {}
        self._ep_rank_size: Optional[int] = None

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

    def _reset_ep_balance(self) -> None:
        self._ep_rank_loads_before.clear()
        self._ep_rank_loads_after.clear()
        self._ep_stats.clear()

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

    @staticmethod
    def _validate_nonnegative_integer(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        return int(value)

    @staticmethod
    def _normalize_rank_loads(name: str, values: Any) -> tuple[int, ...]:
        if isinstance(values, torch.Tensor):
            if values.ndim != 1:
                raise ValueError(f"{name} must have one-dimensional rank-load shape.")
            if values.dtype == torch.bool or values.is_floating_point() or values.is_complex():
                raise ValueError(f"{name} entries must be non-negative integers.")
            values = values.detach().cpu().tolist()
        else:
            try:
                values = tuple(values)
            except TypeError as exc:
                raise ValueError(f"{name} must have one-dimensional rank-load shape.") from exc

        if not values:
            raise ValueError(f"{name} must contain at least one EP-rank load.")
        normalized = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} entries must be non-negative integers.")
            normalized.append(int(value))
        return tuple(normalized)

    def record_ep_balance(
        self,
        layer_index: int,
        before_rank_loads: Any,
        after_rank_loads: Any,
        active_replicas: int,
        moved_tokens: int,
    ) -> None:
        """Accumulate physical EP-rank balance telemetry for one routing plan.

        The caller supplies one before/after load per EP rank. The monitor
        validates each plan before mutating interval state and deliberately
        stores the values on the corresponding router-count device.
        """
        if self._paused or self._disabled:
            return

        layer_index = self._validate_nonnegative_integer("layer_index", layer_index)
        if layer_index >= len(self._layer_order):
            raise ValueError(
                f"layer_index {layer_index} does not match an attached router layer "
                f"(registered layers: {len(self._layer_order)})."
            )
        before = self._normalize_rank_loads("before_rank_loads", before_rank_loads)
        after = self._normalize_rank_loads("after_rank_loads", after_rank_loads)
        if len(before) != len(after):
            raise ValueError("before_rank_loads and after_rank_loads must have the same shape.")
        if sum(before) != sum(after):
            raise ValueError("before_rank_loads and after_rank_loads must conserve routed tokens.")
        if self._ep_rank_size is not None and len(before) != self._ep_rank_size:
            raise ValueError(f"EP rank-load size must remain {self._ep_rank_size} across layers; got {len(before)}.")
        active_replicas = self._validate_nonnegative_integer("active_replicas", active_replicas)
        moved_tokens = self._validate_nonnegative_integer("moved_tokens", moved_tokens)

        router_mid = self._layer_order[layer_index]
        router_counts = self._counts.get(router_mid)
        if router_counts is None:
            raise ValueError(
                f"layer_index {layer_index} has no router counts yet; physical telemetry must follow router recording."
            )

        device = router_counts.device
        before_tensor = torch.tensor(before, dtype=torch.long, device=device)
        after_tensor = torch.tensor(after, dtype=torch.long, device=device)
        stats_tensor = torch.tensor(
            (active_replicas, moved_tokens, sum(before)),
            dtype=torch.long,
            device=device,
        )
        if layer_index in self._ep_rank_loads_before:
            self._ep_rank_loads_before[layer_index] += before_tensor
            self._ep_rank_loads_after[layer_index] += after_tensor
            self._ep_stats[layer_index] += stats_tensor
        else:
            self._ep_rank_loads_before[layer_index] = before_tensor
            self._ep_rank_loads_after[layer_index] = after_tensor
            self._ep_stats[layer_index] = stats_tensor
        if self._ep_rank_size is None:
            self._ep_rank_size = len(before)

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

    def _stack_and_reduce_ep_balance(
        self,
        num_layers: int,
        device: torch.device,
        rank_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero_loads = torch.zeros(rank_size, dtype=torch.long, device=device)
        zero_stats = torch.zeros(3, dtype=torch.long, device=device)
        before = torch.stack([self._ep_rank_loads_before.get(i, zero_loads) for i in range(num_layers)])
        after = torch.stack([self._ep_rank_loads_after.get(i, zero_loads) for i in range(num_layers)])
        stats = torch.stack([self._ep_stats.get(i, zero_stats) for i in range(num_layers)])

        packed = torch.cat((before.reshape(-1), after.reshape(-1), stats.reshape(-1)))
        if self.dp_group is not None and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self.dp_group)
        before_size = before.numel()
        after_size = after.numel()
        before = packed[:before_size].reshape_as(before)
        after = packed[before_size : before_size + after_size].reshape_as(after)
        stats = packed[before_size + after_size :].reshape_as(stats)
        return before, after, stats

    def _resolve_interval_ep_rank_size(self, device: torch.device) -> Optional[int]:
        local_has_ep = bool(self._ep_rank_loads_before)
        local_rank_size = self._ep_rank_size if local_has_ep else 0
        assert local_rank_size is not None
        metadata = torch.tensor(
            (int(local_has_ep), local_rank_size, local_rank_size * local_rank_size),
            dtype=torch.long,
            device=device,
        )
        if self.dp_group is not None and dist.is_initialized():
            dist.all_reduce(metadata, op=dist.ReduceOp.SUM, group=self.dp_group)

        present_count, rank_size_sum, rank_size_square_sum = (int(value) for value in metadata.tolist())
        if present_count == 0:
            return None
        if rank_size_sum % present_count != 0 or present_count * rank_size_square_sum != rank_size_sum**2:
            raise ValueError("EP rank-load size must be consistent across the configured DP+SP/FSDP group.")
        return rank_size_sum // present_count

    def _collect_interval_snapshot(
        self,
    ) -> tuple[
        torch.Tensor,
        Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        torch.device,
    ]:
        matrix = self._stack_and_reduce()
        collective_device = matrix.device
        if matrix.numel() == 0:
            return matrix.float(), None, collective_device

        interval_rank_size = self._resolve_interval_ep_rank_size(matrix.device)
        ep_balance = None
        if interval_rank_size is not None:
            # The group-wide metadata agreement above makes every rank enter
            # this physical collective with the same packed shape. Ranks that
            # recorded no physical telemetry contribute zero rows.
            ep_balance = self._stack_and_reduce_ep_balance(
                matrix.shape[0],
                matrix.device,
                interval_rank_size,
            )
        return matrix, ep_balance, collective_device

    @staticmethod
    def _prepare_snapshot_for_format(
        matrix: torch.Tensor,
        ep_balance: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[
        torch.Tensor,
        Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        matrix = matrix.float().cpu()
        row_sums = matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
        matrix = matrix / row_sums
        if ep_balance is not None:
            ep_balance = tuple(value.cpu() for value in ep_balance)
        return matrix, ep_balance

    def _commit_interval(self, current_step: int) -> None:
        self._last_step_range = (self._accumulate_start_step, current_step)
        self._reset_counts()
        self._reset_ep_balance()
        self._accumulate_start_step = current_step + 1

    def _format_succeeded_on_group(self, local_success: bool, device: torch.device) -> bool:
        success = torch.tensor(int(local_success), dtype=torch.long, device=device)
        if self.dp_group is not None and dist.is_initialized():
            dist.all_reduce(success, op=dist.ReduceOp.MIN, group=self.dp_group)
        return bool(success.item())

    def get_load_matrix(self, current_step: int = 0) -> torch.Tensor:
        """Return normalized ``[num_moe_layers, num_experts]`` load matrix and reset.

        Rows sum to 1.0. Issues one CUDA sync via the host transfer.
        """
        matrix, ep_balance, _ = self._collect_interval_snapshot()
        if matrix.numel() != 0:
            matrix, _ = self._prepare_snapshot_for_format(matrix, ep_balance)
            self._commit_interval(current_step)
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

    @staticmethod
    def compute_rank_imbalance(rank_load_matrix: torch.Tensor) -> torch.Tensor:
        """Return exact normalized EP-rank imbalance for every layer row."""
        rank_load_matrix = rank_load_matrix.float()
        rank_count = rank_load_matrix.shape[1]
        totals = rank_load_matrix.sum(dim=1)
        safe_totals = totals.clamp(min=1.0).unsqueeze(1)
        imbalance = (rank_count * rank_load_matrix / safe_totals - 1.0).abs().mean(dim=1)
        return torch.where(totals > 0, imbalance, torch.zeros_like(imbalance))

    def _format_metrics(
        self,
        load_matrix: torch.Tensor,
        ep_balance: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        prefix: str,
    ) -> Dict[str, Any]:
        num_layers = load_matrix.shape[0]
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

        if ep_balance is not None:
            before, after, stats = ep_balance
            before_imbalance = self.compute_rank_imbalance(before)
            after_imbalance = self.compute_rank_imbalance(after)
            active_replicas = stats[:, 0]
            moved_tokens = stats[:, 1]
            total_routed_tokens = stats[:, 2]
            moved_fraction = torch.where(
                total_routed_tokens > 0,
                moved_tokens.float() / total_routed_tokens.float().clamp(min=1.0),
                torch.zeros_like(total_routed_tokens, dtype=torch.float),
            )

            metrics[f"{prefix}/ep_rank_load_before_heatmap"] = self.build_ep_rank_heatmap_image(before, "before")
            metrics[f"{prefix}/ep_rank_load_after_heatmap"] = self.build_ep_rank_heatmap_image(after, "after")
            for i in range(num_layers):
                metrics[f"{prefix}/ep_rank_imbalance_before/layer_{i}"] = before_imbalance[i].item()
                metrics[f"{prefix}/ep_rank_imbalance_after/layer_{i}"] = after_imbalance[i].item()
                metrics[f"{prefix}/ep_active_replicas/layer_{i}"] = active_replicas[i].item()
                metrics[f"{prefix}/ep_moved_tokens/layer_{i}"] = moved_tokens[i].item()
                metrics[f"{prefix}/ep_total_routed_tokens/layer_{i}"] = total_routed_tokens[i].item()
                metrics[f"{prefix}/ep_moved_token_fraction/layer_{i}"] = moved_fraction[i].item()

            for name, values in (
                ("ep_rank_imbalance_before", before_imbalance),
                ("ep_rank_imbalance_after", after_imbalance),
                ("ep_moved_token_fraction", moved_fraction),
            ):
                metrics[f"{prefix}/{name}/max"] = values.max().item()
                metrics[f"{prefix}/{name}/avg"] = values.mean().item()
            for name, values in (
                ("ep_active_replicas", active_replicas),
                ("ep_moved_tokens", moved_tokens),
                ("ep_total_routed_tokens", total_routed_tokens),
            ):
                metrics[f"{prefix}/{name}/sum"] = values.sum().item()
                metrics[f"{prefix}/{name}/max"] = values.max().item()
                metrics[f"{prefix}/{name}/avg"] = values.float().mean().item()
        return metrics

    def compute_metrics(
        self,
        current_step: int,
        prefix: str = "moe",
        format_only_on: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Produce a backend-agnostic metrics dict for the current interval.

        **Collective**: this calls ``all_reduce`` over the configured
        DP+SP/FSDP group. Every member must call this method even if only one
        global rank logs the result. Pass ``format_only_on=False`` on
        non-logging ranks to skip the scalar + heatmap build (the rank still
        participates in collectives and resets). EP siblings are excluded.

        Returns a dict with:

        - ``{prefix}/expert_load_heatmap``: PIL ``Image`` (when matplotlib is available).
        - ``{prefix}/{max,min,avg}_vio/layer_{i}``: per-layer scalars.
        - ``{prefix}/{max,min,avg}_vio/{max,avg}``: across-layer aggregates.

        Returns an empty dict if no data was recorded or ``format_only_on`` is False.
        """
        should_format = format_only_on is not False
        load_matrix, ep_balance, collective_device = self._collect_interval_snapshot()
        num_layers = load_matrix.shape[0]
        if num_layers == 0:
            return {}

        metrics: Dict[str, Any] = {}
        format_error: Optional[Exception] = None
        previous_step_range = self._last_step_range
        if should_format:
            self._last_step_range = (self._accumulate_start_step, current_step)
            try:
                load_matrix, ep_balance = self._prepare_snapshot_for_format(load_matrix, ep_balance)
                metrics = self._format_metrics(load_matrix, ep_balance, prefix)
            except Exception as exc:
                format_error = exc
                self._last_step_range = previous_step_range

        if self._format_succeeded_on_group(format_error is None, collective_device):
            self._commit_interval(current_step)
        else:
            self._last_step_range = previous_step_range
            if format_error is not None:
                raise format_error
            raise RuntimeError("MoE metric formatting failed on another DP+SP/FSDP group rank.")

        if not should_format:
            return {}
        return metrics

    def build_ep_rank_heatmap_image(self, rank_load_matrix: torch.Tensor, stage: str):
        """Build a before/after PIL heatmap of normalized physical EP-rank load."""
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        if stage not in ("before", "after"):
            raise ValueError("stage must be 'before' or 'after'.")
        row_sums = rank_load_matrix.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        normalized = rank_load_matrix.float() / row_sums
        start, end = self._last_step_range
        stage_title = "Before Temporary Replicas" if stage == "before" else "After Temporary Replicas"

        fig, ax = plt.subplots(
            figsize=(max(8, rank_load_matrix.shape[1] * 0.8), max(4, rank_load_matrix.shape[0] * 0.2))
        )
        im = ax.imshow(normalized.numpy(), aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("Physical EP Rank")
        ax.set_ylabel("MoE Layer Index")
        ax.set_title(f"MoE EP Rank Load {stage_title} (Steps {start}-{end})")
        fig.colorbar(im, ax=ax, label="Normalized Routed Token Fraction")
        fig.tight_layout()

        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf).copy()
        finally:
            buf.close()

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
