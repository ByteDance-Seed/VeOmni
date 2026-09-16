"""Declarative whole-item routing, orthogonal to a module's token-wise SP hooks."""

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import torch

from ....distributed.parallel_state import get_parallel_state
from ....utils.data_balance.module_balance import BalancePlan, ModuleDataBalancer


@dataclass(frozen=True)
class ItemTensorField:
    """A packed input tensor and the metadata key containing its per-item splits."""

    name: str
    lengths_key: str


@dataclass(frozen=True)
class ModuleStructure:
    """Declare computational structure rather than a preferred parallel strategy."""

    global_attention: bool = False
    non_overlapping_patchify: bool = False
    whole_item_partition: bool = True


@dataclass(frozen=True)
class DataBalanceSpec:
    """Module-owned metric, input partitioning, output inverse and capability."""

    fields: tuple[ItemTensorField, ...]
    output_names: tuple[str, ...]
    output_lengths_key: str
    cost_lengths_key: str
    structure: ModuleStructure
    cost_exponent: float = 2
    cost_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if not names or len(set(names)) != len(names) or not self.output_names:
            raise ValueError("A balance specification needs unique input fields and output inverses.")
        if self.cost_fn is None and (not math.isfinite(self.cost_exponent) or self.cost_exponent < 0):
            raise ValueError("Cost exponent must be finite and nonnegative.")


@dataclass(frozen=True)
class BalanceStrategy:
    """Primary execution strategy plus independent cross-DP item balancing."""

    strategy: str
    balance_dp: bool


def resolve_balance_strategy(
    structure: ModuleStructure, *, sp_size: int, dp_size: int, override: Optional[str] = None
) -> BalanceStrategy:
    """Derive SP-slice / DP-balance / replicate from structure and local topology.

    Token-wise SP already balances global, halo-free attention *within* an SP
    group; it does not address skew across DP groups. ``off`` disables only that
    cross-DP optimization, leaving existing SP execution intact.
    """
    if sp_size < 1 or dp_size < 1:
        raise ValueError("Parallel dimensions must be positive.")
    if override not in (None, "auto", "off"):
        raise ValueError("data_balance_override must be 'auto' or 'off'.")
    token_sp = structure.global_attention and structure.non_overlapping_patchify
    balance_dp = structure.whole_item_partition and dp_size > 1 and override != "off"
    if token_sp and sp_size > 1:
        strategy = "sp_slice"
    elif balance_dp:
        strategy = "dp_balance"
    else:
        strategy = "replicate"
    return BalanceStrategy(strategy, balance_dp)


class DataBalanceMixin:
    """Opt-in by multiple inheritance; the VeOmni executor opens one call scope.

    Hooks call the generic pre-routing after module-owned CPU metadata is in
    hand, before SP slicing, and post-routing after SP gathering. The native HF
    graph path never opens this scope and remains free of DP communication.
    No process group or ambient ParallelState is captured at construction.
    """

    data_balance_specs: Mapping[str, DataBalanceSpec] = {}

    @contextmanager
    def data_balance_scope(self):
        if getattr(self, "_data_balance_active", False):
            raise RuntimeError("Reentrant data-balance hooks are not supported.")
        self._data_balance_active = True
        self._data_balance_plans: dict[str, BalancePlan] = {}
        try:
            yield
            if self._data_balance_plans:
                raise RuntimeError("A data-balance hook did not restore its outputs.")
        finally:
            self._data_balance_active = False
            self._data_balance_plans = {}

    def data_balance_pre(self, method: str, tensors: dict, metadata: dict) -> tuple[dict, bool]:
        """Route declared fields using this module's *currently scoped* DP group."""
        spec = self.data_balance_specs.get(method)
        if spec is None or not getattr(self, "_data_balance_active", False) or not self.training:
            return tensors, False
        ps = get_parallel_state()
        strategy = resolve_balance_strategy(
            spec.structure,
            sp_size=ps.sp_size,
            dp_size=ps.dp_size,
            override=getattr(self.config, "data_balance_override", None),
        )
        if not strategy.balance_dp:
            return tensors, False
        if ps.dp_group is None:
            raise RuntimeError("A non-singleton DP dimension requires a module-local group.")
        # Let the core share local cost errors before any tensor collective, so
        # a malformed callback on one owner cannot leave its peers hanging.
        try:
            lengths = torch.tensor(metadata[spec.cost_lengths_key], dtype=torch.float64)
            costs = spec.cost_fn(lengths) if spec.cost_fn is not None else lengths.pow(spec.cost_exponent)
            costs = costs.tolist() if isinstance(costs, torch.Tensor) else (() if costs is None else costs)
        except (KeyError, TypeError, ValueError, OverflowError):
            # A nonfinite sentinel fails even for an owner with zero items.
            costs = [float("nan")]
        if method in self._data_balance_plans:
            raise RuntimeError("An invocation already owns a routing plan.")
        balanced, plan = ModuleDataBalancer(ps.dp_group).balance(
            {field.name: tensors.get(field.name) for field in spec.fields},
            {field.name: metadata.get(field.lengths_key) for field in spec.fields},
            metadata.get(spec.output_lengths_key),
            costs=costs,
        )
        self._data_balance_plans[method] = plan
        return {**tensors, **balanced}, True

    def data_balance_post(self, method: str, outputs: dict) -> dict:
        """Inverse declared token outputs only after their full SP gather."""
        plan = getattr(self, "_data_balance_plans", {}).pop(method, None)
        if plan is None:
            return outputs
        restored = dict(outputs)
        for name in self.data_balance_specs[method].output_names:
            value = outputs[name]
            restored[name] = (
                [plan.restore(layer) for layer in value] if isinstance(value, list) else plan.restore(value)
            )
        return restored


__all__ = [
    "ItemTensorField",
    "ModuleStructure",
    "DataBalanceSpec",
    "BalanceStrategy",
    "resolve_balance_strategy",
    "DataBalanceMixin",
]
