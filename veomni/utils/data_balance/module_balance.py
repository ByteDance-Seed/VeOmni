# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module-agnostic whole-item DP scheduling and differentiable reverse routing."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .balance_sorting_algo import post_mbs_balancing_greedy_without_pad


def greedy_cost_assignment(costs: Sequence[float], num_replicas: int) -> tuple[tuple[int, ...], ...]:
    """Assign descending costs to the least-loaded bin; ties prefer original order/rank."""
    table = torch.arange(len(costs), dtype=torch.long).reshape(-1, 1)
    bins = post_mbs_balancing_greedy_without_pad(
        table,
        num_replicas,
        0,
        cost_fn=lambda _: torch.tensor(costs, dtype=torch.float64),
    )
    return tuple(tuple(bucket[:, 0].tolist()) for bucket in bins)


class _Exchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group, send_counts, recv_counts):
        ctx.group, ctx.send_counts, ctx.recv_counts = group, send_counts, recv_counts
        output = tensor.new_empty((sum(recv_counts), *tensor.shape[1:]))
        dist.all_to_all_single(output, tensor.contiguous(), list(recv_counts), list(send_counts), group=group)
        return output

    @staticmethod
    def backward(ctx, grad):
        # Incoming grad is only read; output owns storage (including empty tensors).
        output = grad.new_empty((sum(ctx.send_counts), *grad.shape[1:]))
        dist.all_to_all_single(
            output, grad.contiguous(), list(ctx.send_counts), list(ctx.recv_counts), group=ctx.group
        )
        return output, None, None, None


def _exchange(tensor, group, send_counts, recv_counts):
    # None explicitly means a singleton DP dimension, NEVER the world group.
    return (
        tensor
        if group is None or dist.get_world_size(group) == 1
        else _Exchange.apply(tensor, group, send_counts, recv_counts)
    )


def _permute(tensor: torch.Tensor, lengths: Sequence[int], order: Sequence[int]) -> torch.Tensor:
    if not order:
        return tensor[:0]
    chunks = tensor.split(tuple(lengths), dim=0)
    return torch.cat([chunks[index] for index in order], dim=0)


@dataclass(frozen=True)
class BalanceItem:
    owner: int
    index: int
    input_lengths: tuple[int, ...]
    output_length: int
    cost: float


@dataclass(frozen=True)
class BalancePlan:
    """Per-call routing context; no ambient ParallelState is consulted in backward."""

    group: dist.ProcessGroup | None
    rank: int
    field_names: tuple[str, ...]
    original: tuple[BalanceItem, ...]
    buckets: tuple[tuple[BalanceItem, ...], ...]

    @property
    def assigned(self) -> tuple[BalanceItem, ...]:
        return self.buckets[self.rank]

    @property
    def output_lengths(self) -> tuple[int, ...]:
        return tuple(item.output_length for item in self.assigned)

    def restore(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return full post-SP output tokens to their original DP owner and item order.

        All DP ranks must call this (and its backward), even with no owned items.
        The module must fold an empty result's zero-valued gradient anchor into
        its downstream carrier/loss; skipping that anchor can skip collectives.
        """
        if tensor.shape[0] != sum(self.output_lengths):
            raise ValueError("Output length does not match this invocation's routing plan.")
        size = len(self.buckets)
        send = tuple(sum(i.output_length for i in self.assigned if i.owner == r) for r in range(size))
        received_items = tuple(i for bucket in self.buckets for i in bucket if i.owner == self.rank)
        recv = tuple(sum(i.output_length for i in bucket if i.owner == self.rank) for bucket in self.buckets)
        received = _exchange(tensor, self.group, send, recv)
        inverse = tuple(sorted(range(len(received_items)), key=lambda j: received_items[j].index))
        return _permute(received, tuple(i.output_length for i in received_items), inverse)


class ModuleDataBalancer:
    """Balance packed tensor fields with module-provided per-item lengths and costs.

    Tensor fields may have different input splits (e.g. patches vs one grid row),
    and output splits may differ from input splits (e.g. spatial merging).
    A cost callback receives host-side input-length tuples and returns scheduling
    costs, not token counts. Schemas and collective order must agree on all ranks.
    """

    def __init__(self, group: dist.ProcessGroup | None, cost_fn: Callable | None = None):
        self.group = group
        self.cost_fn = cost_fn

    def balance(
        self,
        tensors: Mapping[str, torch.Tensor],
        input_lengths: Mapping[str, Sequence[int]],
        output_lengths: Sequence[int],
        costs: Sequence[float] | None = None,
    ) -> tuple[dict[str, torch.Tensor], BalancePlan]:
        size = 1 if self.group is None else dist.get_world_size(self.group)
        rank = 0 if self.group is None else dist.get_rank(self.group)
        names, schema, outputs = (), (), ()
        lengths = {}
        error = None
        try:
            names = tuple(sorted(tensors))
            if not names or set(names) != set(input_lengths):
                raise ValueError("Tensor fields and input-length fields must match and be nonempty.")
            if any(not isinstance(tensors[name], torch.Tensor) or tensors[name].ndim == 0 for name in names):
                raise ValueError("Packed tensor fields need a leading item dimension.")
            lengths = {name: tuple(int(n) for n in input_lengths[name]) for name in names}
            outputs = tuple(int(n) for n in output_lengths)
            # Backward collectives must participate on every rank, including
            # empty owners. Reject rank-asymmetric autograd before routing.
            schema = tuple(
                (name, tuple(tensors[name].shape[1:]), str(tensors[name].dtype), tensors[name].requires_grad)
                for name in names
            )
            for name in names:
                if len(lengths[name]) != len(outputs) or any(n < 0 for n in lengths[name]):
                    raise ValueError("Every tensor field needs one nonnegative split per item.")
                if sum(lengths[name]) != tensors[name].shape[0]:
                    raise ValueError("Packed tensor size does not match item splits.")
            if any(n < 0 for n in outputs):
                raise ValueError("Output splits must be nonnegative.")
            if costs is None:
                costs = self.cost_fn(lengths) if self.cost_fn else [n * n for n in outputs]
            costs = tuple(float(c) for c in costs)
            if len(costs) != len(outputs) or any(not math.isfinite(c) or c < 0 for c in costs):
                raise ValueError("Expected one finite nonnegative cost per item.")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            costs = ()
        local = (schema, lengths, outputs, costs, error)
        # Only small host metadata uses object collectives; tensor payloads never
        # use pickle. Validation errors are shared before any payload exchange.
        gathered = [local]
        if size > 1:
            gathered = [None] * size
            dist.all_gather_object(gathered, local, group=self.group)
        if any(meta[4] is not None for meta in gathered) or any(meta[0] != schema for meta in gathered):
            raise ValueError(f"DP balancer metadata/schema mismatch: {[m[4] for m in gathered]}")
        items = tuple(
            BalanceItem(owner, index, tuple(meta[1][name][index] for name in names), output, meta[3][index])
            for owner, meta in enumerate(gathered)
            for index, output in enumerate(meta[2])
        )
        assignment = greedy_cost_assignment([item.cost for item in items], size)
        buckets = tuple(
            tuple(sorted((items[index] for index in indices), key=lambda i: (i.owner, i.index)))
            for indices in assignment
        )
        original = tuple(i for i in items if i.owner == rank)
        plan = BalancePlan(self.group, rank, names, original, buckets)
        result = {}
        outgoing = tuple(i for bucket in buckets for i in bucket if i.owner == rank)
        for field, name in enumerate(names):
            send = tuple(sum(i.input_lengths[field] for i in bucket if i.owner == rank) for bucket in buckets)
            recv = tuple(sum(i.input_lengths[field] for i in plan.assigned if i.owner == r) for r in range(size))
            packed = _permute(tensors[name], lengths[name], tuple(i.index for i in outgoing))
            result[name] = _exchange(packed, self.group, send, recv)
        return result, plan
