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

# Sorting algorithm for data balance
import heapq
import math
from typing import Callable, List, Optional

import torch


@torch.no_grad()
def post_mbs_balancing_greedy_without_pad(
    all_data_lengths: torch.Tensor,
    num_replicas: int,
    dim: int,
    *,
    cost_exponent: float = 2,
    cost_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> List[torch.Tensor]:
    """
    A greedy bin-packing sorting algorithm designed for encoder data balance.
    It initializes a number of bins equal to the dp group size, and iteratively assigns data (sorted in descending order
    based on data length) to the bin with the smallest current load.

    The default load is the sum of lengths^2 (legacy ViT behavior). A module can
    provide a different exponent or a callable returning one scheduling cost per
    input length. Custom costs are sorted descending and accumulated directly;
    they are not squared again. Neither costs nor assignments redefine token counts.

    The bin with the smallest load is tracked with a min-heap keyed on ``(accumulated_load, dp_rank)``. This costs
    ``O(log num_replicas)`` per assignment instead of the ``O(num_replicas)`` scan an ``argmin`` would take, and the
    ``dp_rank`` tie-breaker keeps assignments deterministic by preferring the lowest rank when loads are equal. The
    greedy scheduling runs on host-side Python scalars, so the sorted rows are materialized on CPU up front and the
    per-rank buckets are rebuilt into device tensors only once at the end.

    Args:
        all_data_lengths: the length information of data gathered from all dp ranks
        num_replicas: the size of dp group
        dim: the dimension along with the data in all_data_lengths is used for sorting
        cost_exponent: nonnegative finite exponent; 1 models linear token cost
        cost_fn: optional callable from the length vector to a same-shaped cost
            tensor; overrides cost_exponent. It must be deterministic across ranks.

    Returns:
        a list of ${dp group size} tensors, where each tensor stores the sequence length and coordinate of the data
        assigned to the respective dp rank after balancing
    """
    if all_data_lengths.ndim != 2 or not 0 <= dim < all_data_lengths.shape[1] or num_replicas < 1:
        raise ValueError("Expected a 2D item table, valid length column, and positive replica count.")
    lengths = all_data_lengths[:, dim]
    if lengths.is_complex() or not bool(torch.isfinite(lengths).all()) or bool((lengths < 0).any()):
        raise ValueError("Item lengths must be finite and nonnegative.")
    if cost_fn is None and (not math.isfinite(cost_exponent) or cost_exponent < 0):
        raise ValueError("Cost exponent must be finite and nonnegative.")
    if cost_fn is not None:
        costs = cost_fn(lengths)
        if not isinstance(costs, torch.Tensor) or costs.shape != lengths.shape:
            raise ValueError("Cost callable must return a tensor with one cost per item.")
        if costs.is_complex() or not bool(torch.isfinite(costs).all()) or bool((costs < 0).any()):
            raise ValueError("Scheduling costs must be finite and nonnegative.")
        # Custom costs can be non-monotone in length. Sort host scalars so large
        # integer costs do not become false ties when cast to float32 on AiCore.
        cost_values = costs.cpu().tolist()
        order = sorted(range(len(cost_values)), key=lambda i: (-cost_values[i], i))
        sort_indice = torch.tensor(order, dtype=torch.long, device=all_data_lengths.device)
    else:
        # Keep the exact legacy tie ordering for every existing caller. All
        # nonnegative powers are monotone, so length order is also cost order.
        sort_indice = torch.argsort(lengths.float(), descending=True)
        try:
            cost_values = [length**cost_exponent for length in lengths.cpu().tolist()]
            if any(not math.isfinite(cost) for cost in cost_values):
                raise ValueError("Scheduling costs must be finite.")
        except OverflowError as exc:
            raise ValueError("Scheduling costs must be finite.") from exc
    # Note: AiCore does not support dtype int32 or int 64 for argsort. Sort descending by length, then move the rows to
    # host so the greedy loop below works on cheap Python scalars rather than issuing per-item device ops.
    sorted_rows = all_data_lengths[sort_indice].cpu().tolist()
    sorted_costs = [cost_values[i] for i in sort_indice.cpu().tolist()]

    # Seed one bin per rank; the largest items pre-fill the first `pre_fill_num` bins so every rank starts non-empty.
    pre_fill_num = min(num_replicas, len(sorted_rows))
    buckets = [[row] for row in sorted_rows[:pre_fill_num]]
    buckets.extend([] for _ in range(num_replicas - pre_fill_num))

    load_heap = [(cost, dp_rank) for dp_rank, cost in enumerate(sorted_costs[:pre_fill_num])]
    load_heap.extend((0, dp_rank) for dp_rank in range(pre_fill_num, num_replicas))
    heapq.heapify(load_heap)

    # Assign each remaining item to the currently least-loaded rank and push its updated load back onto the heap.
    for row, cost in zip(sorted_rows[pre_fill_num:], sorted_costs[pre_fill_num:]):
        load, target_dp_rank = heapq.heappop(load_heap)
        buckets[target_dp_rank].append(row)
        heapq.heappush(load_heap, (load + cost, target_dp_rank))

    # Flatten the buckets into a single contiguous tensor and split it back per rank, so the caller receives one device
    # tensor per rank instead of a list of per-item tensors.
    bucket_sizes = [len(bucket) for bucket in buckets]
    rank_table = torch.tensor(
        [row for bucket in buckets for row in bucket],
        dtype=all_data_lengths.dtype,
        device=all_data_lengths.device,
    ).reshape(-1, all_data_lengths.shape[1])
    return list(rank_table.split(bucket_sizes))


SORTING_ALGO_FUNC = {
    "post_mbs_balancing_greedy_without_pad": post_mbs_balancing_greedy_without_pad,
}
