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
    cost_exponent: Optional[float] = None,
    cost_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> List[torch.Tensor]:
    """
    A greedy bin-packing sorting algorithm designed for encoder data balance.
    It initializes a number of bins equal to the dp group size, and iteratively assigns data (sorted in descending order
    based on data length) to the bin with the smallest current load.

    The default load is the sum of lengths^2 (legacy ViT behavior). A module can
    provide a different exponent or a callable returning one scheduling cost per
    input row. Custom costs are sorted descending and accumulated directly;
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
        cost_exponent: nonnegative finite exponent; omitted means integer 2. An
            integer exponent keeps integer costs exact; a float exponent uses floats.
        cost_fn: callable receiving a cloned item table and dim, returning [N]
            or [N, 1] scalar costs. [N, K] with K > 1 is reserved for vector
            scheduling and raises NotImplementedError. Must be deterministic
            across ranks; cannot be combined with an explicit cost_exponent.

    Returns:
        a list of ${dp group size} tensors, where each tensor stores the sequence length and coordinate of the data
        assigned to the respective dp rank after balancing
    """
    if all_data_lengths.ndim != 2 or not 0 <= dim < all_data_lengths.shape[1] or num_replicas < 1:
        raise ValueError("Expected a 2D item table, valid length column, and positive replica count.")
    lengths = all_data_lengths[:, dim]
    if lengths.is_complex():
        raise ValueError("Item lengths must be finite and nonnegative.")
    if cost_fn is not None and cost_exponent is not None:
        raise ValueError("Specify either cost_exponent or cost_fn, not both.")
    exponent = 2 if cost_exponent is None else cost_exponent
    if not math.isfinite(exponent) or exponent < 0:
        raise ValueError("Cost exponent must be finite and nonnegative.")
    if cost_fn is None:
        # Preserve the legacy device sort, including equal-length tie ordering.
        sort_indice = torch.argsort(lengths.float(), descending=True)
        order = None
    else:
        costs = cost_fn(all_data_lengths.clone(), dim)
        if not isinstance(costs, torch.Tensor) or costs.ndim not in (1, 2) or costs.shape[0] != lengths.shape[0]:
            raise ValueError("Cost callable must return an [N] or [N, K] tensor.")
        if costs.ndim == 2:
            if costs.shape[1] != 1:
                raise NotImplementedError("Vector cost scheduling (K != 1) is not implemented.")
            costs = costs[:, 0]
        if costs.is_complex():
            raise ValueError("Scheduling costs must be finite and nonnegative.")
        cost_values = costs.cpu().tolist()
        if any(not math.isfinite(cost) or cost < 0 for cost in cost_values):
            raise ValueError("Scheduling costs must be finite and nonnegative.")
        order = sorted(range(len(cost_values)), key=lambda i: (-cost_values[i], i))
        sort_indice = torch.tensor(order, dtype=torch.long, device=all_data_lengths.device)

    sorted_rows = all_data_lengths[sort_indice].cpu().tolist()
    sorted_lengths = [row[dim] for row in sorted_rows]
    if any(not math.isfinite(length) or length < 0 for length in sorted_lengths):
        raise ValueError("Item lengths must be finite and nonnegative.")
    sorted_costs = (
        [length**exponent for length in sorted_lengths] if order is None else [cost_values[i] for i in order]
    )
    if any(isinstance(cost, float) and not math.isfinite(cost) for cost in sorted_costs):
        raise ValueError("Scheduling costs must be finite.")

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
