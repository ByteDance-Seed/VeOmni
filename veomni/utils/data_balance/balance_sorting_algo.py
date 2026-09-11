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
from typing import List

import torch


@torch.no_grad()
def post_mbs_balancing_greedy_without_pad(
    all_data_lengths: torch.Tensor,
    num_replicas: int,
    dim: int,
) -> List[torch.Tensor]:
    """
    A greedy bin-packing sorting algorithm designed for encoder data balance.
    It initializes a number of bins equal to the dp group size, and iteratively assigns data (sorted in descending order
    based on data length) to the bin with the smallest current load.

    The load of a bin is defined as the sum of the lengths^2 of its elements.

    The bin with the smallest load is tracked with a min-heap keyed on ``(accumulated_load, dp_rank)``. This costs
    ``O(log num_replicas)`` per assignment instead of the ``O(num_replicas)`` scan an ``argmin`` would take, and the
    ``dp_rank`` tie-breaker keeps assignments deterministic by preferring the lowest rank when loads are equal. The
    greedy scheduling runs on host-side Python scalars, so the sorted rows are materialized on CPU up front and the
    per-rank buckets are rebuilt into device tensors only once at the end.

    Args:
        all_data_lengths: the length information of data gathered from all dp ranks
        num_replicas: the size of dp group
        dim: the dimension along with the data in all_data_lengths is used for sorting

    Returns:
        a list of ${dp group size} tensors, where each tensor stores the sequence length and coordinate of the data
        assigned to the respective dp rank after balancing
    """
    # Note: AiCore does not support dtype int32 or int 64 for argsort. Sort descending by length, then move the rows to
    # host so the greedy loop below works on cheap Python scalars rather than issuing per-item device ops.
    sort_indice = torch.argsort(all_data_lengths[:, dim].float(), descending=True)
    sorted_rows = all_data_lengths[sort_indice].cpu().tolist()

    # Seed one bin per rank; the largest items pre-fill the first `pre_fill_num` bins so every rank starts non-empty.
    pre_fill_num = min(num_replicas, len(sorted_rows))
    buckets = [[row] for row in sorted_rows[:pre_fill_num]]
    buckets.extend([] for _ in range(num_replicas - pre_fill_num))

    load_heap = [(row[dim] ** 2, dp_rank) for dp_rank, row in enumerate(sorted_rows[:pre_fill_num])]
    load_heap.extend((0, dp_rank) for dp_rank in range(pre_fill_num, num_replicas))
    heapq.heapify(load_heap)

    # Assign each remaining item to the currently least-loaded rank and push its updated load back onto the heap.
    for row in sorted_rows[pre_fill_num:]:
        load, target_dp_rank = heapq.heappop(load_heap)
        buckets[target_dp_rank].append(row)
        heapq.heappush(load_heap, (load + row[dim] ** 2, target_dp_rank))

    # Flatten the buckets into a single contiguous tensor and split it back per rank, so the caller receives one device
    # tensor per rank instead of a list of per-item tensors.
    bucket_sizes = [len(bucket) for bucket in buckets]
    rank_table = torch.tensor(
        [row for bucket in buckets for row in bucket],
        dtype=all_data_lengths.dtype,
        device=all_data_lengths.device,
    )
    return list(rank_table.split(bucket_sizes))


SORTING_ALGO_FUNC = {
    "post_mbs_balancing_greedy_without_pad": post_mbs_balancing_greedy_without_pad,
}
