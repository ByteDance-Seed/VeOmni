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
) -> List[List[torch.Tensor]]:
    """
    A greedy bin-packing sorting algorithm designed for encoder data balance.
    It initializes a number of bins equal to the dp group size, and iteratively assigns data (sorted in descending order
    based on data length) to the bin with the smallest current load.

    The load of a bin is defined as the sum of the lengths^2 of its elements

    Args:
        all_data_lengths: the length information of data gathered from all dp ranks
        num_replicas: the size of dp group
        dim: the dimension along with the data in all_data_lengths is used for sorting

    Returns:
        a list that contains ${dp group size} buckets, where each bucket stores the sequence length and coordinate of
        the data assigned to the respective dp rank after balancing
    """
    # Note: AiCore does not support dtype int32 or int 64 for argsort
    sort_indice = torch.argsort(all_data_lengths[:, dim].float(), descending=True)
    all_data_lengths = all_data_lengths[sort_indice]
    lengths_per_sequence = (all_data_lengths[:, dim] ** 2).cpu()

    pre_fill_num = min(num_replicas, len(all_data_lengths))
    balanced_image_dp_batch = [[all_data_lengths[i]] if i < pre_fill_num else [] for i in range(num_replicas)]

    # Use a min-heap keyed on (accumulated load, dp rank) to pick the least-loaded bin in O(log n)
    # instead of an O(n) argmin scan per assignment. The rank tie-breaker preserves the argmin behavior
    # of selecting the lowest-index bin when loads are equal.
    load_heap = [(int(lengths_per_sequence[i]) if i < pre_fill_num else 0, i) for i in range(num_replicas)]
    heapq.heapify(load_heap)

    for i, sequence_lentgh in enumerate(all_data_lengths[pre_fill_num:]):
        target_load, target_dp_group = heapq.heappop(load_heap)
        balanced_image_dp_batch[target_dp_group].extend([sequence_lentgh])
        heapq.heappush(load_heap, (target_load + int(lengths_per_sequence[i + num_replicas]), target_dp_group))

    return balanced_image_dp_batch


SORTING_ALGO_FUNC = {
    "post_mbs_balancing_greedy_without_pad": post_mbs_balancing_greedy_without_pad,
}
