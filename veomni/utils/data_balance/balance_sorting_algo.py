# Sorting algorithm for data balance
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
    dp_group_total_length = torch.empty(num_replicas, dtype=torch.long)
    dp_group_total_length[:pre_fill_num] = lengths_per_sequence[:pre_fill_num]
    balanced_image_dp_batch = [[all_data_lengths[i]] if i < pre_fill_num else [] for i in range(num_replicas)]

    for i, sequence_lentgh in enumerate(all_data_lengths[pre_fill_num:]):
        target_dp_group = dp_group_total_length.argmin()
        balanced_image_dp_batch[target_dp_group].extend([sequence_lentgh])
        dp_group_total_length[target_dp_group] += lengths_per_sequence[i + num_replicas]

    return balanced_image_dp_batch


SORTING_ALGO_FUNC = {
    "post_mbs_balancing_greedy_without_pad": post_mbs_balancing_greedy_without_pad,
}
