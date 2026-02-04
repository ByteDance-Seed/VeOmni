import random

import torch

from veomni.utils.data_balance.balance_sorting_algo import SORTING_ALGO_FUNC
from veomni.utils.device import get_device_type


FAKE_WORLD_SIZE = 8
WORKLOAD_CAL_RULE = {"s2": lambda x: x**2}


def fake_data_construct():
    # Construct fake data for each dp rank (set world size = 8)

    # Simulate all data all gather from each dp rank after data balancing
    balanced_fake_data_lengths_per_dp = [
        torch.arange(start=1, end=6, dtype=torch.long, device=get_device_type()) * 200 for _ in range(FAKE_WORLD_SIZE)
    ]

    # Simulate unbalanced data on each dp rank based on 'balanced_fake_data_lengths_per_dp'
    def random_partition(total_num, n_parts):
        """Randomly divides 'total_num' into n parts such that the sum of the parts equals 'total_num'"""
        if total_num < n_parts:
            raise ValueError("Total elements must be at least the number of parts.")
        splits = sorted(random.sample(range(1, total_num), n_parts - 1))
        split_lengths = (
            [splits[0]] + [splits[i] - splits[i - 1] for i in range(1, len(splits))] + [total_num - splits[-1]]
        )
        return split_lengths

    all_fake_data = torch.cat(balanced_fake_data_lengths_per_dp)
    all_fake_data_shuffle = all_fake_data[torch.randperm(len(all_fake_data))]
    random_split = random_partition(len(all_fake_data_shuffle), FAKE_WORLD_SIZE)
    unbalanced_fake_data_lengths_per_dp = all_fake_data_shuffle.split(random_split)

    return unbalanced_fake_data_lengths_per_dp, balanced_fake_data_lengths_per_dp


def check_balance_sorting(rank_table, balanced_data_gt, role="s2"):
    # Calculate workload of ground truth data
    gt_workloads = [WORKLOAD_CAL_RULE[role](gt).sum() for gt in balanced_data_gt]

    # Check whether the load matches the given ground truth
    # Calculate the workload of current rank
    rank_table_cur_rank_wl = [WORKLOAD_CAL_RULE[role](torch.cat(rt)).sum() for rt in rank_table]

    # Since the ground truth is a manually constructed perfectly balanced data distribution,
    # the load on each DP group after applying the reordering algorithm must exactly match that of the ground truth;
    # otherwise, the reordering algorithm is considered to underperform.
    torch.testing.assert_close(rank_table_cur_rank_wl, gt_workloads)
    print("check pass!")


def test_post_mbs_balancing_greedy_without_pad_s2():
    unbalanced_fake_data_lengths_per_dp, balanced_fake_data_lengths_per_dp = fake_data_construct()

    # Use sorting algorithm to process lengths
    test_func = SORTING_ALGO_FUNC["post_mbs_balancing_greedy_without_pad"]
    rank_table = test_func(
        torch.cat(unbalanced_fake_data_lengths_per_dp).unsqueeze(-1), num_replicas=FAKE_WORLD_SIZE, dim=0
    )

    check_balance_sorting(rank_table, balanced_fake_data_lengths_per_dp)
