import random

import torch

from veomni.utils.data_balance.balance_sorting_algo import SORTING_ALGO_FUNC
from veomni.utils.data_balance.data_balance import Qwen3VLEncoderDataBalance
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


def _reference_greedy(all_data_lengths, num_replicas, dim):
    # Original tensor-argmin implementation, kept as the ground truth for the heap scheduler.
    sort_indice = torch.argsort(all_data_lengths[:, dim].float(), descending=True)
    all_data_lengths = all_data_lengths[sort_indice]
    lengths_per_sequence = (all_data_lengths[:, dim] ** 2).cpu()

    pre_fill_num = min(num_replicas, len(all_data_lengths))
    dp_group_total_length = torch.empty(num_replicas, dtype=torch.long)
    dp_group_total_length[:pre_fill_num] = lengths_per_sequence[:pre_fill_num]
    balanced_image_dp_batch = [[all_data_lengths[i]] if i < pre_fill_num else [] for i in range(num_replicas)]

    for i, sequence_length in enumerate(all_data_lengths[pre_fill_num:]):
        target_dp_group = dp_group_total_length.argmin()
        balanced_image_dp_batch[target_dp_group].extend([sequence_length])
        dp_group_total_length[target_dp_group] += lengths_per_sequence[i + num_replicas]

    return balanced_image_dp_batch


def test_post_mbs_balancing_greedy_matches_tensor_argmin_reference():
    device = get_device_type()
    source_rank = torch.arange(32, device=device) % 4
    source_index = torch.arange(32, device=device)
    lengths = torch.tensor(
        [
            256,
            9408,
            784,
            5120,
            320,
            8960,
            640,
            3072,
            1120,
            8624,
            768,
            4096,
            448,
            7296,
            1280,
            5760,
            384,
            9072,
            1728,
            3520,
            2560,
            4800,
            640,
            8160,
            896,
            5440,
            1536,
            6800,
            320,
            9216,
            2304,
            784,
        ],
        device=device,
    )
    all_data_lengths = torch.stack((source_rank, source_index, lengths), dim=1)

    rank_table = SORTING_ALGO_FUNC["post_mbs_balancing_greedy_without_pad"](all_data_lengths, num_replicas=4, dim=2)
    reference = _reference_greedy(all_data_lengths, num_replicas=4, dim=2)

    # The heap scheduler must produce the same per-bucket assignment as the tensor-argmin reference.
    assert all(
        torch.equal(torch.stack(actual), torch.stack(expected)) for actual, expected in zip(rank_table, reference)
    )
    data_list, normalized_table = Qwen3VLEncoderDataBalance.rank_table_mapping(rank_table, dp_rank=2)
    assert all(torch.equal(actual, torch.stack(expected)) for actual, expected in zip(normalized_table, reference))
    assert all(index.dtype == torch.long for index in data_list)
