import random

import pytest
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
    # Each rank_table entry is a tensor of the rows assigned to that rank; sum length^2 over its rows.
    rank_table_cur_rank_wl = [WORKLOAD_CAL_RULE[role](rt).sum() for rt in rank_table]

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
    # Original tensor-argmin implementation, kept as the ground truth for the heap scheduler. Returns one stacked
    # tensor per rank to match the current sorting-algo output.
    sort_indice = torch.argsort(all_data_lengths[:, dim].float(), descending=True)
    all_data_lengths = all_data_lengths[sort_indice]
    lengths_per_sequence = (all_data_lengths[:, dim] ** 2).cpu()

    pre_fill_num = min(num_replicas, len(all_data_lengths))
    dp_group_total_length = torch.empty(num_replicas, dtype=torch.long)
    dp_group_total_length[:pre_fill_num] = lengths_per_sequence[:pre_fill_num]
    buckets = [[all_data_lengths[i]] if i < pre_fill_num else [] for i in range(num_replicas)]

    for i, sequence_length in enumerate(all_data_lengths[pre_fill_num:]):
        target_dp_group = dp_group_total_length.argmin()
        buckets[target_dp_group].append(sequence_length)
        dp_group_total_length[target_dp_group] += lengths_per_sequence[i + num_replicas]

    return [torch.stack(bucket) for bucket in buckets]


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

    # The heap scheduler must produce the same per-rank assignment as the tensor-argmin reference.
    assert all(torch.equal(actual, expected) for actual, expected in zip(rank_table, reference))
    data_list, normalized_table = Qwen3VLEncoderDataBalance.rank_table_mapping(rank_table, dp_rank=2)
    assert all(torch.equal(actual, expected) for actual, expected in zip(normalized_table, reference))
    assert all(index.dtype == torch.long for index in data_list)


@pytest.mark.parametrize("lengths,replicas", [([], 8), ([7], 8), ([0, 0], 4), ([9, 9, 9, 9], 2)])
def test_sorter_empty_and_small_batches(lengths, replicas):
    table = torch.tensor([[i, n] for i, n in enumerate(lengths)], dtype=torch.long).reshape(-1, 2)
    result = SORTING_ALGO_FUNC["post_mbs_balancing_greedy_without_pad"](table, replicas, dim=1)
    assert len(result) == replicas
    assert all(bucket.ndim == 2 and bucket.shape[1] == 2 for bucket in result)
    assert all(bucket.dtype == table.dtype and bucket.device == table.device for bucket in result)
    recovered = torch.cat(result)
    assert torch.equal(recovered[torch.argsort(recovered[:, 0])], table)


@pytest.mark.parametrize("device", list(dict.fromkeys(["cpu", get_device_type()])))
def test_linear_and_quadratic_costs_assign_differently(device):
    table = torch.tensor([[i, n] for i, n in enumerate([10, 9, 8, 7, 6])], device=device)
    sorter = SORTING_ALGO_FUNC["post_mbs_balancing_greedy_without_pad"]
    quadratic = sorter(table, 2, 1)
    linear = sorter(table, 2, 1, cost_exponent=1)
    callable_linear = sorter(table, 2, 1, cost_fn=lambda table, dim: table[:, dim])
    assert [b[:, 0].tolist() for b in quadratic] == [[0, 3], [1, 2, 4]]
    assert [b[:, 0].tolist() for b in linear] == [[0, 3, 4], [1, 2]]
    assert all(torch.equal(a, b) for a, b in zip(linear, callable_linear))


@pytest.mark.parametrize("device", list(dict.fromkeys(["cpu", get_device_type()])))
def test_custom_costs_not_sorted_by_length_or_squared_twice(device):
    table = torch.tensor([[i, n] for i, n in enumerate([1, 2, 3, 4, 5])], device=device)
    result = SORTING_ALGO_FUNC["post_mbs_balancing_greedy_without_pad"](
        table,
        2,
        1,
        cost_fn=lambda table, dim: torch.tensor([10, 9, 8, 7, 6], device=table.device),
    )
    assert [b[:, 0].tolist() for b in result] == [[0, 3, 4], [1, 2]]


@pytest.mark.parametrize("device", list(dict.fromkeys(["cpu", get_device_type()])))
def test_custom_large_integer_costs_preserve_order(device):
    table = torch.tensor([[0, 1], [1, 1]], device=device)
    result = SORTING_ALGO_FUNC["post_mbs_balancing_greedy_without_pad"](
        table,
        2,
        1,
        cost_fn=lambda table, dim: torch.tensor([2**60, 2**60 + 1], device=table.device),
    )
    assert [b[:, 0].tolist() for b in result] == [[1], [0]]


@pytest.mark.parametrize("exponent", [-1, float("inf"), float("nan")])
def test_invalid_exponent(exponent):
    with pytest.raises(ValueError):
        SORTING_ALGO_FUNC["post_mbs_balancing_greedy_without_pad"](
            torch.tensor([[1]]),
            2,
            0,
            cost_exponent=exponent,
        )


def test_exact_integer_cost_does_not_overflow():
    table = torch.tensor([[1000], [999], [998]])
    result = SORTING_ALGO_FUNC["post_mbs_balancing_greedy_without_pad"](table, 2, 0, cost_exponent=200)
    assert [bucket[:, 0].tolist() for bucket in result] == [[1000], [999, 998]]


@pytest.mark.parametrize("exponent", [2, 1, -3])
def test_callable_and_explicit_exponent_are_mutually_exclusive(exponent):
    with pytest.raises(ValueError, match="not both"):
        SORTING_ALGO_FUNC["post_mbs_balancing_greedy_without_pad"](
            torch.ones(2, 1), 2, 0, cost_exponent=exponent, cost_fn=lambda table, dim: table[:, dim]
        )


def test_callable_receives_independent_full_table():
    table = torch.tensor([[0, 1, 10], [1, 2, 20]])

    def cost_fn(rows, dim):
        assert dim == 1
        rows[:, dim] = 0
        return rows[:, 2:3]

    result = SORTING_ALGO_FUNC["post_mbs_balancing_greedy_without_pad"](table, 2, 1, cost_fn=cost_fn)
    assert torch.equal(table, torch.tensor([[0, 1, 10], [1, 2, 20]]))
    assert [bucket[:, 0].tolist() for bucket in result] == [[1], [0]]
    assert torch.equal(torch.cat(result)[[1, 0]], table)


def test_vector_costs_are_explicitly_reserved():
    with pytest.raises(NotImplementedError, match="Vector cost"):
        SORTING_ALGO_FUNC["post_mbs_balancing_greedy_without_pad"](
            torch.ones(2, 2), 2, 0, cost_fn=lambda table, dim: table
        )


@pytest.mark.parametrize(
    "costs",
    [torch.tensor([-1.0]), torch.tensor([float("nan")]), torch.tensor([float("inf")]), torch.ones(1, 1, 1), [1]],
)
def test_invalid_callable_costs(costs):
    with pytest.raises(ValueError):
        SORTING_ALGO_FUNC["post_mbs_balancing_greedy_without_pad"](
            torch.tensor([[1]]),
            2,
            0,
            cost_fn=lambda table, dim: costs,
        )


@pytest.mark.parametrize(
    "table,replicas,dim",
    [
        (torch.ones(2), 2, 0),
        (torch.ones(2, 1), 0, 0),
        (torch.ones(2, 1), 2, 1),
        (torch.tensor([[-1]]), 2, 0),
        (torch.tensor([[float("nan")]]), 2, 0),
    ],
)
def test_invalid_sorter_inputs(table, replicas, dim):
    with pytest.raises(ValueError):
        SORTING_ALGO_FUNC["post_mbs_balancing_greedy_without_pad"](table, replicas, dim)
