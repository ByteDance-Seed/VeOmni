# Sorting algorithm for data balance
import torch


def post_mbs_balancing_greedy_without_pad(
        global_data_length: torch.Tensor,
        num_replicas: int,
) -> list[list[tuple[int, int, torch.Tensor]]]:
    sort_indice = torch.argsort(global_data_length[:, 2], descending=True)
    global_data_length = global_data_length[sort_indice]
    lengths_per_sequence = (global_data_length[:, 2] ** 2).cpu()

    dp_group_total_length = torch.empty(num_replicas, dtype=torch.long)
    dp_group_total_length[:] = lengths_per_sequence[: num_replicas]
    balanced_image_dp_batch = [[global_data_length[i]] for i in range(num_replicas)]

    for i, sequence_lentgh in enumerate(global_data_length[num_replicas:]):
        target_dp_group = dp_group_total_length.argmin()
        balanced_image_dp_batch[target_dp_group].extend([sequence_lentgh])
        dp_group_total_length[target_dp_group] += lengths_per_sequence[i + num_replicas]

    return balanced_image_dp_batch


SORTING_ALGO_FUNC = {
    'post_mbs_balancing_greedy_without_pad': post_mbs_balancing_greedy_without_pad,
}
