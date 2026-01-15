import torch
import torch.nn.functional as F
import torch.distributed as dist

from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils.data_balance.balance_sorting_algo import SORTING_ALGO_FUNC


class EncoderDataBalance:
    def __init__(
            self,
            sorting_algo_name="post_mbs_balancing_greedy_without_pad",
            spatial_merge_unit=None
    ):
        self.state_buffer = {}
        self.merge_down_ratio = spatial_merge_unit
        self.sorting_algo = self._set_sorting_algo(sorting_algo_name)
        self.dp_group = get_parallel_state().dp_group

    def balance_data(self, pixel_values, grid_thw):
        # split input data
        split_batch = {}
        pixel_values_length = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2])
        split_batch['pixel_values'] = pixel_values.npu().split(pixel_values_length.tolist(), dim=0)
        split_batch['image_grid_thw'] = grid_thw.npu()

        split_lengths = [
            pixel_values_length.npu(),
            torch.empty(
                pixel_values_length.shape[0], dtype=torch.long, device='npu'
            ).fill_(grid_thw.shape[-1])
        ]
        balanced_datas = self.all_to_all_redistribution(
            data_lengths=torch.stack(split_lengths, dim=-1),
            datas=split_batch,
            data_type="image",
        )
        balanced_grid_thw = torch.cat(balanced_datas['image_grid_thw'])
        balanced_pixel_values = torch.cat(balanced_datas['pixel_values'])

        return balanced_pixel_values, balanced_grid_thw

    def data_bridge(self, hidden_state, deepstack_feature_lists):
        recoverd_hidden_state = self._recover_image_data(hidden_state)
        if deepstack_feature_lists:
            recovered_deepstack_feature_lists = [
                self._recover_image_data(df)
                for df in deepstack_feature_lists
            ]
        else:
            recovered_deepstack_feature_lists = []

        return recoverd_hidden_state, recovered_deepstack_feature_lists

    def _recover_image_data(self, hidden_state):
        recovered_hidden_state = self.all_to_all_communication(
            list(hidden_state.split(self.state_buffer['image']["pixel_values_split"])),
            self.state_buffer['image']["pixel_values_origin"],
            (hidden_state.shape[-1],),
            self.dp_group
        )
        recovered_hidden_state = torch.cat(recovered_hidden_state).split(
            self.state_buffer['image']["image_grid_thw_origin_split"])
        origin_hidden_state = [None] * len(recovered_hidden_state)
        for i, idx in enumerate(self.state_buffer['image']['pixel_values_data_list']):
            origin_hidden_state[idx] = recovered_hidden_state[i]

        return torch.cat(origin_hidden_state)

    @staticmethod
    def _set_sorting_algo(sorting_algo_name):
        return SORTING_ALGO_FUNC[sorting_algo_name]

    def all_to_all_redistribution(
            self,
            data_lengths: torch.Tensor,
            datas: dict[str, torch.Tensor],
            data_type='Unknown data',
            **kwargs):
        dp_rank = self.dp_group.rank()
        num_replicas = self.dp_group.size()

        cur_bs = torch.tensor(data_lengths.shape[0], dtype=torch.long, device=data_lengths.device)
        all_gather_bs = [torch.empty(1, dtype=torch.long, device=data_lengths.device) for _ in range(num_replicas)]
        dist.all_gather(all_gather_bs, cur_bs, group=self.dp_group)

        gathered_lengths = [
            torch.empty((all_gather_bs[i], *data_lengths.shape[1:]), dtype=data_lengths.dtype, device=data_lengths.device)
            for i in range(num_replicas)
        ]
        dist.all_gather(gathered_lengths, data_lengths, group=self.dp_group)

        samples_lengths = [
            F.pad(torch.cat([
                torch.arange(
                    len(batch), dtype=batch.dtype, device=batch.device
                ).view(-1, 1),
                batch
            ], dim=-1), pad=(1, 0), value=i)
            for i, batch in enumerate(gathered_lengths)
        ]
        samples_lengths = torch.cat(samples_lengths)

        rank_table = self.sorting_algo(
            samples_lengths, num_replicas,
        )
        data_list, rank_table = self.rank_table_mapping(rank_table, dp_rank)

        balance_data_mapping_index = [torch.where(rank_table[dp_rank][:, 0] == r)[0] for r in range(num_replicas)]
        self.state_buffer[data_type] = {
            'balance_data_mapping_index': torch.cat(balance_data_mapping_index),
        }
        balanced_datas = {}
        balanced_data_lengths = torch.empty(
            num_replicas, 2, dtype=rank_table[dp_rank].dtype, device=rank_table[dp_rank].device
        )
        sample_num_per_rank = torch.bincount(rank_table[dp_rank][:, 0], minlength=num_replicas)
        for i, (data_name, data) in enumerate(datas.items()):
            reorganized_data = self.data_reorganization(data, data_list)
            balanced_data_dim = ()

            if data_name != 'pixel_values':
                balanced_data_dim = (*data[0].shape[1:],)
                balanced_data_lengths[:, 0] = sample_num_per_rank
                balanced_data_lengths[:, 1] = torch.stack([l[0, i] for l in gathered_lengths])
                origin_data = torch.cat(reorganized_data)
                self.state_buffer[data_type][f"{data_name}_origin_split"] = (
                        origin_data[:, 0] * origin_data[:, 1] * origin_data[:, 2] // self.merge_down_ratio).tolist()
            else:
                balanced_data_lengths[:, 0] = 0
                balanced_data_lengths[:, 0].index_add_(0, rank_table[dp_rank][:, 0], rank_table[dp_rank][:, 2 + i])
                balanced_data_lengths[:, 1] = data[0].shape[-1]
                self.state_buffer[data_type][f"{data_name}_split"] = (
                        balanced_data_lengths[:, 0] // self.merge_down_ratio).tolist()
                self.state_buffer[data_type][f"{data_name}_origin"] = [
                    (d.shape[0] // self.merge_down_ratio, )
                    for d in reorganized_data
                ]
                self.state_buffer[data_type][f"{data_name}_data_list"] = torch.cat(data_list)
            balanced_data = self.all_to_all_communication(
                reorganized_data, balanced_data_lengths, balanced_data_dim, self.dp_group)
            balanced_datas[data_name] = balanced_data

        return balanced_datas

    def reverse_all_to_all_redistribution(self):
        pass

    @staticmethod
    def rank_table_mapping(rank_table, dp_rank):
        rank_table = [torch.stack(rt) for rt in rank_table]
        rank_table_for_current_rank = [rt[rt[:, 0] == dp_rank][:, 1] for rt in rank_table]

        return rank_table_for_current_rank, rank_table

    @staticmethod
    def data_reorganization(data, data_list):
        if isinstance(data, torch.Tensor):
            new_data_group_per_rank = [data[new_group_idxs] for new_group_idxs in data_list]
        else:
            new_data_group_per_rank = [
                torch.cat([data[idx] for idx in new_group_idxs])
                if new_group_idxs.numel() != 0
                else torch.tensor([], dtype=data[0].dtype, device=data[0].device)
                for new_group_idxs in data_list
            ]
        return new_data_group_per_rank

    @staticmethod
    def all_to_all_communication(data, balanced_data_lengths, data_dim, dp_process_group):
        balanced_data_cache = [
            torch.empty(
                (*new_length, *data_dim), dtype=data[0].dtype, device=data[0].device
            ).squeeze(-1) for new_length in balanced_data_lengths
        ]
        dist.all_to_all(balanced_data_cache, data, group=dp_process_group)
        return balanced_data_cache
