from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils import helper
from veomni.utils.data_balance.balance_sorting_algo import SORTING_ALGO_FUNC


logger = helper.create_logger(__name__)


class Qwen3VLEncoderDataBalance:
    def __init__(
        self,
        spatial_merge_unit: int,
        sorting_algo_name: str = "post_mbs_balancing_greedy_without_pad",
    ):
        """
        A data balance algorithm for Qwen3 VL encoder. The algorithm provides two interface functions:
            1. balance_data: performs data balance across dp group on the input data "pixel_values" and "grid_thw"
            2. data_bridge: restores the encoder output (hidden_states and deepstack_feature_lists) to their original dp group, ensuring correct downstream LLM computation

        Args:
            spatial_merge_unit: should be equal to encoder.spatial_merge_unit. This parameter is used to adapt the shape of encoder's output.
            sorting_algo_name: choose the sorting algorithm to use.
        """
        logger.info_rank0("Initializing Qwen3 vl encoder data balance...")
        self.state_buffer = defaultdict(dict)
        self.merge_down_ratio = spatial_merge_unit
        self.sorting_algo = self._set_sorting_algo(sorting_algo_name)
        self.dp_group = get_parallel_state().dp_group
        logger.info_rank0("Successfully initialized Qwen3 vl encoder data balance")

    def balance_data(
        self, pixel_values: Optional[torch.Tensor], grid_thw: Optional[torch.Tensor], data_type: str = "image"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.state_buffer[data_type] = {}
        # split pixel value
        split_batch = {}
        # Unify the type of grid_thw to long. Normal type of grid thw is int64, while dummy grid thw is int32
        grid_thw = grid_thw.long()
        pixel_values_length = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        split_batch["pixel_values"] = pixel_values.split(pixel_values_length.tolist(), dim=0)
        split_batch["image_grid_thw"] = grid_thw

        # balanced pixel value
        balanced_datas = self.all_to_all_redistribution(
            data_lengths=pixel_values_length,
            datas=split_batch,
            data_type=data_type,
        )

        # reorganize data to vision encoder input form
        balanced_grid_thw = torch.cat(balanced_datas["image_grid_thw"])
        balanced_pixel_values = torch.cat(balanced_datas["pixel_values"])

        return balanced_pixel_values, balanced_grid_thw

    def data_bridge(
        self,
        hidden_state: Optional[torch.Tensor],
        deepstack_feature_lists: Optional[List],
        require_grad: bool = True,
        data_type: str = "image",
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        recoverd_hidden_state = self.reverse_all_to_all_redistribution(
            hidden_state, require_grad=require_grad, data_type=data_type
        )
        if deepstack_feature_lists:
            recovered_deepstack_feature_lists = [
                self.reverse_all_to_all_redistribution(df, require_grad=require_grad, data_type=data_type)
                for df in deepstack_feature_lists
            ]
        else:
            recovered_deepstack_feature_lists = []

        return recoverd_hidden_state, recovered_deepstack_feature_lists

    def all_to_all_redistribution(
        self, data_lengths: torch.Tensor, datas: Dict[str, torch.Tensor], data_type: str = "image"
    ) -> Dict[str, List[torch.Tensor]]:
        dp_rank = self.dp_group.rank()
        num_replicas = self.dp_group.size()

        # Get all the data length from each dp rank
        # Considering each dp group may have different number of image, we firstly get the number of images in each dp rank
        cur_bs = torch.tensor(data_lengths.shape[0], dtype=torch.long, device=data_lengths.device)
        all_gather_bs = [torch.empty(1, dtype=torch.long, device=data_lengths.device) for _ in range(num_replicas)]
        dist.all_gather(all_gather_bs, cur_bs, group=self.dp_group)
        # Then get all data lengths in each dp rank
        gathered_lengths = [
            torch.empty(
                (all_gather_bs[i], *data_lengths.shape[1:]), dtype=data_lengths.dtype, device=data_lengths.device
            )
            for i in range(num_replicas)
        ]
        dist.all_gather(gathered_lengths, data_lengths, group=self.dp_group)

        # Add coordinate for each sample, each sample's dim is:
        #     dim 0: dp rank
        #     dim 1: the position of sample within the corresponding dp rank
        #     dim 2: sample length
        samples_lengths = [
            F.pad(
                torch.cat(
                    [
                        torch.arange(len(batch), dtype=batch.dtype, device=batch.device).view(-1, 1),
                        batch.unsqueeze(-1),
                    ],
                    dim=-1,
                ),
                pad=(1, 0),
                value=i,
            )
            for i, batch in enumerate(gathered_lengths)
        ]
        samples_lengths = torch.cat(samples_lengths)
        # Redistribute all samples across dp groups using function "self.sorting_algo", ensuring balanced workload among the dp ranks
        rank_table = self.sorting_algo(samples_lengths, num_replicas, dim=2)
        # Based on the rank_table, determine the distribution of current rank's data (i.e. data_list) across the load-balanced dp ranks
        data_list, rank_table = self.rank_table_mapping(rank_table, dp_rank)

        balanced_datas = {}
        balanced_data_lengths = torch.empty(
            num_replicas, 2, dtype=rank_table[dp_rank].dtype, device=rank_table[dp_rank].device
        )
        sample_num_per_rank = torch.bincount(rank_table[dp_rank][:, 0], minlength=num_replicas)
        for i, (data_name, data) in enumerate(datas.items()):
            # Allocate data in current dp rank to the specified dp ranks according to the data_list
            reorganized_data = self.data_reorganization(data, data_list)
            # Get new all-data shapes of current rank to implement all-to-all operation
            # self.state_buffer store the information of shape, split, which will be used in reverse data distribution
            balanced_data_dim = ()
            balanced_data_lengths[:, 1] = data[0].shape[-1]
            if data_name != "pixel_values":
                balanced_data_dim = (*data[0].shape[1:],)
                balanced_data_lengths[:, 0] = sample_num_per_rank
                origin_data = torch.cat(reorganized_data)
                self.state_buffer[data_type][f"{data_name}_origin_split"] = (
                    origin_data[:, 0] * origin_data[:, 1] * origin_data[:, 2] // self.merge_down_ratio
                ).tolist()
            else:
                balanced_data_lengths[:, 0] = 0
                balanced_data_lengths[:, 0].index_add_(0, rank_table[dp_rank][:, 0], rank_table[dp_rank][:, 2 + i])
                self.state_buffer[data_type][f"{data_name}_split"] = (
                    balanced_data_lengths[:, 0] // self.merge_down_ratio
                ).tolist()
                self.state_buffer[data_type][f"{data_name}_origin"] = [
                    (d.shape[0] // self.merge_down_ratio,) for d in reorganized_data
                ]
                self.state_buffer[data_type][f"{data_name}_data_list"] = torch.cat(data_list)
            # execute all to all communication to redistribute data across dp ranks
            balanced_data = self.all_to_all_communication(reorganized_data, balanced_data_lengths, balanced_data_dim)
            balanced_datas[data_name] = balanced_data

        return balanced_datas

    def reverse_all_to_all_redistribution(
        self, hidden_state: torch.Tensor, require_grad: bool, data_type: str = "image"
    ) -> torch.Tensor:
        # Redistribute the data back to its original dp rank
        recovered_hidden_state = self.all_to_all_communication(
            list(hidden_state.split(self.state_buffer[data_type]["pixel_values_split"])),
            self.state_buffer[data_type]["pixel_values_origin"],
            (hidden_state.shape[-1],),
            require_grad=require_grad,
        )
        # Split the concatenated data and restoring the original ordering
        recovered_hidden_state = torch.cat(recovered_hidden_state).split(
            self.state_buffer[data_type]["image_grid_thw_origin_split"]
        )
        origin_hidden_state = [None] * len(recovered_hidden_state)
        for i, idx in enumerate(self.state_buffer[data_type]["pixel_values_data_list"]):
            origin_hidden_state[idx] = recovered_hidden_state[i]

        return torch.cat(origin_hidden_state)

    def all_to_all_communication(
        self, data: List[torch.Tensor], balanced_data_lengths: List[tuple], data_dim: tuple, require_grad=False
    ) -> List[torch.Tensor]:
        balanced_data_cache = [
            torch.empty((*new_length, *data_dim), dtype=data[0].dtype, device=data[0].device).squeeze(-1)
            for new_length in balanced_data_lengths
        ]
        if require_grad:
            # This API is an official API that supports backward, but incurs a slight additional overhead in execution time
            from torch.distributed.nn.functional import all_to_all
        else:
            from torch.distributed import all_to_all
        all_to_all(balanced_data_cache, data, group=self.dp_group)
        return balanced_data_cache

    @staticmethod
    def rank_table_mapping(rank_table: list, dp_rank: int) -> Tuple[list, list]:
        # Validate rank_table before stacking
        for i, rt in enumerate(rank_table):
            assert len(rt) > 0, f"rank_table[{i}] is empty (expected non-empty tensor list)"

        rank_table = [torch.stack(rt) for rt in rank_table]
        rank_table_for_current_rank = [rt[rt[:, 0] == dp_rank][:, 1] for rt in rank_table]

        return rank_table_for_current_rank, rank_table

    @staticmethod
    def data_reorganization(data: Union[torch.Tensor, list], data_list: list) -> List[Union[torch.tensor, list]]:
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
    def _set_sorting_algo(sorting_algo_name: str) -> Any:
        if sorting_algo_name in SORTING_ALGO_FUNC:
            return SORTING_ALGO_FUNC[sorting_algo_name]
        else:
            raise ValueError(
                f"encoder data balance sorting algorithm name '{sorting_algo_name}' "
                f"does not be implemented, allowed algotighms: {list(SORTING_ALGO_FUNC.keys())}"
            )
