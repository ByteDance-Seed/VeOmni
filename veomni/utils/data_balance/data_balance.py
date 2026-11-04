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

from typing import Any, List, Optional, Tuple, Union

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils import helper
from veomni.utils.data_balance.balance_sorting_algo import SORTING_ALGO_FUNC
from veomni.utils.data_balance.module_balance import BalancePlan, ModuleDataBalancer


logger = helper.create_logger(__name__)


class Qwen3VLEncoderDataBalance:
    """Compatibility entry points backed by the shared whole-item DP balancer.

    New module consumers use ModuleDataBalancer and retain an immutable plan
    per invocation. This shim preserves balance_data/data_bridge and separate
    image/video slots; it no longer owns an independent transport implementation.
    As in the legacy API, a second balance_data call for the same data_type
    replaces that slot. Every DP rank must participate in bridge and backward.
    """

    def __init__(
        self,
        spatial_merge_unit: int,
        sorting_algo_name: str = "post_mbs_balancing_greedy_without_pad",
    ):
        if spatial_merge_unit < 1:
            raise ValueError("spatial_merge_unit must be positive.")
        logger.info_rank0("Initializing Qwen3 vl encoder data balance...")
        self.state_buffer: dict[str, BalancePlan] = {}
        self.merge_down_ratio = spatial_merge_unit
        self.sorting_algo = self._set_sorting_algo(sorting_algo_name)
        self.dp_group = get_parallel_state().dp_group
        self.balancer = ModuleDataBalancer(self.dp_group)
        logger.info_rank0("Successfully initialized Qwen3 vl encoder data balance")

    def balance_data(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, data_type: str = "image"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_thw = grid_thw.long()
        lengths = grid_thw.prod(dim=1).tolist()
        balanced, plan = self.balancer.balance(
            {"pixel_values": pixel_values, "image_grid_thw": grid_thw},
            {"pixel_values": lengths, "image_grid_thw": [1] * len(lengths)},
            [length // self.merge_down_ratio for length in lengths],
            # Preserve the legacy quadratic RAW-patch scheduling cost; merged
            # output lengths are routing splits, not a replacement cost model.
            costs=[length * length for length in lengths],
        )
        self.state_buffer[data_type] = plan
        return balanced["pixel_values"], balanced["image_grid_thw"]

    def data_bridge(
        self,
        hidden_state: torch.Tensor,
        deepstack_feature_lists: Optional[List],
        require_grad: bool = True,
        data_type: str = "image",
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        plan = self.state_buffer[data_type]
        # Respect require_grad without enabling autograd inside an outer
        # no_grad/inference scope. Each output branch reuses the immutable plan.
        enable_grad = require_grad and torch.is_grad_enabled()
        with torch.set_grad_enabled(enable_grad):
            recovered = plan.restore(hidden_state)
            deepstack = [plan.restore(feature) for feature in deepstack_feature_lists or []]
        if not enable_grad:
            # Empty singleton restores may be views retaining requires_grad.
            recovered = recovered.detach()
            deepstack = [feature.detach() for feature in deepstack]
        return recovered, deepstack

    @staticmethod
    def rank_table_mapping(rank_table: list, dp_rank: int) -> Tuple[list, list]:
        # Retain the legacy sorter helper for compatibility, not for transport.
        for i, rt in enumerate(rank_table):
            assert rt.numel() > 0, f"rank_table[{i}] is empty (expected a non-empty tensor)"
        return [rt[rt[:, 0] == dp_rank][:, 1] for rt in rank_table], rank_table

    @staticmethod
    def data_reorganization(data: Union[torch.Tensor, list], data_list: list) -> List[Union[torch.tensor, list]]:
        if isinstance(data, torch.Tensor):
            return [data[indexes] for indexes in data_list]
        return [
            torch.cat([data[index] for index in indexes])
            if indexes.numel()
            else torch.tensor([], dtype=data[0].dtype, device=data[0].device)
            for indexes in data_list
        ]

    @staticmethod
    def _set_sorting_algo(sorting_algo_name: str) -> Any:
        if sorting_algo_name in SORTING_ALGO_FUNC:
            return SORTING_ALGO_FUNC[sorting_algo_name]
        raise ValueError(
            f"encoder data balance sorting algorithm name '{sorting_algo_name}' "
            f"is not implemented, allowed algorithms: {list(SORTING_ALGO_FUNC)}"
        )
