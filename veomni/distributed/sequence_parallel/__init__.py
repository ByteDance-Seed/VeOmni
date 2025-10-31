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


from .async_ulysses import (
    async_ulysses_output_projection,
    async_ulysses_qkv_projection,
    divide_qkv_linear_bias,
    divide_qkv_linear_weight,
)
from .comm import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
    get_unified_sequence_parallel_group,
    get_unified_sequence_parallel_rank,
    get_unified_sequence_parallel_world_size,
    init_sequence_parallel,
    set_context_parallel_group,
    set_data_parallel_group,
    set_ulysses_sequence_parallel_group,
    set_unified_sequence_parallel_group,
)
from .data import (
    gather_outputs,
    sequence_parallel_preprocess,
    slice_input_tensor,
    slice_input_tensor_scale_grad,
    slice_position_embedding,
    sp_pad_and_slice,
)
from .loss import reduce_sequence_parallel_loss
from .ulysses import (
    all_to_all_images,
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
)
from .utils import pad_tensor, unpad_tensor, vlm_images_a2a_meta


__all__ = [
    "init_sequence_parallel",
    "set_data_parallel_group",
    "get_data_parallel_group",
    "get_data_parallel_rank",
    "set_ulysses_sequence_parallel_group",
    "get_ulysses_sequence_parallel_world_size",
    "get_ulysses_sequence_parallel_rank",
    "get_ulysses_sequence_parallel_group",
    "set_context_parallel_group",
    "get_context_parallel_group",
    "get_context_parallel_rank",
    "get_context_parallel_world_size",
    "set_unified_sequence_parallel_group",
    "get_unified_sequence_parallel_group",
    "get_unified_sequence_parallel_rank",
    "get_unified_sequence_parallel_world_size",
    "slice_input_tensor",
    "slice_input_tensor_scale_grad",
    "slice_position_embedding",
    "sp_pad_and_slice",
    "sequence_parallel_preprocess",
    "gather_heads_scatter_seq",
    "gather_seq_scatter_heads",
    "all_to_all_images",
    "gather_outputs",
    "vlm_images_a2a_meta",
    "pad_tensor",
    "unpad_tensor",
    "reduce_sequence_parallel_loss",
    "async_ulysses_qkv_projection",
    "async_ulysses_output_projection",
    "divide_qkv_linear_weight",
    "divide_qkv_linear_bias",
]
