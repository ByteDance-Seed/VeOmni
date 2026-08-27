# Copyright 2026 Bytedance Ltd. and/or its affiliates
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


from .gdn_headwise import (
    GdnHeadwiseLayout,
    compile_gdn_headwise_layout,
    prepare_gdn_headwise_inputs,
    restore_gdn_headwise_output,
)
from .packed_sharding import (
    PackedContextParallelPartition,
    apply_packed_context_parallel_partition,
    build_packed_context_parallel_partition,
    pad_packed_samples,
    padded_sample_lengths,
    reorder_sample_major_to_ulysses_rank_major,
    reorder_ulysses_rank_major_to_sample_major,
    ulysses_local_cu_from_global,
    ulysses_local_cu_to_cp_local_cu,
    ulysses_local_head_count,
)
from .ring_attention import (
    AttentionWithCp,
    dense_causal_attention,
    ringattn_context_parallel,
    simulate_packed_ring_causal_attention,
    simulate_ring_causal_attention,
)


__all__ = [
    "GdnHeadwiseLayout",
    "PackedContextParallelPartition",
    "AttentionWithCp",
    "apply_packed_context_parallel_partition",
    "build_packed_context_parallel_partition",
    "compile_gdn_headwise_layout",
    "dense_causal_attention",
    "pad_packed_samples",
    "padded_sample_lengths",
    "prepare_gdn_headwise_inputs",
    "reorder_sample_major_to_ulysses_rank_major",
    "reorder_ulysses_rank_major_to_sample_major",
    "ringattn_context_parallel",
    "restore_gdn_headwise_output",
    "simulate_packed_ring_causal_attention",
    "simulate_ring_causal_attention",
    "ulysses_local_cu_from_global",
    "ulysses_local_cu_to_cp_local_cu",
    "ulysses_local_head_count",
]
