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


from .gdn_kcp import (
    all_gather_affine_hm,
    assert_kcp_comm_bytes_independent_of_seq,
    assert_kcp_cp_group_identity,
    local_affine_summary,
    prefix_merge_initial_state,
    resolve_kcp_initial_state,
)
from .gdn_lossless import (
    GdnLosslessRuntimePlan,
    align_gdn_varlen_chunks,
    aligned_gdn_cu_seqlens,
    attach_state_dependency,
    compile_gdn_lossless_runtime_plan,
    exchange_conv_halo,
    make_state_participation,
    make_state_template,
    owned_to_physical,
    physical_to_owned,
    receive_initial_state,
    send_final_state,
    trim_conv_halo,
    unpad_gdn_varlen_output,
)
from .gdn_ownership import (
    GDN_NATIVE_CHUNK_SIZE,
    GdnLosslessPlan,
    GdnRankPlan,
    build_gdn_lossless_plan,
    validate_gdn_lossless_plan,
)
from .gdn_runtime import (
    GdnCpEventCount,
    GdnCpOperation,
    GdnCpPhase,
    GdnCpRuntimeIdentity,
    GdnCpRuntimeObserver,
    GdnCpRuntimeSnapshot,
    make_gdn_cp_runtime_observer,
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
    "GDN_NATIVE_CHUNK_SIZE",
    "GdnCpEventCount",
    "GdnCpOperation",
    "GdnCpPhase",
    "GdnCpRuntimeIdentity",
    "GdnCpRuntimeObserver",
    "GdnCpRuntimeSnapshot",
    "GdnLosslessPlan",
    "GdnLosslessRuntimePlan",
    "GdnRankPlan",
    "PackedContextParallelPartition",
    "AttentionWithCp",
    "apply_packed_context_parallel_partition",
    "all_gather_affine_hm",
    "align_gdn_varlen_chunks",
    "aligned_gdn_cu_seqlens",
    "attach_state_dependency",
    "assert_kcp_comm_bytes_independent_of_seq",
    "assert_kcp_cp_group_identity",
    "build_gdn_lossless_plan",
    "build_packed_context_parallel_partition",
    "compile_gdn_lossless_runtime_plan",
    "dense_causal_attention",
    "exchange_conv_halo",
    "make_state_participation",
    "make_state_template",
    "make_gdn_cp_runtime_observer",
    "owned_to_physical",
    "pad_packed_samples",
    "padded_sample_lengths",
    "physical_to_owned",
    "prefix_merge_initial_state",
    "receive_initial_state",
    "reorder_sample_major_to_ulysses_rank_major",
    "reorder_ulysses_rank_major_to_sample_major",
    "ringattn_context_parallel",
    "resolve_kcp_initial_state",
    "send_final_state",
    "simulate_packed_ring_causal_attention",
    "simulate_ring_causal_attention",
    "trim_conv_halo",
    "unpad_gdn_varlen_output",
    "ulysses_local_cu_from_global",
    "ulysses_local_head_count",
    "validate_gdn_lossless_plan",
    "local_affine_summary",
]
