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
# See the License for the specific language governing limitations
# under the License.

"""Checkpoint convert protocol, fused-expert FQN helpers, and weight I/O."""

from .convert import (
    CheckpointTensorConverter,
    ConvertedCheckpointTensor,
    FqnToIndexMappingConverter,
    checkpoint_converter_is_dim0_zero_pad,
    export_weights,
    get_checkpoint_tensor_converter,
    get_fqn_to_index_mapping_converter,
    maybe_convert_checkpoint_tensor,
    maybe_convert_fqn_to_index_mapping,
    parse_fqn_to_index_mapping_from_json,
    prepare_fqn_to_index_mapping_for_model,
    resolve_fqn_to_index_mapping_for_save,
    shard_index_from_filename,
)
from .moe_map import (
    PER_EXPERT_SPLIT_TO_FUSED_PATTERN,
    convert_per_expert_fqn_mapping_to_fused,
)
from .weights import (
    init_empty_weights,
    load_model_weights,
    load_model_weights_ep_sharded,
    rank0_load_and_broadcast_weights,
    save_model_assets,
    save_model_weights,
)


__all__ = [
    "CheckpointTensorConverter",
    "ConvertedCheckpointTensor",
    "FqnToIndexMappingConverter",
    "PER_EXPERT_SPLIT_TO_FUSED_PATTERN",
    "checkpoint_converter_is_dim0_zero_pad",
    "convert_per_expert_fqn_mapping_to_fused",
    "export_weights",
    "get_checkpoint_tensor_converter",
    "get_fqn_to_index_mapping_converter",
    "init_empty_weights",
    "load_model_weights",
    "load_model_weights_ep_sharded",
    "maybe_convert_checkpoint_tensor",
    "maybe_convert_fqn_to_index_mapping",
    "parse_fqn_to_index_mapping_from_json",
    "prepare_fqn_to_index_mapping_for_model",
    "rank0_load_and_broadcast_weights",
    "resolve_fqn_to_index_mapping_for_save",
    "save_model_assets",
    "save_model_weights",
    "shard_index_from_filename",
]
