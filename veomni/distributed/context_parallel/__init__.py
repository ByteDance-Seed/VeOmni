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

"""Context parallelism for DeepSeek-V4 sparse attention."""

from .dsa_cp import all_gather_compressed_rows, all_gather_kv, exchange_compressor_halos
from .sharding import local_window_range, rebase_window_indices, window_owner_counts


__all__ = [
    "all_gather_compressed_rows",
    "all_gather_kv",
    "exchange_compressor_halos",
    "local_window_range",
    "rebase_window_indices",
    "window_owner_counts",
]
