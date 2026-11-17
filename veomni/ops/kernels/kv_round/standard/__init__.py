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

"""Model-agnostic storage-rounding implementations.

These know only the tensor contract: group the last dimension into
``block_size``-wide blocks, optionally stop at ``quant_features``, round to
``float8_e4m3fn`` with ``ue8m0`` scales, and hand the dequantized values back
through a straight-through estimator. They do not know what a KV cache, a NoPE
channel or an indexer head is -- a model-specific variant supplies that.
"""
