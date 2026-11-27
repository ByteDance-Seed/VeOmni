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


"""Qwen4-Exp query-sparse block indexer.

``qsa_indexer`` takes already-projected ReLU block scores and a visibility
mask, and returns the compact top-k block ids. Projections, RoPE, packed
metadata, and the compressed-row gather stay in the model.
"""

from ...registry import register_op
from .standard import eager as qsa_indexer_eager


register_op(
    "qsa_indexer",
    "standard",
    "eager",
    description="PyTorch QSA block top-k with a finite visibility mask",
    wrapper=qsa_indexer_eager.wrapper,
)
