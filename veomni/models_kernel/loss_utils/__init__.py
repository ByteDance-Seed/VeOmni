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

"""Model-facing loss policy around local ``VeomniKernel`` handles."""

from .chunk_logprobs import chunk_logprobs_function
from .chunk_topk_distill import chunk_topk_distill_function
from .cross_entropy_loss import ForCausalLMLoss, ForSequenceClassificationLoss
from .load_balancing_loss import load_balancing_loss


__all__ = [
    "ForCausalLMLoss",
    "ForSequenceClassificationLoss",
    "chunk_logprobs_function",
    "chunk_topk_distill_function",
    "load_balancing_loss",
]
