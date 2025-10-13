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

from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_paralle_plan():
    # TODO: maybe should support first_k_dense_replace and moe_layer_freq
    ep_plan = {
        "language_model.model.layers.*.mlp.experts.gate_proj": Shard(0),
        "language_model.model.layers.*.mlp.experts.up_proj": Shard(0),
        "language_model.model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    parallel_plan = ParallelPlan(
        ep_plan=ep_plan,
    )
    return parallel_plan
