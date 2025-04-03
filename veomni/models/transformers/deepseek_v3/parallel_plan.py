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
    ep_plan = {
        "model.layers.*.mlp.experts.fc1_1": Shard(0),
        "model.layers.*.mlp.experts.fc1_2": Shard(0),
        "model.layers.*.mlp.experts.fc2": Shard(0),
    }
    parallel_plan = ParallelPlan(
        ep_plan=ep_plan,
    )
    return parallel_plan
