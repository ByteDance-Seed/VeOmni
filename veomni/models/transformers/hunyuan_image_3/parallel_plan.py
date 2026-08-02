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

from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_parallel_plan() -> ParallelPlan:
    return ParallelPlan(
        extra_parallel_plan={
            "ep": {
                "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
                "model.layers.*.mlp.experts.down_proj": Shard(0),
            }
        },
        # The frozen VAE encoder must stay replicated FP32 on every rank -- BF16
        # perturbs the online latents by 4-6%. This declaration is authoritative;
        # the model's ``get_fsdp_ignored_params`` hook is still called, but only
        # for its ``.float()`` side effect (the meta-path cast must precede the
        # root shard). When a recipe builds no VAE the pattern matches nothing.
        fsdp_ignored_param_fqn_patterns=["vae.*"],
    )


__all__ = ["get_parallel_plan"]
