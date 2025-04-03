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


import torch
from vescale.plan import ParallelType, VescalePlan


# TODO: add more model type
SET_TP_SHARD_PLAN_FUNC = {}


def build_vescale_plan(
    model_config: dict,
    tp_size: int = 1,
    pp_size: int = 1,
    use_doptim: bool = False,
    use_fsdp: bool = False,
    use_manual_eager: bool = False,
    use_mixed_precision: bool = True,
    clip_grad: float = 0.0,
):
    """Build parallel plan for P6 model.

    Args:
        model_config (dict): model config dict
        tp_size (int): size of tensor parallelism
        pp_size (int): size of pipeline parallelism
        use_doptim (bool): whether to use DistributedOptimizer (zero)
        use_manual_eager (bool): whether to use manual eager for tensor parallelism
        use_mixed_precision (bool): whether to enable mixed precision, where parameters will be saved
            and updated in additional fp32 copy, and gradients will be accumulated with fp32.
        clip_grad (float): gradient clipping threshould
    """
    model_type = model_config.model_type
    if use_doptim and use_fsdp:
        raise RuntimeError("Cannot simutaneously use FSDP and DistributedOptimizer.")

    plan = VescalePlan()

    # get device mesh
    ngpus = torch.distributed.get_world_size()
    if ngpus % (tp_size * pp_size) != 0:
        raise ValueError("total gpu number must be divisible by tp_size * pp_size ")
    if pp_size > 1:
        raise NotImplementedError("pp size only support 1")
    dp_size = ngpus // (tp_size * pp_size)
    print(f"creating {tp_size} tp, {pp_size} pp, {dp_size} dp...")

    mesh = {}
    # setup dp mesh
    dp_name = ParallelType.FSDP if use_fsdp else ParallelType.DP
    mesh[dp_name] = dp_size
    # setup tp mesh
    if tp_size > 1:
        mesh[ParallelType.TP] = tp_size
    # setup pp mesh
    if pp_size > 1:
        mesh[ParallelType.PP] = pp_size

    plan.set_global_mesh("cuda", mesh)

    # tensor parallel
    if tp_size > 1:
        plan = SET_TP_SHARD_PLAN_FUNC[model_type](plan, tp_size, model_config, use_manual_eager)

    # dist optimizer: this must go before setting up data parallel
    # due to `use_distributed_optimizer field`
    if use_doptim:
        plan.dist_optimizer(grad_to_fp32=use_mixed_precision, overlap_param_gather=False, clip_grad=clip_grad)

    # data parallel fsdp / ddp
    if use_fsdp:
        if use_doptim:
            raise ValueError("fsdp and doptim can not be used together")
        if tp_size > 1:
            raise NotImplementedError("vescale FSDP cannot work with TP for now")

        from vescale.fsdp.api import MixedPrecision, ShardingStrategy

        mp = None
        if use_mixed_precision:
            mp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        plan.dist_fsdp(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
        )
    else:
        plan.dist_data_parallel(
            grad_in_fp32=use_mixed_precision,
            overlap_grad_reduce=False,
        )
    return plan
