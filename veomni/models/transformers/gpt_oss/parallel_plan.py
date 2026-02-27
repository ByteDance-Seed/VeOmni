from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_paralle_plan():
    ep_plan = {
        "model.layers.*.mlp.experts.gate_up_proj_blocks": Shard(0),
        "model.layers.*.mlp.experts.down_proj_blocks": Shard(0),
        "model.layers.*.mlp.experts.gate_up_proj_bias": Shard(0),
        "model.layers.*.mlp.experts.down_proj_bias": Shard(0),
    }
    parallel_plan = ParallelPlan(
        ep_plan=ep_plan,
    )
    return parallel_plan
