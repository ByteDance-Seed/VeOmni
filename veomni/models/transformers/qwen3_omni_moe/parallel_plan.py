from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_parallel_plan():
    # Thinker experts use stacked 3-D weight tensors (num_experts, out, in),
    # so EP shards along dim-0 (the expert dimension).
    # NOTE: Talker training is not supported yet. Only thinker EP is planned here.
    ep_plan = {
        "thinker.model.layers.*.mlp.experts.gate_proj": Shard(0),
        "thinker.model.layers.*.mlp.experts.up_proj": Shard(0),
        "thinker.model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    parallel_plan = ParallelPlan(
        ep_plan=ep_plan,
    )
    return parallel_plan
