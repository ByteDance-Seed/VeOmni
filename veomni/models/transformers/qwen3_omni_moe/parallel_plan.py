from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_parallel_plan():
    # NOTE: Expert Parallelism (EP) with FSDP2 is NOT ready for this model yet.
    # This parallel plan is only added to prevent errors during model initialization.
    # TODO: Implement proper EP support for Qwen3OmniMoe.
    ep_plan = {
        "thinker.model.layers.*.mlp.experts.*.gate_proj.weight": Shard(0),
        "thinker.model.layers.*.mlp.experts.*.up_proj.weight": Shard(0),
        "thinker.model.layers.*.mlp.experts.*.down_proj.weight": Shard(0),
    }
    parallel_plan = ParallelPlan(
        ep_plan=ep_plan,
    )
    return parallel_plan
