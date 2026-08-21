from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_parallel_plan():
    """Build the EP sharding plan for trunk and MTP fused expert weights."""
    ep_plan = {
        "model.language_model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.language_model.layers.*.mlp.experts.down_proj": Shard(0),
        "mtp.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "mtp.layers.*.mlp.experts.down_proj": Shard(0),
    }
    parallel_plan = ParallelPlan(
        extra_parallel_plan={
            "ep": ep_plan,
        }
    )
    parallel_plan.extra_parallel_fsdp_no_shard_module["ep"] = {
        "model.language_model.layers.*.mlp.experts",
        "mtp.layers.*.mlp.experts",
    }
    return parallel_plan
