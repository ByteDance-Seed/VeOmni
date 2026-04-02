from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_parallel_plan(use_gate_up_proj: bool = True):
    """Return the expert-parallel plan for Qwen3-MoE.

    Args:
        use_gate_up_proj: When True (default, v5 path), shard on the fused
            ``gate_up_proj`` parameter.  When False (non-v5 path), shard on
            the separate ``gate_proj`` / ``up_proj`` parameters instead.
    """
    if use_gate_up_proj:
        ep_plan = {
            "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
            "model.layers.*.mlp.experts.down_proj": Shard(0),
        }
    else:
        ep_plan = {
            "model.layers.*.mlp.experts.gate_proj": Shard(0),
            "model.layers.*.mlp.experts.up_proj": Shard(0),
            "model.layers.*.mlp.experts.down_proj": Shard(0),
        }
    parallel_plan = ParallelPlan(
        extra_parallel_plan={
            "ep": ep_plan,
        }
    )
    return parallel_plan
