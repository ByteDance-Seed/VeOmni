from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_parallel_plan(use_gate_up_proj: bool = True):
    """Return the expert-parallel plan for Qwen3-Omni-MoE (thinker only).

    Thinker experts use stacked 3-D weight tensors (num_experts, out, in),
    so EP shards along dim-0 (the expert dimension).

    Args:
        use_gate_up_proj: When True (default, v5 path), shard on the fused
            ``gate_up_proj`` parameter. When False (v4 path), shard on the
            separate ``gate_proj`` / ``up_proj`` parameters instead.

    NOTE: Talker training is not supported yet. Only thinker EP is planned here.
    """
    if use_gate_up_proj:
        ep_plan = {
            "thinker.model.layers.*.mlp.experts.gate_up_proj": Shard(0),
            "thinker.model.layers.*.mlp.experts.down_proj": Shard(0),
        }
    else:
        ep_plan = {
            "thinker.model.layers.*.mlp.experts.gate_proj": Shard(0),
            "thinker.model.layers.*.mlp.experts.up_proj": Shard(0),
            "thinker.model.layers.*.mlp.experts.down_proj": Shard(0),
        }
    parallel_plan = ParallelPlan(
        extra_parallel_plan={
            "ep": ep_plan,
        }
    )
    return parallel_plan
