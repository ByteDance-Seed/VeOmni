from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_vlm_parallel_plan() -> ParallelPlan:
    return ParallelPlan(
        extra_parallel_plan={
            "ep": {
                "model.language_model.layers.*.mlp.experts.gate_up_proj": Shard(0),
                "model.language_model.layers.*.mlp.experts.down_proj": Shard(0),
            },
        }
    )
