from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def _build_parallel_plan(experts_prefix: str) -> ParallelPlan:
    return ParallelPlan(
        extra_parallel_plan={
            "ep": {
                f"{experts_prefix}.gate_up_proj": Shard(0),
                f"{experts_prefix}.down_proj": Shard(0),
            },
        }
    )


def get_vlm_parallel_plan() -> ParallelPlan:
    return _build_parallel_plan("model.language_model.layers.*.mlp.experts")


def get_text_parallel_plan() -> ParallelPlan:
    return _build_parallel_plan("model.layers.*.mlp.experts")
