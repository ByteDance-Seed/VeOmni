from torch.distributed._tensor import Shard

from ......distributed.parallel_plan import ParallelPlan


def get_parallel_plan():
    emb_plan = {
        "embed_tokens.weight": Shard(0),
    }
    parallel_plan = ParallelPlan(
        extra_parallel_plan={
            "emb": emb_plan,
        }
    )
    return parallel_plan
