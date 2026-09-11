from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan


def get_parallel_plan():
    """Shard PLE tables over ``ple`` and routed experts over ``ep``.

    PLE uses row sharding over ``ple`` and a persistent column shard over the
    complementary ``ple_fsdp`` mesh. FSDP2 ignores these weights, so they stay
    ``[V / ple_size, E / ple_fsdp_size]`` throughout forward and backward. The
    disjoint expert tensors use the regular EP layout: expert dimension over
    ``ep`` and hidden dimension over ``ep_fsdp``.
    """
    ple_plan = {
        "model.language_model.layers.*.ple.ple_embedding.ngram_embedding.shard_*.weight": Shard(0),
    }
    ep_plan = {
        "model.language_model.layers.*.mlp.experts.gate_up_proj": Shard(0),
        "model.language_model.layers.*.mlp.experts.down_proj": Shard(0),
    }
    persistent_modules = {
        "ple": {"model.language_model.layers.*.ple.ple_embedding": 1},
    }
    parallel_plan = ParallelPlan(
        extra_parallel_plan={"ple": ple_plan, "ep": ep_plan},
        extra_parallel_persistent_modules=persistent_modules,
    )
    return parallel_plan
