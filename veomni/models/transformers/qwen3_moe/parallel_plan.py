from torch.distributed._tensor import Shard

from ....distributed.parallel_plan import ParallelPlan
from ....utils.import_utils import is_transformers_version_greater_or_equal_to


def get_parallel_plan():
    if is_transformers_version_greater_or_equal_to("5.0.0"):
        # Since transformers v5, HuggingFace changed the per-expert keys to merged expert tensors and concatenated
        # gate and up-projection. In transformers 4.x, veomni asks for merged expert tensors and do not concatenate
        # gate and up-projection.
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
        ep_plan=ep_plan,
    )
    return parallel_plan
