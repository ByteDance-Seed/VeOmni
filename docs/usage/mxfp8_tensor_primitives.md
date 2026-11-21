# Using MXFP8Linear with FSDP2

Native MXFP8 requires VeOmni’s `float8` extra and a supported NVIDIA GPU (SM100 or newer). Start with FP32 weights. `from_float()` preserves their dtype.

Given a dense `model.proj` and an initialized FSDP2 mesh, convert before sharding and create the optimizer last. With MXFP8 all-gather, conversion replaces the weight parameter.

```python
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from veomni.distributed.mxfp8.tensors import MXFP8Linear

model.proj = MXFP8Linear.from_float(model.proj, all_gather=True)
policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
fully_shard(model, mesh=mesh, mp_policy=policy)
optimizer = torch.optim.AdamW(model.parameters())
```

- `all_gather=True` quantizes each shard in both orientations before a multi-rank gather. Use BF16 compute parameters and FP32 gradient reduction. The weight must be nonempty, contiguous and evenly sharded along dimension 0, with both the local row count and input width divisible by 32.
- `all_gather=False` keeps the original parameter and uses the standard FSDP2 gather. The layer quantizes the gathered weight for computation.

Reshard before exporting a state dict. Export preserves the master weights and their DTensor placements. Exporting gathered compute weights raises an error.

CPU tests cover FSDP2 and checkpoint resume with emulated MXFP8 arithmetic. The native GPU CI job runs when the upstream repository sets `MXFP8_SM100_RUNNER_LABEL`.
