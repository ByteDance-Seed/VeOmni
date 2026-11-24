# Using MXFP8Linear with FSDP2

Install the `float8` extra (`uv sync --extra gpu --extra float8 --dev`) for the pinned `torchao==0.18.0`. Native MXFP8 execution requires an NVIDIA GPU with compute capability 10.0 or newer (SM100+).

Pass `from_float()` a dense `nn.Linear` with a 2-D FP32 master weight. Both input and output widths must be positive multiples of 32. Register module hooks on the converted projection. Conversion creates a new module and rejects projections with existing hooks.

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

The MXFP8 weight subclass owns the gather hooks for a dense 2-D projection. It exposes the FP32 master for export and reconstructs both compute views after the gather. VeOmni’s [DeepGEMM FP8 AllGather installer](../../veomni/distributed/fsdp2/fp8_all_gather.py#L81) attaches hooks to 3-D expert weights after sharding.

Reshard before exporting a state dict. Export preserves the master weights and their DTensor placements. Exporting gathered compute weights raises an error.

CPU tests cover FSDP2 and checkpoint resume with emulated MXFP8 arithmetic. The native GPU CI job runs when the upstream repository sets `MXFP8_SM100_RUNNER_LABEL`.
