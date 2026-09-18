# Design and Migrations

Read these documents for implementation rationale, kernel contracts, and migration history. For user-facing setup, start with the [User Guide](../usage/index.md).

```{toctree}
:maxdepth: 1
:caption: Implementation design

kernel_selection
fused_moe_kernels
local_parallel_state
patchgen
unified_kernel_registry
verl_topk_distill_integration
deepseek_v4_context_parallel
deepseek_v4_indexer_loss
qwen4_exp_ple_2d_parallelism
../usage/hdfs_fuse_patch
```

```{toctree}
:maxdepth: 1
:caption: Transformers integration

../transformers_v5/index
../transformers_v5/veomni_fused_attention
../transformers_v5/veomni_flash_attention_kernel_adapter
../transformers_v5/transformers_v5_moe_weight_loading
```

```{toctree}
:maxdepth: 1
:caption: Migration history

../migrations/index
```
