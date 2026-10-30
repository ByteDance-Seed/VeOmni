# Selective expert CPU offload

FSDP2 can keep selected expert modules' compute-weight shards and accumulated
gradients on CPU while the rest of the model stays on the accelerator. This
uses native PyTorch `CPUOffloadPolicy`; it is separate from optimizer-state
offload and does not change expert ownership, gradient division or precision.

```yaml
model:
  accelerator:
    fsdp_config:
      fsdp_mode: fsdp2
      offload: false
      offload_pin_memory: false
      cpu_offload_extra_parallel_names: [ep]
      cpu_offload_exclude_module_patterns:
        - "*.layers.5.mlp.experts"
        - "*.layers.11.mlp.experts"
```

The exclusion list keeps complete matching expert modules resident on the
accelerator. Patterns match complete module FQNs using the parallel plan's `*`
wildcard rules; every exclusion must match a selected expert module. An empty
list offloads all selected experts. Keeping 8 or 16 layers resident changes this
list, not the implementation or checkpoint layout.

## Placement and update lifecycle

[`VeOmniModelRuntime._build_parallelized_model()`](../../veomni/models/model_runtime.py#L401)
forwards these settings to
[`parallelize_model_fsdp2()`](../../veomni/distributed/torch_parallelize.py#L432).
The latter applies CPU offload only to non-excluded
EP FSDP groups, loads the model, and moves those groups to CPU before optimizer
construction. FSDP stages their weights for forward/backward and accumulates
their reduced gradients on CPU. Non-expert groups and persistent ExtraParallel
parameters, such as PLE tables, retain their existing placement.

Global gradient clipping computes the norm with the existing ownership groups.
CPU gradient shards are scaled locally with the same global coefficient as
resident accelerator gradients. Full model, optimizer-state and checkpoint
precision policies are unchanged; FP32 masters and moment offload must be
configured through the chosen optimizer separately.

The model's `_cpu_offload_resident_module_fqns` lists the matched resident
modules. Offloaded parameters carry `_veomni_optimizer_compute_device` so a
CPU-capable optimizer can distinguish storage placement from the compute device.
Standard `DistributedMuon` does not support this placement and rejects it;
specialized CPU-capable optimizer recipes are independent of this feature.

## Supported configuration

- FSDP2, EP enabled, and expert FSDP mesh size 1.
- Global FSDP offload disabled; explicit `ep` selection only.
- Eager execution; `torch.compile` is rejected for this placement.
- `offload_pin_memory` applies to the selected CPU groups as well.

CPU RAM holds both the compute-weight shadow and accumulated gradients in
addition to any optimizer state. Initial loading may still materialize the
compute shards on the accelerator before they move to CPU. Residency does not
change packing, Ulysses, CP, routing, parameter names or checkpoint ownership.
The caller remains responsible for choosing a host-memory budget and validating
the complete training/checkpoint configuration.
