# Per-Module Parallelism

Read this before editing per-module `accelerator` blocks, DDP/FSDP2 behavior,
extra parallel dimensions, or distributed/eager inference configs.

## Config Ownership

- Top-level `accelerator` defines the default/global topology.
- A module may override with its own `accelerator:` block in a modules YAML.
- Same topology should reuse the global `ParallelState`.
- Different topology gets a module-local `ParallelState`.

## Common Patterns

```yaml
text_encoder:
  accelerator:
    extra_parallel_sizes: [4]
    extra_parallel_names: ["emb"]
    extra_parallel_placement_innermost: [false]

vision_encoder:
  accelerator:
    fsdp_config:
      fsdp_mode: ddp

backbone:
  accelerator:
    fsdp_config:
      fsdp_mode: fsdp2
```

## Runtime Rules

- Module code that calls `get_parallel_state()` must run under the module's
  parallel-state scope.
- A module with sharded weights should expose `get_parallel_plan()`.
- DDP wraps modules and does not proxy all attributes. Hook dispatch may need to
  unwrap for `pre_forward` / `post_forward` while calling the wrapper for actual
  forward.
- Mixed DDP/FSDP2/ExtraParallel modules require the Omni-specific grad clipping
  path, not a naive `clip_grad_norm_` over all parameters.
- Per-module extra-parallel lists may need deduplication before mesh creation;
  check the current trainer helper before changing this area.

## Inference

- Eager inference is the default for modules without a distributed accelerator.
- A module needs distributed launch when its inference accelerator uses non-eager
  FSDP/DDP.
- Under `torchrun`, eager full-replica modules should pin to the rank's device;
  avoid `device_map="auto"` fanning every rank across all GPUs.
- Keep separate module config files for eager and distributed inference when
  both modes are supported.
