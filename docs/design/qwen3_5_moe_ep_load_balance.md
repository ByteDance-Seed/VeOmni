# Qwen3.5-MoE EP load balancing

Temporary expert replicas redistribute routed tokens across EP ranks. The
feature is opt-in; OFF keeps the existing execution path. No persistent
parameters, optimizer entries or checkpoint keys are added.

## Usage and scope

```yaml
train:
  moe_ep_load_balance:
    enabled: true
    max_replicas_per_rank: 1
    optimize_memory: true
```

Use full-SFT Qwen3.5-MoE with FSDP2, EP>1, ep_outside=false and
resolved dp_replicate_size=1. Select fused_triton on CUDA or fused_npu on
Ascend. CPU/activation offload, LoRA and checkpoint resume are rejected.
Gradient checkpointing and checkpoint saving remain allowed. Replica slots
must not exceed the original expert rows per EP rank. See the
[configuration reference](../usage/arguments.md#moeepbalanceconfig).

## Execution and precision

- The planner gathers actual token histograms, chooses deterministic temporary
  replicas and assigns concrete occurrences without changing router weights.
- Dispatch restores logical order before combine and gradient reduction.
  Temporary weights return contributions to their original owners.
- CUDA keeps ordered FP32 prefixes and the original grouped-GEMM reduction
  configuration. Aligned chunk permutation and weighted combine retain
  fallbacks for unsupported shapes and layouts.
- Native NPU restores owner X/dY in OFF order before the original grouped
  weight-gradient kernel. Independently rounded partial wgrads are not summed.
- Memory mode reads local/replica weights separately. Per-layer buffer leases
  retain graph ownership, track all consuming streams and exclude failed work
  from reuse. Legacy retains a concatenated weight copy.
- Physical before/after load, active replicas and moved tokens are recorded
  once per EP group and reduced over the owning FSDP group. Monitoring is
  disabled by default; set moe_load_balance_monitor_interval to collect it.

## Validation results and limits

| Cohort | Scope | Memory throughput vs matched OFF | Precision / balancing |
| --- | --- | ---: | --- |
| GPU GP04 | 4 H20-3e, L8/EP4, 5 groups × 3 modes × 100 steps | +0.253% mean | Loss/grad-norm exact; every measured ON step active |
| GPU GP06 | Same topology and repeated protocol | +6.472% pooled, +0.116% paired median | Exact curves; pooled gain influenced by one slow OFF group |
| A2 C04 | 8 Ascend 910B3, L24/EP8, 20 steps/mode | −21.15% | Exact curves; four strict native partial-migration cases |
| A2 R02 | Same source, scoped input-Embedding FSDP grouping, 20 steps/mode | −12.65% | Eight matched layout windows exact; ON active every step |

A2 R02 Memory is 4.128807 s/step and matched OFF mean is 3.606386 s/step.
The shared layout also speeds OFF up; it is not an EP-enablement speedup.
NPU performance nonregression remains unmet. GPU GP06 does not establish a
repeatable 5% gain. Five-percent gain is aspirational, not the acceptance gate.

GPU timing excludes 20 warmup steps; NPU qualification excludes five. Timing
uses the slowest-rank synchronized whole window and one global token count.
Thresholds and hardware/software/source cohorts are not pooled across runs.
Native precision uses abs(delta)<=1e-4+1e-3*abs(reference). Short NPU windows
are qualification evidence, not the five-group formal performance protocol.
Memory C04 allocated peak is 50.283 GiB, near OFF 50.624–50.626 GiB and below
Legacy 54.772 GiB; memory savings do not establish a throughput benefit.

[Immutable published measurements and raw evidence](https://github.com/ZhentaoFan/VeOmni/tree/fc459f6c92bcbcfe4e2178a81f7ce1f1755bf158/docs/performance)
retain their own source and runtime identities. EP planning, transport, native
math and GPU/NPU kernel files in this scope reduction are unchanged. Removal
of fixture-only model guards and optional display code is packaging cleanup;
no new full-tree hardware run is claimed for this cleanup.

## Regression coverage

Retained tests cover deterministic/conservative planning, actual multi-rank
routing and owner gradients, BF16 partial migrations, empty segments, frozen
weights, retained graphs, stream/event lifetimes, unsupported configurations,
physical scalar aggregation and CUDA two-storage kernels. The hardware E2E
keeps the native hybrid fixture and fails closed when required by accelerator
CI. Its modified fixture still needs a fresh hardware CI run; CPU skips are
not accelerator validation. CP, HSDP, multi-node, offload, resume and LoRA are
not claimed as supported combinations.
