# Accelerator CI scheduling

GPU and NPU unit/e2e suites use the same case planner, batching logic and device
pool. `gpu_tests.json` and `npu_tests.json` define test selection and conditional
or per-selection environment settings. Resource requirements come from the
launch configuration of each collected case.

```sh
uv run --frozen python -m scripts.ci.run_accelerator_tests --backend cuda --suite unit --include-conditional
uv run --frozen python -m scripts.ci.run_accelerator_tests --backend cuda --suite e2e
uv run --frozen python -m scripts.ci.run_accelerator_tests --backend npu --suite unit --serial
```

NPU workflows use serial dispatch until concurrent HCCL/device isolation has
been validated on Ascend hardware. GPU/NPU device discovery, owned host IDs,
child visibility and logical-rank mapping are handled by accelerator adapters.
NPU assignments stay inside the runner's existing eight-card slice.

## Launch plans

An unmarked case defaults to one card of the selected accelerator backend, or
CPU on a CPU-only runner. A CPU plan reserves no accelerator cards. Multi-card
tests expose their existing `nproc`, `world_size` or parallel configuration
through `pytest.mark.launch_plan(provider=...)`. A provider may be a shared
`LaunchSpec`, a `CasePlan`, or a pure function of collected parameters and the
runner's immutable capabilities. Module/class markers can supply a shared plan;
`SequenceParallelTest` obtains its plan from its `world_size` property.

Use the same configuration to launch workers. Missing or undersized plans fail
before a distributed launch. `skipif` describes eligibility and does not declare
a device count. Use `runner_device_count()` for hardware-dependent launch
configuration and eligibility, and the runtime's visible count for indexing
actual devices. See the [test API and examples](../../docs/design/ci_case_scheduling.md#test-api)
and [test author guide](../../docs/testing.md#launch-plans-for-new-tests).

## Execution and isolation

Collection resolves plans with the complete owned device slice. Sequential
launches reserve their maximum; concurrent launches reserve their sum. Cases
with matching resource requirements and source-file fixture affinity are
batched automatically. Fixed per-case weights are 60 seconds for multi-card
cases and 2 seconds for single-card or CPU cases. The default batch target is
120 estimated seconds with at most 64 cases. These weights only control batch
size; they are not runtime predictions or dispatch priorities.

The queue is stably sorted by card count, largest first, with collection order
preserved among equal requests. Scan the queue and dispatch every batch that
fits the currently free devices, assigning disjoint leases. Smaller work can
backfill idle cards. Once the accelerator queue head has waited 120 seconds,
the next dispatch check pauses admission of later accelerator batches until
the head fits. CPU work has its own concurrency limit. Each batch has an
isolated working directory and cache. Temporary directories use short paths
under `/tmp` so DataLoader Unix socket names stay within the platform limit;
they are cleaned up after the batch, or retained with failure artifacts. Rendezvous ports
come from the shared launch helpers and a locked registry that prevents
reuse by pending launches. Visibility is set before importing torch,
and launch validation checks nproc and child device assignments against the
case plan and lease. Every distributed launch starts fresh workers.

Only a launcher's own descendants are cleaned up, before its lease is released.
Failures stop new dispatches by default while active batches finish cleanup.
`--keep-going` records failures and attempts all batches; `--serial` disables
concurrent dispatch for diagnosis. CPU BLAS and compiler threads are bounded.

## Results and local commands

Results are written under `outputs/accelerator-ci/{unit,e2e}` and uploaded by CI,
including on failure. Use a fresh `--output-dir` for each local measurement.
Artifacts include full case inventory, resolved plans, batches, device
assignments, per-case timings/outcomes, logs, JUnit XML, observed distributed
launches, source fingerprints and the suite's elapsed wall time. Sum of batch
or case durations is not elapsed wall time when work overlaps. Measured timings,
including `durations.json`, are reporting artifacts and are not scheduling inputs.

`--collect-only` writes the inventory and batch plan without running tests.
`--only` selects comma-separated manifest entries. `--pytest-arg` passes an
argument through collection and execution. The conditional Qwen3.5 GDN test
retains its changed-path gate; use `--include-conditional` for complete local
GPU acceptance.

## Small integration fixtures

`tests/toy_config/ci/` contains opt-in smaller fixtures for synthetic checkpoint,
LoRA, model parity and e2e tests. `tests/tools/ci_config.py` resolves only existing
alternatives at explicitly opted-in build sites. Shared example configs and
processor/tokenizer integration fixtures remain separate.

These configs preserve model families, expert counts/top-k, backend selection,
parallel world sizes and data shapes. Qwen3.5 retains three linear-attention
layers plus full attention, and DeepSeek-V3 retains dense plus MoE layers.
Multimodal fixtures retain their token IDs, vocabularies and RoPE dimensions;
text-only fixtures use vocabularies of 4096. Qwen3-MoE retains four MoE
layers for the LoRA per-layer EP slicing assertions. Vision/audio widths are matched to
the text connectors. Existing numerical tolerances and checkpoint/optimizer
round-trip assertions are unchanged. GPU e2e selection includes one Wan alignment invocation. Qwen3-MoE
EP1/4/8 saveload cases check HF export alongside DCP/optimizer restoration and
independent CLI DCP-to-HF verification.

Fused checkpoint formats also require distinguishable weight layouts: for
Qwen3-VL-MoE, keep expert intermediate width different from hidden width, and
keep twice the intermediate width different from hidden width. A square
projection deliberately raises instead of guessing whether to transpose it.
