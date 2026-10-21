# Accelerator CI scheduling

GPU and NPU unit/e2e suites use the same case planner, batching logic and device
pool. `suites/gpu.json` and `suites/npu.json` define test selection and conditional
or per-selection environment settings. Resource requirements come from the
launch configuration of each collected case.

```sh
uv run --frozen python -m scripts.ci.accelerator --backend cuda --suite unit --include-conditional
uv run --frozen python -m scripts.ci.accelerator --backend cuda --suite e2e
uv run --frozen python -m scripts.ci.accelerator --backend npu --suite unit
uv run --frozen python -m scripts.ci.accelerator --backend npu --suite e2e
```

GPU/NPU unit and e2e workflows all use parallel dispatch.
GPU/NPU device discovery, owned host IDs, child visibility and logical-rank
mapping are handled by accelerator adapters. NPU assignments stay inside the
runner's existing eight-card slice. Use `--serial` to serialize a local
diagnostic run.

## Adding tests to CI

Update `suites/gpu.json` and/or `suites/npu.json` when a new test file is outside
existing collection scopes. Each manifest contains both unit and e2e selections;
`suite` chooses the job that collects the entry. New functions or parameter
combinations already included by a selection, and files inside collected
directories, need no extra entry. Account for any node-ID or pytest filters.

Keep machine/dependency setup and triggers in workflow YAML. Selection-specific
environment settings use the manifest's `env` field; card requirements come
from the test's launch plan. See
[Registering Tests in CI](../../../docs/testing.md#registering-tests-in-ci) for
the manifest example, registration rules and collection check command.

## Launch plans

An unmarked case defaults to one card of the selected accelerator backend, or
CPU on a CPU-only runner. A CPU plan reserves no accelerator cards. Mark CPU-only
preprocessing, config checks, eager CPU references and mocked backend tests with
`pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))`. Use a module
`pytestmark` only when every case is CPU-only; mixed files need case-level
markers. Hardware names in test functions do not determine resource allocation.
Multi-card tests expose their existing `nproc`, `world_size` or parallel configuration
through `pytest.mark.launch_plan(provider=...)`. A provider may be a shared
`LaunchSpec`, a `CasePlan`, or a pure function of collected parameters and the
runner's immutable capabilities. Module/class markers can supply a shared plan;
`SequenceParallelTest` obtains its plan from its `world_size` property.

Use the same configuration to launch workers. Missing or undersized plans fail
before a distributed launch. `skipif` describes eligibility and does not declare
a device count. Use `runner_device_count()` for hardware-dependent launch
configuration and eligibility, and the runtime's visible count for indexing
actual devices. See the [test API and examples](../../../docs/design/ci_case_scheduling.md#test-api)
and [test author guide](../../../docs/testing.md#launch-plans-for-new-tests).

## Execution and isolation

Collection resolves plans with the complete owned device slice. Sequential
launches reserve their maximum; concurrent launches reserve their sum. Cases
with matching resource requirements and source-file fixture affinity are
batched automatically. Fixed per-case weights are 60 seconds for multi-card
cases and 2 seconds for single-card or CPU cases. The default batch target is
120 estimated seconds with at most 64 cases. These weights only control batch
size; they are not runtime predictions or dispatch priorities.

Cases already skipped during collection retain their declared plans. An
over-capacity skipped case uses a skipped-only batch with the complete owned
device slice, preserving import-time visibility without invalidating the queue.
Runnable requests are never reduced to fit.

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

The launcher's owned process group and tracked descendants are drained before
its lease is released, even if the launcher has already exited.
Ctrl-C cancels owned collection/execution subprocesses before the thread pool
waits for callbacks. Cancellation reaches nested worker scopes; process cleanup
still drains each owned group and tracked descendants before releasing its lease.
By default, the first failed pytest phase or worker stops admission and cancels
other running batches in this suite/job. The runner waits for bounded process
cleanup before exiting with failure. It does not wait for those tests to finish.
`--keep-going` records failures and attempts all cases/batches without cancelling
peers. Failure to drain owned processes always stops admission and cancels
running peers, even with `--keep-going`, because the affected device lease
cannot be safely reused. `--serial` uses the same failure policy with one batch at a time.
CPU BLAS and compiler threads are bounded.

## Results and local commands

Results are written under `outputs/accelerator-ci/{unit,e2e}`, including on
failure. CI workflows retain progress and failure diagnostics in the Actions
console log; they do not upload the output directory as a GitHub artifact.
Full result files are available in the job workspace while it exists.
Use a fresh `--output-dir` for each local measurement.
Result files include full case inventory, resolved plans, batches, device
assignments, per-case timings/outcomes, logs, JUnit XML, observed distributed
launches, source fingerprints and the suite's elapsed wall time. Sum of batch
or case durations is not elapsed wall time when work overlaps. Measured timings,
including `durations.json`, are reporting artifacts and are not scheduling inputs.

Foreground case statuses use `[unit-09-case-0001][STATUS][devices=0,1,2,3] nodeid`
(or `[e2e-57-case-0001]` and `[devices=cpu]`). The entry ID comes from the
manifest: `09-unit` becomes `unit-09`, `npu-e2e-01` becomes `e2e-01`, and
`ci-unit` becomes `unit-ci`. Each case gets a one-based, four-digit index within
its entry's collected inventory, including parameterized cases. This index is
independent of batching, card-count sorting and parallel completion order.
All batches from one entry share its entry ID; no batch number is shown.

`START` / `DONE` are batch-level statuses: they use the entry ID, file and case
count. `RUN`, case outcomes, and running-case heartbeats use the full case ID
and pytest node ID, including parameters. Outcome statuses include
`(completed/total)` across the suite; cancelled cases do not advance this
counter. `RUNNING` heartbeats include elapsed batch time every 30 seconds.
Collection uses the entry ID; final summaries use `[unit]` / `[e2e]`. Both use
`[devices=-]` because they do not describe an individual device lease. Internal
batch names and artifact filenames are unchanged. Execution records carry the
entry `display_id`; aggregated `cases.json` carries each case's full
`display_id` for locating outcomes by their foreground IDs.

```text
[unit-09][START][devices=cpu] tests/foo/test_bar.py (cases=2)
[unit-09-case-0001][RUN][devices=cpu] tests/foo/test_bar.py::test_a
[unit-09-case-0001][PASSED (12/4508)][devices=cpu] tests/foo/test_bar.py::test_a
[e2e-57-case-0001][FAILED (3/120)][devices=0,1,2,3] tests/e2e/test_parallel.py::test_alignment
    ... traceback ...
```

Failures print the node ID and pytest traceback summary (up to 200 lines),
followed by the complete failed worker's combined stdout/stderr. GitHub Actions
places this output in a collapsible group labelled with the manifest entry ID;
local runs print it inline. This preserves training-rank tracebacks and torchrun
errors that precede pytest's outer `CalledProcessError`. Logs are streamed in
bounded chunks after worker cleanup; successful batches keep compact progress
output. GitHub Actions also receives an error annotation. Cancelled
batches are labelled separately from failed tests. The final summary reports
outcomes, failed/cancelled batches, batches never started and the first failure.

When a worker is cancelled before pytest writes its final results, the runner
saves observed outcomes in a partial results artifact. Cases interrupted while
running are marked `cancelled`; never-started cases are not counted as executed.
JUnit output is retained when pytest had time to write it. Partial runs are
failures and are not complete CI coverage measurements.

`--collect-only` writes the inventory and batch plan without running tests.
`--only` selects comma-separated manifest entries. `--pytest-arg` passes an
argument through collection and execution. The conditional Qwen3.5 GDN test
retains its changed-path gate; use `--include-conditional` for complete local
GPU acceptance.

## Small integration fixtures

`tests/toy_config/` is the shared fixture set for local tests and CI. Tests load
the canonical `*_toy/config.json` paths directly, with no environment-specific
copies or path rewriting. See the [toy config guide](../../../tests/toy_config/README.md)
for fixture selection and constraints. Processor assets stay alongside their configs.

These configs preserve model families, expert counts/top-k, backend selection,
parallel world sizes and data shapes. Qwen3.5 retains three linear-attention
layers plus full attention, and DeepSeek-V3 retains dense plus MoE layers.
Multimodal fixtures retain their token IDs, vocabularies and RoPE dimensions;
text-only fixtures use vocabularies of 4096. Qwen3-MoE retains four MoE
layers for the LoRA per-layer EP slicing assertions. Vision/audio widths are matched to
the text connectors. Existing numerical tolerances and checkpoint/optimizer
round-trip assertions are unchanged. GPU and NPU e2e suites each include one
Wan alignment invocation. Qwen3-MoE EP1/4/8 saveload cases check HF export
alongside DCP/optimizer restoration and independent CLI DCP-to-HF verification.

Fused checkpoint formats also require distinguishable weight layouts: for
Qwen3-VL-MoE, keep expert intermediate width different from hidden width, and
keep twice the intermediate width different from hidden width. A square
projection deliberately raises instead of guessing whether to transpose it.

## Module layout

| Module | Responsibility |
|---|---|
| `__main__.py` | Shared command-line entry point |
| `runner.py` | Collection, dispatch and result aggregation |
| `reporting.py` | Foreground progress, failure diagnostics and partial outcomes |
| `launch_plan.py` | Pure launch descriptions and runner capabilities |
| `batching.py` | Fixed-weight batching and card-count ordering |
| `scheduler.py` | Disjoint device leases, greedy dispatch and draining |
| `pytest_plugin.py` | Case inventory, launch validation and pytest results |
| `runtime/devices.py` | GPU/NPU probing and child visibility |
| `runtime/workers.py` | Workspaces, interruptible thread pools, subprocess timeouts and scoped cleanup |
| `runtime/ports.py` | Run-scoped rendezvous port allocation |
| `suites/gpu.json`, `suites/npu.json` | Unit/e2e selections and environment settings |

CPU scheduler tests live in `tests/ci/` and are selected by both manifests.
Documentation checks are separate; see the [CI tool index](../README.md).
