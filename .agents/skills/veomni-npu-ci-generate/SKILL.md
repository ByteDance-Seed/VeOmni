---
name: veomni-npu-ci-generate
description: "Generate or extend Ascend NPU tests and CI for a specified VeOmni module, file, or feature. Guides source inspection, UT/ST design, real-device execution, failure diagnosis and CI wiring. Trigger: 'generate NPU CI', 'add Ascend tests', '补齐 NPU 测试'. Use veomni-gpu-ci-migrate for an existing GPU test scope and veomni-new-op for a missing production kernel."
---

## Before You Start

1. Read `AGENTS.md`, `.agents/knowledge/constraints.md` and
   `.agents/knowledge/testing.md`. Use the actual workflow as the source of truth
   if a documented test path or selection rule has changed.
2. Read `.agents/knowledge/architecture.md` when locating module ownership or
   backend dispatch; read `.agents/knowledge/uv.md` before changing dependencies.
3. Resolve the inputs below from the request and repository. Ask only for missing
   information that prevents progress; do not ask again for an authorized action.
4. Track the phases in a short plan. Each phase ends with a concrete output.
   A blocked case stays in the report while independent cases continue.

| Input | How to resolve it |
|---|---|
| Repository and revision | User path, then Git root and HEAD; preserve the initial dirty diff |
| Target | Module, source file or feature; resolve feature names to entry points |
| Runtime | Requested Python executable/environment and CANN setup script |
| Devices | User allocation plus current visibility, processes and free memory |
| Test scope | Target behavior plus directly affected callers and existing tests |
| Output | Requested evidence directory, outside tracked source when practical |

Keep environment paths, device IDs and example module names in task inputs.
Do not turn a particular practice case into a requirement for every module.
This skill supplies the procedure; it does not require bundled scripts.

## Phase 0: Establish Environment and Baseline

### 0.1 Locate the implementation

From the repository root, use read-only discovery:

```bash
git rev-parse HEAD
git status --short
rg --files veomni tests .github/workflows
rg -n '<target_symbol>' veomni tests
```

Replace placeholders before running commands. Read the whole relevant function,
its callers, fixtures and existing assertions, rather than only search hits.

Record the source path/symbol, candidate test files and their CI owner. If the
target has no NPU implementation, identify whether a portable eager path exists.
Route a missing production kernel to `/veomni-new-op`; do not invent behavior
inside the test or silently replace the target with a different implementation.

### 0.2 Admit the workload

1. Source the selected CANN script and invoke the selected Python explicitly.
   Confirm `sys.executable`, package versions and imported VeOmni location.
   An editable installation must not import a different checkout.
2. Inspect `npu-smi info`, visible logical devices and free memory. Record a
   timestamp. Availability can change; check again before a multi-device launch.
3. Determine the worker count from the launcher. A test that spawns four ranks
   needs four usable devices even if another test only needs two.
4. Check the required imports, backend availability and model/data assets.
   Record asset identity/revision. Separate missing assets from operator failures.
5. If only CPU is available, proceed with source analysis and CPU contracts,
   but mark real-NPU coverage blocked. Never reset cards or kill unrelated jobs.

Do not install incompatible optional packages to eliminate a skip. Investigate
shared-library search paths separately from Python dependency resolution.

### 0.3 Save the baseline

Run the smallest existing suite that covers the target and affected callers.
Use explicit files/selectors and save command, environment, log, exit code and
JUnit. A typical Linux invocation after setting task-specific variables is:

```bash
mkdir -p "$EVIDENCE_DIR/baseline"
"$PYTHON" -m pytest "$TEST_PATH" -ra   --junitxml="$EVIDENCE_DIR/baseline/results.xml"   >"$EVIDENCE_DIR/baseline/pytest.log" 2>&1
test_rc=$?
printf '%s\n' "$test_rc" >"$EVIDENCE_DIR/baseline/exit-code.txt"
```

The variables above are task inputs, not repository defaults. For distributed
tests use the launch pattern discovered in the repository; do not wrap a test
that already spawns workers in another multi-process launcher.

**Output:** source map, environment/device record and baseline results, including
pre-existing failures/skips. Do not describe an unavailable baseline as passing.

## Phase 1: Turn Source Behavior into a Test Matrix

### 1.1 Trace the execution path

For each public entry point, identify:

1. Input shapes, dtypes, layouts, devices and optional arguments.
2. Outputs and invariants, including mutation, caches and state restoration.
3. Backend selection: device helpers, registry/OpSlot binding, lazy imports.
4. Gradients required by callers, including frozen parameters.
5. Invalid inputs and expected errors or documented fallback behavior.
6. Existing assertions that already cover any of these behaviors.

Do not assume that moving a tensor to NPU selects the intended kernel. Read the
dispatch path and retain evidence that the test reaches the claimed backend.

### 1.2 Choose assertions and the cheapest valid test tier

| Behavior | Assertions | Tier |
|---|---|---|
| Metadata/configuration | Exact values, dtype, shape, device, invalid options | CPU UT; add NPU only for device-sensitive behavior |
| Numerical kernel | Independent reference output and precision-justified tolerance | Real NPU test |
| Differentiable kernel | Output, input/parameter gradients, supported frozen/non-contiguous cases | Real NPU test |
| Loss normalization | Independent denominator, zero case and gradient scaling | UT; separate mocked collectives from real communication |
| State/checkpoint | State transitions, failure ordering, restored values and next update | UT or real round trip as needed |
| Collective | Rank-specific expected values, synchronization and handle semantics | Mocked UT plus real multi-rank ST where claimed |

Build one row per missing behavior:

| ID | Source behavior | Existing assertion | Gap | Independent oracle | Parameters | UT/ST | Devices | Planned test |
|---|---|---|---|---|---|---|---|---|

Choose meaningful boundaries, not an unrestricted Cartesian product. For example,
a normalization kernel may need supported variants, representative shapes,
precision types and gradients; it does not automatically need a full training run.
Use integer equality for exact metadata. An import-only assertion is appropriate
only when import/registration behavior is the actual contract.

**Output:** a matrix with a reason for every new case. If existing tests fully
cover the requested behavior, explain that finding rather than duplicating them.

## Phase 2: Implement Tests and Select the CI Entry

### 2.1 Choose the file

Apply the ownership rules from testing knowledge in order:

1. Extend an already-selected test when it owns the behavior.
2. Add a file only for a separate behavior/fixture/launcher that needs it.
3. Read both exact-file and parent-directory CI selections.
4. Retain existing CPU/CUDA parameters and guards.

Use `veomni.utils.device` helpers where appropriate, after reading their actual
signatures. Guard optional imports at their use sites. Keep mocks local to a test
or fixture and restore patched modules/state; an unexpected mock kernel call
should fail when the test only claims dispatch or metadata coverage.

### 2.2 Implement the assertion before broadening parameters

For numerical tests:

1. Construct initialized, reproducible inputs and an independent reference.
   Do not compare the target with a second call to the same target.
2. Clone inputs for target and reference so mutation or autograd state is separate.
3. Execute the real NPU path; compare output with justified atol/rtol.
4. Apply the same scalar objective/upstream gradient to both paths.
5. Compare every required gradient and check frozen-parameter behavior where valid.
6. Add an optimizer step only if it establishes an additional training invariant.
7. Expand to the matrix parameters after one representative case works.

For state tests, explicitly arrange the initial state, perform the operation,
and assert both the observable result and the next relevant state/action.
For negative tests, assert the expected exception/condition; do not catch all
exceptions and treat them as success.

### 2.3 Implement distributed ST only when needed

1. Inspect whether the existing test uses torchrun or spawns its own workers.
2. Assign each rank its visible device and initialize the actual HCCL group.
3. Use a unique rendezvous and derive expected results analytically per rank.
4. Exercise the claimed collective API and sync/async path. Wait before consuming
   asynchronous output and assert handle semantics where part of the contract.
5. Set stack-compatible collective timeouts and a longer bounded outer lifetime.
   Include startup/compilation allowance; do not copy a timeout from another stack.
6. Propagate worker failures to the parent and destroy process groups in finally.
7. On timeout, clean up only this run's workers. Save rank logs before retrying.

One pytest case that checks six collective paths is still one collected case.

### 2.4 Wire the test without duplicate execution

Inspect the actual unit/e2e owner, runner, image, dependencies and job conditions.
If an existing directory or command selects the case, no new CI line is needed.
Otherwise add the smallest explicit file/selector entry in its owning workflow.

Preserve repository launch conventions. Include a hardware preflight for real-device
coverage, matching the largest launch in that step, bounded execution and JUnit
artifacts. Do not copy private Python/CANN paths or device IDs into public CI.

**Output:** test diff and a mapping from each new case to the CI command that
selects it. Mark proposed but unexecuted CI entries accordingly.

## Phase 3: Execute, Classify Failures and Repair

Run one representative new case, then its parameter set, then the related suite.
Save each attempt separately before editing again. Inspect full tracebacks,
per-rank logs and skip reasons; a zero process exit alone is insufficient.

| Symptom | First check | Next action |
|---|---|---|
| Import/collection error | Selected Python, import location, lazy optional dependencies | Fix environment or isolate an unrelated optional import |
| Library symbol error | CANN setup and dynamic-library resolution | Correct compatible runtime paths; do not guess package upgrades |
| Numerical/gradient mismatch | Backend, initialized inputs, reference, dtype, mutation | Reduce to one failing case; diagnose without relaxing tolerance blindly |
| Hang/timeout | All rank logs, worker count, rendezvous, unmatched collectives | Fix the specific launch/communication cause |
| OOM | Other workloads, input size, per-rank memory | Reserve suitable resources; reducing a performance workload changes its claim |
| Skip/no tests collected | Guards, selectors, markers, dependencies | Record missing coverage; do not count as execution |
| Missing tokenizer/data | Asset identity and cache configuration | Supply the requested asset or record a blocker |

Use one falsifiable hypothesis per fix. Rerun the failing case after the change,
then affected passing cases. After three unsuccessful fixes for the same symptom,
return to source/environment analysis and report the unresolved cause; do not
keep making unrelated changes. Continue independent matrix rows when possible.

**Output:** authoritative results per case and explicit unresolved rows. Label
CPU contracts, mocked behavior, real NPU numerics and distributed ST separately.

## Phase 4: Regression and Delivery

1. Run baseline-related coverage against the final tree and compare by file/node ID.
   Account explicitly for renamed or newly parameterized tests.
2. Run repository quality checks and relevant checks for any helper you changed.
   If the user explicitly limits validation, honor it and list omitted checks.
3. Confirm CI selection including -k/-m, node IDs, ignore options and conditions.
4. For requested reuse, repeat the analysis on materially different interfaces;
   preserve each interface's matrix and evidence.
5. Review the diff for unrelated changes, leaked paths/credentials and weakened
   assertions. Do not add runtime evidence or model caches to source commits.

Deliver a compact report with:

| Field | Required content |
|---|---|
| Scope | Base SHA, changed files, target interfaces |
| Coverage | Matrix IDs mapped to tests and CI selectors |
| Execution | Exact command, Python/CANN/backend/devices, relevant overrides |
| Results | Exit code and pass/fail/error/skip counts, duration/timeout |
| Evidence | Logs/JUnit per attempt; one authoritative final attempt per unit |
| Limitations | Blocker, next action, unexecuted GPU/public-image jobs |

Do not sum retries, invent counts for module-level skips, or describe CPU tests
on an NPU host as accelerator validation. Local success does not establish
maintainer approval, PR publication or merge.

**Completion:** every matrix row has an outcome, every claimed covered case has
execution and CI-selection evidence, and remaining limitations are explicit.