---
name: veomni-gpu-ci-migrate
description: "Migrate a caller-selected VeOmni GPU workflow or test scope to Ascend NPU. Guides workflow tracing, portability classification, semantic-preserving adaptation, real execution, CI wiring and fixed-denominator reporting. Trigger: 'migrate GPU CI to NPU', 'port CUDA tests', '迁移 GPU CI'. For new module coverage without an existing GPU scope use veomni-npu-ci-generate."
---

## Before You Start

1. Read `AGENTS.md`, `.agents/knowledge/constraints.md` and
   `.agents/knowledge/testing.md`. Read architecture knowledge for backend
   dispatch and uv knowledge before dependency changes.
2. Resolve the task inputs below from the request and repository. Ask only when
   a missing value blocks work; preserve previously authorized scope.
3. Track each phase and its output. Keep blocked items visible while progressing
   independent items. This protocol requires no bundled inventory script.
4. Use the actual source workflows at the recorded revision as the source of truth.
   No test directory, model family, package version or launcher is prescribed.

| Input | Required interpretation |
|---|---|
| Source | GPU workflows/jobs or explicit test scope |
| Target | NPU workflow and its runner/image/launch conventions |
| Repository | Working tree, base SHA and pre-existing diff |
| Runtime | Requested Python environment and CANN setup |
| Resources | Permitted devices, memory and runtime budget |
| Evidence | Output directory for inventory, commands, logs and results |

Do not equate "GPU CI" with every file under tests, or "migration" with replacing
cuda strings. The task is to preserve the selected tests' behavioral purpose.

## Phase 0: Freeze Scope and Counting Units

1. Record `git rev-parse HEAD`, `git status --short` and the initial diff.
2. Save original source/target workflow contents or their immutable revision.
3. State which workflows/jobs and optional file scopes are included. For a broad
   request, inventory all requested sources before choosing a practice subset.
4. List original workflows, jobs, commands, unique files and collected cases
   separately. Choose one unit for the headline migration rate.
5. Record which original units are already selected by NPU CI before edits.
6. Keep newly generated tests and added parameters in a separate list.

Use a table with these columns:

| ID | Workflow/job/step | Original command | Working directory | Conditions/matrix | File/node selector | Existing NPU selection |
|---|---|---|---|---|---|---|

Retain repeated command rows for provenance; deduplicate only the unique-file
count. A case count requires collection evidence, not a count of test functions.

**Output:** frozen scope, source versions, original inventory skeleton and declared
counting unit. Do not silently shrink the original scope after encountering failures.

## Phase 1: Discover the Actual Tests

### 1.1 Find candidate launch sites

From the repository root, start with:

```bash
rg --files .github/workflows
rg -n 'pytest|torchrun|working-directory|matrix|uses:|run:' .github/workflows
rg --files tests
```

These searches find candidates; read the full relevant YAML and invoked scripts.
The absence of a direct pytest line does not prove a job has no tests.

### 1.2 Resolve each command in its execution context

For each source job/step, work through this list:

1. Determine the effective workflow/job/step working directory.
2. Resolve repository-relative file paths from that directory.
3. Inspect job/step conditions, matrix includes/excludes and environment variables.
   Preserve distinct variants when dependencies, backends or selectors differ.
4. Follow reusable workflows and input values. For external workflows, pin the
   referenced revision where available; mark inaccessible content unresolved.
5. Follow shell/Python wrappers to their actual test invocation. Account for
   multiple commands, cd/source state, shell conditionals and loops.
6. Recognize direct pytest, Python -m pytest and uv wrappers, but inspect wrapper
   options that change environment, project or working directory.
7. Resolve file/directory arguments, node selectors, -k/-m filters, ignore flags
   and pytest configuration. Read conftest/custom collectors when relevant.
8. Keep dynamic paths and unknown conditions unresolved until traced. Never
   assume an unresolved command selects no tests.

Example: two commands separated by && are two provenance rows, while a directory
and a repeated explicit file may refer to the same unique file. Preserve both
sources but do not count the file twice.

Do not run arbitrary workflow shell just to discover its targets. If safe
collection is necessary after inspecting imports and fixtures, use the intended
environment and preserve its errors/skips. For a direct pytest target:

```bash
"$PYTHON" -m pytest "$TEST_PATH" --collect-only -q
```

Replay the real selection options as separate arguments. A collection command
imports code; it is not a substitute for source inspection or real execution.

### 1.3 Read each selected test

Record the behavior being asserted, required fixture/assets, actual backend,
device guard, optional imports and launcher. Trace the implementation if the
test name alone does not establish portability.

**Output:** resolved inventory plus unresolved rows with source location, reason
and next action. Cross-check it against every selected workflow command. Do not
advance an unresolved row as if it were an eligible or successfully migrated test.

## Phase 2: Classify and Run a Baseline

### 2.1 Assign a source-backed classification

| Class | Decision rule | Action |
|---|---|---|
| Direct reuse | Existing assertions and execution path already work in target environment | Run unchanged; wire only if missing |
| Adaptation required | Behavioral purpose is portable but devices, backend, imports or fixtures differ | Record the exact required change |
| Blocked | Required kernel, dependency, hardware or asset is unavailable | Record concrete blocker and available independent subsets |
| Unresolved | Selection or portability cannot yet be determined | Continue source tracing; keep denominator uncertainty explicit |

For mixed files, classify case groups separately while keeping the original file
in the file-level denominator. Example: CPU metadata cases can run while CUDA
vendor-kernel cases remain blocked; the whole file is partial, not fully migrated.

Distinguish these execution kinds:

- CPU contract, including explicitly mocked dispatch/communication.
- Real NPU numerical/gradient test.
- Real distributed ST.
- Static/collection-only inspection.

### 2.2 Admit and baseline the selected workload

1. Source the requested CANN environment and confirm the selected Python,
   torch/torch_npu versions and imported checkout location.
2. Inspect current device visibility, live workloads and free memory.
   Derive device count from each launch, not the host total.
3. Check dependencies and required tokenizer/model/data snapshots; record identity.
4. Determine whether tests spawn workers internally or expect an external launcher.
   Do not nest launchers accidentally.
5. Run a bounded baseline and save commands, environment, logs, exit codes and
   JUnit before editing. If a GPU baseline cannot run, use only clearly identified
   prior evidence or mark it unavailable; NPU execution cannot establish GPU parity.

Never reset devices or stop unrelated users' jobs. If resources are unavailable,
continue CPU/source work and keep real-device rows blocked.

**Output:** classification matrix, minimal adaptation plan, and baseline outcomes.
For each proposed change, state the original assertion that must remain true.

## Phase 3: Adapt One Behavioral Group at a Time

### 3.1 Apply the first relevant adaptation

| Original obstacle | Permitted adaptation | Evidence required |
|---|---|---|
| Hard-coded CUDA device with portable operations | Existing device helpers or CPU/CUDA/NPU parameters | Same assertions on the intended backend |
| NCCL-specific launch for portable collectives | NPU device binding and HCCL launch | Real multi-rank results, not just initialization |
| Optional CUDA import blocks independent CPU contracts | Lazy/use-site import or scoped dependency isolation | Independent contracts run; unavailable kernels remain explicit |
| CPU-only parameter despite supported NPU operation | Add NPU parameter without removing CPU | Same forward/gradient/state assertions |
| CUDA memory API in portable workload | Existing backend memory abstraction | Same workload, synchronization and reduction assertion |
| Vendor kernel, bitwise guarantee or unsupported compiler | Keep backend restriction or implement separately | No unsupported claim of NPU equivalence |

Read helper signatures before using them. Keep baseline tolerances and error
semantics. If a tolerance change is necessary, establish an independent precision
reason and document the changed contract rather than silently loosening it.

For numerical tests, use initialized inputs, an independent reference and matching
upstream gradients. Preserve frozen parameters and supported non-contiguous inputs.
For metadata/state tests, preserve exact values, failure ordering and restoration.

### 3.2 Keep dependency isolation honest

1. Determine whether the missing package is part of the tested behavior.
2. If yes, record a blocker or use an existing supported implementation.
3. If no, isolate only the import needed by the independent contract.
4. Scope mocks with fixtures/monkeypatch and restore state after the test.
5. Make an unexpected call into a stub kernel fail.
6. Label the result CPU/mock contract coverage, not that package's NPU numerics.

Avoid broad module-level skips when reusable independent cases can be collected.
Do not remove legitimate CUDA guards from architecture-specific tests.

### 3.3 Preserve distributed semantics

Use the actual HCCL backend and real NPU tensors for ST. Bind each rank correctly,
derive rank-specific expected outputs and check wait/handle ordering for async
operations. Use unique rendezvous, stack-compatible collective timeouts and a
longer outer timeout, and destroy process groups in finally.

Propagate rank failures to the parent. A timeout is not a pass. Stop only the
workers created by this run and retain logs before a corrected retry.

**Output:** a focused diff mapped to original matrix rows. Implementing a missing
production backend is separate work routed to `/veomni-new-op`; adding coverage
with no original GPU counterpart belongs in the new-tests list.

## Phase 4: Execute and Wire the Target CI

### 4.1 Run incrementally and diagnose

Run one adapted case, its related parameters, then its file/behavior group.
Use per-attempt logs/JUnit and capture the real exit code before other commands.

| Failure | Inspect first | Response |
|---|---|---|
| Collection/import | Python, import location, optional package relevance | Repair environment or isolate independent contracts |
| Numerical mismatch | Backend, reference, dtype, initialized storage, mutation | Minimize one case and test a single hypothesis |
| Hang/timeout | Rank logs, resource budget, rendezvous, collective ordering | Correct the specific cause before retrying |
| OOM | Workload size, per-rank memory, other users | Allocate suitable resources; do not shrink a performance claim silently |
| Missing asset | Snapshot/revision and cache settings | Supply correct asset or mark blocked |
| All skipped/no collection | Guards and exact selectors | Report unvalidated coverage |

After three failed fixes for the same symptom, revisit the analysis and report the
unresolved cause instead of continuing speculative edits. Continue independent rows.

### 4.2 Connect validated coverage to its owner

1. Read the current NPU workflow and ownership rules in testing knowledge.
2. Check exact-file and directory-level selections, including filters.
3. If already selected, record existing coverage; do not add duplicate commands.
4. Otherwise add a minimal explicit selector under the correct unit/e2e job.
5. Preserve runner/image/dependency conventions and separate CPU-only contract
   overrides from real-NPU/compiler claims.
6. Add a hardware preflight matching each launch's worker count, bounded execution
   and result artifacts. Private machine paths do not belong in public CI.
7. Record whether the command was run locally, in the target image or in public CI.
   A workflow edit alone proves none of those execution outcomes.

**Output:** per-unit results and exact CI-selection mapping. Keep successfully
executed but unwired tests distinct from validated-and-wired migrations.

## Phase 5: Regression and Auditable Coverage Report

### 5.1 Compare with the baseline

Run related original coverage against the final tree and repository quality checks.
Compare file/node IDs, not only totals. Map renamed/parameterized tests explicitly.
If the user limits validation, record what was omitted without claiming success.

Check every final claim against source selection, logs and JUnit. Keep one
authoritative final attempt per unit; retain previous attempts separately. Do not
sum retries or invent a case count for a module-level skip.

### 5.2 Apply fixed denominators

At the declared original granularity, report:

`fully validated and wired original units / all original scoped units`

Use these rules:

1. A file with passing portable cases and blocked original cases is partial.
2. All-skipped, uncollected or unexecuted units are not successfully migrated.
3. Original scope stays fixed; new files/parameters do not enlarge its denominator.
4. Pre-existing NPU selection is not newly migrated credit.
5. Wiring rate and execution coverage are separate metrics.
6. Eligible-only rates require an evidence-backed eligibility list. Keep unknown
   eligibility separate; dependency-blocked does not automatically mean ineligible.
7. CPU/mock validation and real-NPU validation must have separate counts or labels.

Example: among 10 original files, 4 fully pass and are wired, 3 partially pass,
2 are blocked and 1 is unresolved. Full-file completion is 4/10, not 7/7.
If 2 of the 4 were already validated and selected on NPU before this task, only
the other 2 are newly completed migrations. Report new case additions separately.

### 5.3 Deliver evidence and remaining work

Provide one row per original unit:

| Field | Content |
|---|---|
| Provenance | SHA, source workflow/job/step/command and original selector |
| Classification | Direct/adapt/blocked/unresolved and source-level reason |
| Execution | CPU/mock/NPU/ST; Python/CANN/backend/devices and exact command |
| Outcome | Exit code, counts, duration/timeout, final attempt |
| CI selection | Existing/new/missing command with filters and conditions |
| Evidence | Log/JUnit paths |
| Remaining work | Concrete blocker and next action |

Include the final diff, original/eligible denominator explanation and new-test
list. State unexecuted GPU and public-image jobs. Do not include model caches,
credentials or raw runtime evidence in source commits.

**Completion:** all original rows have a resolved outcome or explicit blocker,
claimed successful migrations have execution and CI-selection evidence, and
remaining scope is visible. Maintainer approval, PR publication and merge are
separate outcomes requiring their own evidence.