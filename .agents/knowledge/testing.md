# Test Wiring

How to decide whether a change needs a test, where to put it, and whether CI
will actually run it. Read this before adding a test file.

## CI selects tests by explicit enumeration

GPU/NPU suite selections are enumerated in `scripts/ci/accelerator/suites/gpu.json` and
`scripts/ci/accelerator/suites/npu.json`. Both use the common accelerator case scheduler in
`scripts/ci/accelerator/runner.py`; GPU/NPU unit and e2e workflows all use
parallel dispatch. Add a suite selection for a file outside a collected
directory. Do not add per-file pytest workflow lines or manual device quotas.
CPU-only parsing, preprocessing, eager references and mocked backend tests need
an explicit CPU launch plan. Mark entire modules only when all cases are CPU-only;
use individual case markers in files that also exercise accelerator kernels.
The lightweight device-API workflow runs architecture, op-documentation and
generated-call-site-signature tests with `pytest --noconftest`, without Torch.
The patchgen workflow owns `test_modify_init.py`, using the standalone
`patchgen` package and `pytest --noconftest` to avoid runtime fixture imports.
CPU-planned worker cases scope `torch.utils.checkpoint.DefaultDeviceType` to
`cpu` and restore it afterward. On Ascend, CPU-only checkpoint inputs otherwise
fall back to the registered NPU and AMP capability queries can initialize a
hidden device. Preserve this scope across fixture setup and teardown as well
as the test body; accelerator plans and full-visibility collection stay unchanged.
During that scope, optimizer foreach-capability queries use the canonical
`torch.utils._foreach_utils` helper. torch_npu 2.10 replaces the optimizer's
alias with a device-name lookup that initializes NPU even for CPU parameters.
Restore the prior alias on exit, preserving explicit foreach/fused options.
The restoration scope includes setup itself, so partial initialization failures
also undo changes already applied. Unsupported vendor isolation interfaces
raise an explicit error before the test runs.
torch_npu 2.10 also replaces the multi-worker DataLoader iterator constructor
with unconditional NPU synchronization/affinity calls, even for CPU datasets.
CPU-planned cases temporarily use the vendor module's saved Torch constructor
and restore the patch after teardown or failure. Only replace the known vendor
wrapper; preserve unrelated constructor patches and all accelerator plans.
The vendor Inductor wrapper also registers its NPU backend before compiling
CPU graphs. CPU cases temporarily disable that registration through the loaded
vendor registry and restore its prior disabled state after teardown or failure.
Keep real CPU Inductor compilation and gradient checks enabled; do not replace
compiled-mask coverage with eager execution to avoid hidden-device queries.
CPU cases must also register Torch's built-in CPU device-op overrides when
missing. torch_npu populates the override registry at import; Torch 2.10 then
skips all built-ins because the registry is nonempty. Torch 2.11 repairs missing
keys, so testing only that version misses the NPU CI failure. Preserve existing
CPU/vendor entries and validate the pinned Torch version's startup state.
A synthetic registered backend alone misses vendor monkeypatches: exercise
the pinned wheel's optimizer patch in CPU isolation audits too.

CPU numerical flakes need fresh-process reproduction, not only warm repeats.
The QSA CPU fused-kernel mock promotes only `exp` to FP64 and casts back:
PyTorch 2.11 CPU FP32 `torch.exp` was observed to return inaccurate values on
its first parallel call, reproducing the original 757-element CI mismatch.
Keep the adapter's FP32 arithmetic and strict output/gradient tolerances; the
independent FP64 SDPA oracle alone does not fix an inaccurate candidate mock.

New functions/parameters already included by an existing selection need no
manifest update. Check node-ID and pytest filters as well as file/directory
paths. See `docs/testing.md` under "Registering Tests in CI" for the manifest
example and collection verification command.

Resource plans use the same `nproc/world_size` configuration as worker startup.
Use `pytest.mark.launch_plan(provider=...)` for multi-card tests; an unmarked
case defaults to one selected accelerator card, or CPU on a CPU-only host.
Explicit CPU plans reserve zero cards. See `docs/design/ci_case_scheduling.md`
for the author API, defaults and mismatch rules, and `scripts/ci/accelerator/README.md` for
local commands and result artifacts.

A CPU launch plan isolates devices only when the accelerator runner executes
the test. CPU-only launcher regressions must also isolate their subprocess
environment before importing torch, so ordinary pytest on a GPU/NPU host does
not choose an accelerator backend for CPU tensors. Architecture-specific
kernel tests must check hardware eligibility before importing optional kernels.

Distributed rendezvous must own its listener through worker startup. Use
`tests.tools.launch_utils.spawn_with_rendezvous` for raw spawn workers (the
endpoint is appended to their arguments; initialize with `store=endpoint.connect()`)
or the existing `torchrun` callable helper. Single-node torchrun commands use
`--standalone`; CI keeps agent Store sharing enabled. `find_free_port()` only
returns a number and its run registry only coordinates cooperating callers:
probing, closing, then binding in a child still races with other sockets.
After `mp.spawn` returns its context, the helper reaps workers before releasing
the Store on failure, interruption and success. Partial startup failures before
that return rely on the runner's enclosing process-scope cleanup. Never solve a bind race by retrying an entire
training/checkpoint operation that may already have produced side effects.

A test that passes locally is not necessarily a test CI runs. The entries below
describe selections in `scripts/ci/accelerator/suites/gpu.json` and `npu.json`.

| Path | How CI picks it up |
|------|--------------------|
| `tests/ci/` | whole directory, in both GPU/NPU manifests (`suite: unit`) |
| `tests/data/` | whole directory, in both GPU/NPU manifests (`suite: unit`) |
| `tests/checkpoints/` | whole directory, in both GPU/NPU manifests (`suite: unit`) |
| `tests/ops/` | directory in both GPU/NPU unit manifests, excluding `base/test_op_documentation.py` (device-API workflow) |
| `tests/models/` | unit selection in both manifests, excluding `base/test_modify_init.py` (patchgen workflow) and `transformers/test_generated_call_site_signatures.py` (device-API workflow); GPU additionally excludes `test_omni_fsdp.py` |
| `tests/parallel/context_parallel/` | whole directory, GPU manifest only (`suite: unit`) |
| everything else | **one selection per file** in the appropriate GPU/NPU suite manifest |
| `tests/e2e/test_e2e_parallel.py`, `tests/distributed/test_fsdp_equivalence.py` | GPU/NPU manifests (`suite: e2e`) |
| `tests/models/test_omni_fsdp.py` | GPU manifest `61-e2e` (`suite: e2e`); NPU coverage remains in the models unit selection |

Consequences that cut both ways:

- Adding a file under one of the directory-level entries above is already
  covered. Adding another selection for it is redundant churn.
- Adding a file anywhere else is **invisible to CI** until it is enumerated —
  in the GPU and/or NPU suite manifests. Roughly thirty existing files under
  `tests/{trainer,utils,tools,lora,distributed,parallel,e2e}` are in exactly
  this state, so "a test file exists for X" does not mean X is guarded.
- The GPU unit job runs on one self-hosted L20-8 fleet with a 120-minute budget
  and `-x` fail-fast. Every new file pays process startup plus model build, and a
  new flaky file blocks the whole suite for everyone.

The accelerator runner cancels running peers on the first failure by default;
`--keep-going` continues all cases. Foreground logs must identify case progress,
the original failure and cancelled peers without conflating their outcomes.
Accelerator runner thread pools must cancel owned worker scopes before waiting
for callbacks on interruption. Preserve scoped process cleanup and failure
artifacts; do not terminate unrelated device users or rely on the batch timeout.
Use future completion to prove callbacks drained: an interrupted `Thread.join`
can report a stopped thread before its callback finishes cleanup.

## What the NPU job can actually run

Both NPU workflows build their own venv with `uv sync --extra npu`, so the
packages baked into the image's build-time venv are not on the test path. Three
kernel dependencies land differently, and each decides what a test can reach:

| Package | Where it is | What it gates |
|---------|-------------|---------------|
| `triton-ascend` | installed by an explicit workflow step after the sync (`uv sync` prunes anything outside the lockfile, `uv run --frozen` does not) | the vendored Ascend Triton kernels — `causal_conv1d` and `chunk_gated_delta_rule` `triton`, MoE clamped SwiGLU |
| `fla_npu` | image venv only; the workspace venv does not inherit it | `chunk_gated_delta_rule_implementation=fla_npu`, so that backend is untestable |
| `flash-linear-attention` | `gpu` extra only, so absent from both | the `fla` backends. Its `HardwareRequirement` lists `npu`, but the GatedDeltaNet docs say it raises on Ascend even when installed — treat `fla` as GPU-only |

The runners are 910B (A2, non-arch35), so kernels gated behind `is_arch35()`
stay unreachable whatever is installed.

Check whether a file is wired — by name, and by the directory it sits in:

```bash
TEST_PATH=tests/trainer/test_my_new_thing.py
rg -n "$(basename "$TEST_PATH")" scripts/ci/accelerator/suites/*.json
rg -n "$(dirname  "$TEST_PATH")" scripts/ci/accelerator/suites/*.json
```

## Decision rule

Work down the list and stop at the first match.

**1. The behaviour belongs to an already-enumerated test → add a case to it.**
This is the default and needs no workflow change. Most of these files are driven
by a `pytest.param` table, so a new case is a few lines:

| Change | Extend |
|--------|--------|
| New/changed model registration/build | `tests/models/base/test_auto_registry.py` (`_MODEL_CASES`) |
| Patched-vs-upstream numerics | the matching family test under `tests/models/transformers/` or `tests/models/diffusers/` |
| Host-device sync regressions | `tests/models/transformers/test_model_forward_no_implicit_sync.py` |
| VLM trainer / freeze-ViT | `tests/trainer/test_vlm_trainer.py` |
| MoE checkpoint conversion | `tests/models/base/test_checkpoint_tensor_converter.py` |
| VLM / Omni dummy forward | `tests/distributed/test_dummy_forward.py` (`_vlm_cases` / `_omni_cases`) |
| `torch.compile` support | `tests/distributed/test_torch_compile.py` |
| Gradient checkpointing / selective recompute (SAC) | `tests/distributed/test_gradient_checkpointing.py` |
| Ulysses SP behaviour | `tests/parallel/ulysses/test_ulysses.py` and siblings |
| Checkpoint callback / DCP round-trip | `tests/checkpoints/test_checkpoint_callback.py`, `test_trainer_saveload.py` |
| DiT resume / noise-stream equivalence | `tests/checkpoints/test_dit_resume_align.py` |
| Weight loading / broadcast / EP shard | `tests/utils/test_rank0_load_and_broadcast_weights.py`, `test_moe_ep_sharded_load_matrix.py` |
| Grad clipping with ExtraParallel | `tests/utils/test_extra_parallel_clip_grad_norm.py` |
| FLOPs / MFU accounting | `tests/utils/test_count_flops.py` |
| Optimizer / Muon | `tests/optim/test_muon_*.py`, `test_optimizer_vlm_param_groups.py` |
| LoRA / MoE-LoRA | the enumerated `tests/lora/test_*.py` set |
| End-to-end parallel parity | `tests/e2e/test_e2e_parallel.py` (`text_test_cases` and friends) |

**2. It is a self-contained model, kernel, or data-pipeline unit test → new file
under `tests/models/`, `tests/ops/`, or `tests/data/`.** These directories are
collected wholesale by the GPU and NPU unit workflows, so do not add a per-file
workflow entry. Hardware markers and dependency guards inside the test decide
which cases each job runs.

**3. It genuinely needs a new file elsewhere.** Prefer folding it into an
enumerated file in the same directory first. If a separate file is warranted
(different fixtures, different launcher, meaningfully different runtime), then
add it to the GPU suite manifest, and the NPU suite manifest if the behaviour
is not GPU-specific. Say in the PR what it costs in wall time.

**4. Add no test.** Legitimate cases:

- Pure refactor with the behaviour already covered by an existing test.
- Docs, comments, config-example, or agent-knowledge changes.
- Changes only observable on hardware CI does not have (multi-node, SM90+
  kernels on the SM89 runners, NPU-only paths with no NPU coverage for that
  area).
- Changes whose only failure mode is a build/lint error that another gate
  already catches (`check_patchgen`, `device_api_check`, `check_doc_task_paths`,
  ruff via `check_pr_lint`).

Say so explicitly in the PR's **Test** section, with the reason. "No test
needed" without a reason is what the reviewer is looking for.

## Guidance

- Compiled backward can donate saved buffers and reject `retain_graph=True`.
  Numerical comparisons should use independent leaf tensors for the reference
  graph and differentiate each graph once without retaining it. Keep test inputs
  deterministic; do not disable compiler buffer donation globally for a test.
- A test double wrapping a `VeomniOp` must expose the handle attributes its
  consumer reads, including `impl`. DeepSeek-V4 keeps eager routing weights in
  their incoming dtype and casts them for fused MoE; a recording double must
  identify which path it represents and assert the corresponding dtype.
- CPU substitutes for Diffusers RMSNorm on Ascend hosts must preserve the
  pinned upstream CPU math and dtype promotion. `F.rms_norm` is not a drop-in
  replacement for BF16 inputs with FP32 weights: its BF16 output cannot feed
  an FP32 Linear without autocast. Exercise the Ascend-host CPU fixture path
  on non-Ascend CI through indirect fixture parametrization.
- A standalone Transformers layer can allocate parameters with `torch.empty`
  that only the enclosing `PreTrainedModel.post_init()` initializes. Initialize
  these in the fixture before any rank broadcast; setting a seed does not
  initialize empty storage. DeepSeek-V4 attention needs both `sinks` and the
  compressor/indexer `position_bias` initialized. Poisoning fresh floating-point
  allocations with NaN in a CPU fixture test catches this without depending on
  which values a GPU allocator happens to recycle.
- Do not add a test whose only assertion is that the code imports, unless import
  side effects are the thing that broke (model registration is a real example).
- One new `pytest.param` on an existing table is worth more than a new file with
  the same coverage: it runs in CI immediately and costs almost no wall time.
- If you add a file and choose not to wire it, treat it as a local reproducer,
  not as coverage, and do not claim it in the PR.
- Prefer the cheapest tier that can fail for the right reason: CPU unit test over
  single-GPU test over multi-GPU test over e2e training.
