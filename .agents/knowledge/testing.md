# Test Wiring

How to decide whether a change needs a test, where to put it, and whether CI
will actually run it. Read this before adding a test file.

## CI selects tests by explicit enumeration

A test that passes locally is not necessarily a test CI runs.

| Path | How CI picks it up |
|------|--------------------|
| `tests/data/` | whole directory, in both `gpu_unit_tests.yml` and `npu_unit_tests.yml` |
| `tests/ops/` | whole directory in `gpu_unit_tests.yml`; NPU runs only a few named files |
| `tests/parallel/context_parallel/` | whole directory, `gpu_unit_tests.yml` only |
| everything else | **one `pytest` line per file**, listed in `gpu_unit_tests.yml`, and separately in `npu_unit_tests.yml` when it should run on Ascend |
| `tests/e2e/test_e2e_parallel.py`, `tests/distributed/test_fsdp_equivalence.py` | `gpu_e2e_test.yml` / `npu_e2e_test.yml` |

Consequences that cut both ways:

- Adding a file under one of the directory-level entries above is already
  covered. Adding a workflow line for it is redundant churn.
- Adding a file anywhere else is **invisible to CI** until it is enumerated —
  usually in two workflows. Roughly thirty existing files under
  `tests/{trainer,utils,models,lora,distributed,parallel,e2e}` are in exactly
  this state, so "a test file exists for X" does not mean X is guarded.
- The GPU unit job runs on one self-hosted L20-8 fleet with a 120-minute budget
  and `-x` fail-fast. Every new file pays process startup plus model build, and a
  new flaky file blocks the whole suite for everyone.

Check whether a file is wired — by name, and by the directory it sits in:

```bash
rg -n "$(basename <path/to/test_file.py>)" .github/workflows/
rg -n "$(dirname  <path/to/test_file.py>)" .github/workflows/
```

## Decision rule

Work down the list and stop at the first match.

**1. The behaviour belongs to an already-enumerated test → add a case to it.**
This is the default and needs no workflow change. Most of these files are driven
by a `pytest.param` table, so a new case is a few lines:

| Change | Extend |
|--------|--------|
| New/changed model registration | `tests/models/test_model_registry.py`, `tests/models/test_models_patch.py` (`TEST_CASES`) |
| Patched-vs-upstream numerics | `tests/models/test_models_logits_equal_v5.py` (`CASES` / `_LOADER_CASES`) |
| Host-device sync regressions | `tests/models/test_model_forward_no_implicit_sync.py` |
| VLM trainer / freeze-ViT | `tests/models/test_vlm_trainer.py` |
| MoE checkpoint conversion | `tests/models/test_checkpoint_tensor_converter.py` |
| VLM / Omni dummy forward | `tests/distributed/test_dummy_forward.py` (`_vlm_cases` / `_omni_cases`) |
| `torch.compile` support | `tests/distributed/test_torch_compile.py` |
| Ulysses SP behaviour | `tests/parallel/ulysses/test_ulysses.py` and siblings |
| Checkpoint callback / DCP round-trip | `tests/checkpoints/test_checkpoint_callback.py`, `test_trainer_saveload.py` |
| Weight loading / broadcast / EP shard | `tests/utils/test_rank0_load_and_broadcast_weights.py`, `test_moe_ep_sharded_load_matrix.py` |
| Grad clipping with ExtraParallel | `tests/utils/test_extra_parallel_clip_grad_norm.py` |
| FLOPs / MFU accounting | `tests/utils/test_count_flops.py` |
| Optimizer / Muon | `tests/optim/test_muon_*.py`, `test_optimizer_vlm_param_groups.py` |
| LoRA / MoE-LoRA | the enumerated `tests/lora/test_*.py` set |
| End-to-end parallel parity | `tests/e2e/test_e2e_parallel.py` (`text_test_cases` and friends) |

**2. It is a self-contained kernel or data-pipeline unit test → new file under
`tests/ops/` or `tests/data/`.** Covered automatically on GPU; do not touch
`gpu_unit_tests.yml`. One exception: the NPU job runs `tests/data` wholesale but
enumerates ops files by name, so a new `tests/ops/` file that should run on
Ascend still needs a line in `npu_unit_tests.yml`.

**3. It genuinely needs a new file elsewhere.** Prefer folding it into an
enumerated file in the same directory first. If a separate file is warranted
(different fixtures, different launcher, meaningfully different runtime), then
add it to `gpu_unit_tests.yml`, and to `npu_unit_tests.yml` if the behaviour is
not GPU-specific. Say in the PR what it costs in wall time.

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

- Do not add a test whose only assertion is that the code imports, unless import
  side effects are the thing that broke (model registration is a real example).
- One new `pytest.param` on an existing table is worth more than a new file with
  the same coverage: it runs in CI immediately and costs almost no wall time.
- If you add a file and choose not to wire it, treat it as a local reproducer,
  not as coverage, and do not claim it in the PR.
- Prefer the cheapest tier that can fail for the right reason: CPU unit test over
  single-GPU test over multi-GPU test over e2e training.
