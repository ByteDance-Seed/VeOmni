# Qwen3.5-MoE EP balancing repair audit (2026-09-14)

## Outcome

Three reproducible defects were repaired on top of `feat/issue-1038-ascend`
(`5d0acff1816749da3865d309a30fc7c824a130f4`). These fixes do **not** establish
the performance acceptance criterion of [issue #1038](https://github.com/ByteDance-Seed/VeOmni/issues/1038).
Keep the feature opt-in: the final matched H200 experiment was 1.47% slower
than disabled despite achieving the intended token-load redistribution.
No Ascend/HCCL hardware was used in this audit.

## Defects and repairs

| Defect | Reproduction | Repair |
| --- | --- | --- |
| Greedy planning stops on tied extrema | `[100,100,0,0]` and `[100,100,100,0,0,0]` produced no replicas | Strictly improve the descending-sorted rank-load vector, allowing intermediate moves whose global spread is unchanged |
| Replica-gradient receive layout follows a transpose | A dense strided gradient makes `empty_like(local_grad[row])` non-contiguous; Gloo rejects the receive and its peer loses the connection | Explicit contiguous receive allocation, matching the already-packed sender; preserve logical owner-gradient accumulation |
| EP-global telemetry is counted on every sibling | At EP4, 65,536 actual moved assignments were reported as 262,144 | Only the EP-local leader contributes a global plan; every rank still joins the FSDP/DP+SP monitor collectives |

The first two repairs are `8e18f93` and `94887ce`; the telemetry repair is `8d64d8b`.
No generated model files, optimizer state layout, mathematical routing weights,
or precision settings were changed. Temporary replicas remain non-parameters.

## Validation

- Original planner plus the new regressions: two expected failures, 28 passes,
  one torchrun-only skip. The two tied-hotspot cases then pass; a separate 500-case
  count sweep conserved assignments without increasing maximum load or spread.
- The transposed-gradient regression failed before the receive fix and passes
  with square and non-square shapes `(2,2)`, `(4,2)`, `(2,3)` afterward.
- Telemetry integration failed before the leader fix at EP4 and with two
  non-contiguous EP2 subgroups. Both now pass with exact absolute totals.
- Related planner/executor/dispatch/NPU-seam/config/monitor/reporter regression:
  **193 passed, 3 skipped**. CPU hardware-gate checks: **18 passed, 2 skipped**
  after adding the EP4 E2E fixture. Skips are not accelerator evidence.
- Real four-H200 NCCL probe: all four ranks passed, load `[64,64,0,0]` became
  `[32,32,32,32]`; all three transposed-gradient shapes had zero elementwise
  error. Physical telemetry was exactly 2 replica events, 64 moved and 128 total
  assignments, rather than four times those values.
- All five training arms below completed 32 optimizer steps with finite
  loss/gradient-norm curves. For each matched pair, all 32 points passed
  `rtol=1e-3, atol=1e-4`. Maximum absolute loss difference: `0.0002985001`;
  maximum absolute gradient-norm difference: `0.0013346672` (relative `0.00057513`).
  This is not a claim that every training parameter/gradient is bitwise equal.
- Independent review gates and `make quality` passed using the locked Ruff
  version `0.13.2`. A newer formatter would reformat unrelated baseline files.

The committed hardware E2E now tests both EP2 and EP4, retaining the strict
accelerator prerequisite gate and checking exact replica/moved/total counts.
Both cases passed on four H200s in strict mode (`2 passed`, 204.15 s). These
two-step, blocking-launch cases are functional tests, not performance samples.

## Matched CUDA training experiment

Four H200s on one node; Torch runtime `2.11.0a0+eb65b36914.nv26.02`
(distribution metadata normalizes the suffix to `nv26.2`), Transformers `5.9.0`,
Triton `3.6.0`, FLA/fla-core `0.4.1`, Liger `0.7.0`. Native VLMTrainer/FSDP2,
SP1/EP4, BF16 parameters, FP32 reductions, GBS8/MBS1, 2048-token dummy samples,
four full-attention MoE layers, 16 experts, top-k2. One seeded CPU checkpoint
and one 256-sample dataset were reused and SHA256 recorded. A fixed replay of
logical experts `[0,4]` creates two equally hot owners and two idle EP ranks.

No activation checkpointing or profiler. `CUDA_LAUNCH_BLOCKING=0` is explicit;
the performance wrapper does not import `TestVLMTrainer`, which sets it to `1`.
Each sample times the synchronized forward/backward/optimizer interval and
uses the maximum across ranks; data fetching and end-of-step callbacks are
outside that interval. Reported values are means of **steps 5-32 (28 steps)**,
not maximum or best-step time. This is not whole-job wall-clock time.

| Ordered experiment | Version | Mean timed step (s) |
| --- | --- | ---: |
| First triplet, arm 1 | Disabled, original `5d0acff` | 3.128010 |
| First triplet, arm 2 | Enabled, original `5d0acff` | 3.098271 |
| First triplet, arm 3 | Enabled, planner + receive repairs `94887ce` | 3.141063 |
| Reverse-order final pair, arm 1 | Enabled, all three repairs `8d64d8b` | 3.105436 |
| Reverse-order final pair, arm 2 | Disabled, original `5d0acff` | 3.060469 |

The original enabled arm made **zero** replicas on this workload. Its lower
observed time is not evidence that balancing worked. The final enabled arm
consistently reports imbalance `1 -> 0`, moved fraction `0.5`, and per-step
totals of 16 replica events, 65,536 moved assignments, 131,072 routed assignments.
Its loss and gradient-norm curves are identical to the preceding repaired arm;
the telemetry change did not change those curves.

The first repaired comparison was 0.42% slower than disabled; the final pair
was 1.47% slower. Run-order/environment variation is visible even among disabled
arms. There is no confidence interval or production-scale speedup claim.

## Why further performance work is needed

The load objective measures token imbalance, not net training time. The current
path still, on every MoE forward:

1. Gathers a histogram and moves planning/validation data to the CPU, including
   scalar reads and a full selected-ID copy in `_validate_planner_inputs`.
2. Calls `nonzero` and reads result counts while assigning aliases. CUDA
   `nonzero` synchronizes host and device, as documented by
   [PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nonzero.html).
3. Transfers replica weights again, concatenates all original local expert
   weights with temporary slots, and returns replica gradients with P2P.

These are code-path observations, not a measured percentage attribution from a
kernel trace. A useful next design should first measure those costs separately,
then use a cost-aware admission rule, device-side stable alias construction,
batched transfers, and bounded/versioned reuse across accumulation microbatches.
Reuse must respect FSDP materialization, optimizer updates, checkpoint recompute,
and owner-gradient normalization; simply caching stale expert weights is unsafe.
Do not add those changes without an independent gradient reference and a matched
disabled/enabled performance sweep over balanced, single-hot and tied-hot cases.

The existing toy workload also weakly exposes expert-compute gains. From its
configuration, the routed expert GEMM coefficient is
`4 layers * top2 * 3 * 2048 * 512 = 25,165,824` per token, versus
`2048 * 248320 = 508,559,360` for the output projection alone. This rough
arithmetic inference excludes attention, shared experts and vision; it is not
a profiler result or speedup prediction. Production-shaped experiments and
Ascend hardware validation remain necessary.

## Reproducible artifacts

Local artifact root: `/volume/tzhen/tmps/veomni-ep1038-momo-0914/`.

- `fixture/manifest.json`: checkpoint/data hashes.
- `audit_planner_counts.py`, `planner-count-sweep.json`: the seeded 500-case
  count-only invariant sweep and exact source hash.
- `training/` and `training-final/`: per-arm launch commands, source pins,
  native logs, metrics envelopes and exit codes.
- `training-summary.json`, `training-steps.csv`, `comparison-final.{json,md}`:
  complete-step audit and the repository's standard comparison reporter.
- `nccl-final/rank*.json`: exact gradient and absolute-counter evidence.
- `baseline-*.log`, `regression-telemetry.log`, `cpu-e2e-gates-final.log`:
  failure-first and CPU/Gloo regression evidence.
- `prepare_fixture.py`, `train_fixture.py`, `run_training_comparison.py`,
  `probe_nccl.py`: local reproduction harnesses; these do not modify production
  kernels or the model computation.

Early environment failures are retained separately: inherited `PYTHON_EXEC`
initially selected the system interpreter in torchrun children; missing
`fla-core` stopped the first full-training setup. Neither is counted as a
feature execution result. The isolated environment and child version assertion
prevent those errors from contaminating the successful arms.
