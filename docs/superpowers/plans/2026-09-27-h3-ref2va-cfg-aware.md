# H3 Ref2VA / CFG-aware Implementation Plan

> Execution: inline with executing-plans and test-driven-development; final review
> with veomni-review. No commits, pushes, deployment or training submission.

**Goal:** Extend current public H3 training with a usable Ref2VA recipe and
model-local CFG-aware objectives without copying internal infrastructure.

**Architecture:** Condition preparation owns both layouts and noise sampling.
The transformer owns both forwards and the objective. The existing trainer and
runtime retain optimizer, LoRA, parallelism and checkpoint ownership.

**Tech stack:** Python, PyTorch, Transformers v5, safetensors, pytest, Ruff.

**Spec:** `docs/design/minimax_h3_cfg_aware.md`.

## Global constraints

- Base: GitHub ByteDance-Seed/VeOmni main, `3a9f736f2263479744fa0166ab6e1db7e1c76efc`.
- Branch: `feat/h3-ref2va-cfg-aware-20260927`; no git commits or pushes.
- Preserve default FL2VA outputs and RNG use; no shared trainer or distributed edits.
- New behavior is opt-in via `model.condition_model_cfg`.
- Comments/docs in English; no internal service paths or dependencies.

## Review focus

- CFG branch must reuse noise, reference anchors and timesteps, even when text lengths differ.
- Target geometry mismatch or nonfinite shared embeddings must fail before model execution.
- Detached unconditional branch must not contribute gradients or checkpoint activations.
- Packed sample-mean loss must agree with the serial path, including silent audio.
- No branch must alter unconditional inference or baseline default training.

## Tasks

### Task 1: Objective and configuration

- [x] Establish CPU baseline on `tests/models/test_minimax_h3_packing.py` (41 passed).
- [x] Write failing numerical tests for CFG-calibrated FM loss/gradient scaling,
  constant/sigma schedules and invalid configuration.
- [x] Add model-local `training_objectives.py` and validated condition config fields.
- [x] Test upstream noise/RNG preservation and five-bin video loss weights.

### Task 2: Conditioning and model integration

- [x] Write failing real tiny-model tests for prepared Ref2VA and FL2VA CFG training,
  shared/per-sample layouts, matching noisy targets, detached branch and missing data.
- [x] Add `cfg_conditioning.py` for shared embedding loading and paired-layout preparation.
- [x] Integrate into condition processing and single/packed transformer forwards.
- [x] Verify scale-one regression and packed-versus-serial losses and gradients.

### Task 3: Public recipe and verification

- [x] Add Ref2VA offline recipe and document exact cache contracts and equations.
- [x] Extend H3 tests already enumerated in both GPU and NPU unit workflows.
- [x] Run H3 suite, native LoRA, checkpoint tensor converter and lint; diagnose CPU limits.
- [x] Review uncommitted diff with veomni-review, resolve correctness findings and rerun tests.
- [x] Verify HEAD still equals the upstream base; leave all changes uncommitted.

## Progress / rulings

- Inspection: latest upstream already supports prepared visual Ref2VA; reuse it.
- Isolation: a fresh public clone avoids internal branch/worktree and submodule mutations.
- Scope: first PR slice excludes step gates/warmup/drift and adapter import tooling;
  these need separate runtime integration. Generic upstream LoRA remains available.
- Environment: local Apple Silicon CPU cannot run Linux GPU/NPU extras; use an isolated
  CPU test environment without changing project dependencies or the lockfile.
- Red/green: 13 objective/config tests initially failed; all passed after implementation.
  Six paired-branch tests initially failed; all passed after integration. The malformed
  token-tag case failed before adding modality-tag validation.
- Verification before sampling removal: 239 passed (76 H3 cases) across H3 packing, native LoRA and checkpoint tensor converter;
  one existing PEFT warning about its handling of VeOmni metadata. Python 3.12,
  torch 2.11.0 CPU, transformers 5.16.1, Ruff 0.13.2; `make quality` passed.
- Wider diagnostic: `test_dit_microbatch.py` has 30 setup failures on this Mac because
  its upstream default `load_balancing_loss_implementation=triton` requires Triton.
  This happens in unchanged generic argument validation, before H3/model execution.
  Do not alter generic defaults to make a CPU-only environment resemble GPU CI.
- Hardware gate remains open: no GPU/NPU FSDP2/SP training, pretrained checkpoint
  quality, throughput or distributed-resume equivalence was established here.
- Independent review: safe for local preview; no confirmed correctness findings.
  Two-rank CPU/Gloo FSDP2 was checked against an unsharded reference with single
  samples (checkpointing off/on) and packed samples (checkpointing on), two
  consecutive forward/backward cycles each. Reconstructed gradients matched
  at rtol=1e-4/atol=1e-5. This is not accelerator or SP validation.
- Before a production-training claim: add accelerator parity for FSDP2 + Ulysses
  with unequal branch lengths, BF16, LoRA and checkpointing; validate distributed
  resume equivalence. Keep those checks separate from CPU results.
- Scope update: removed the added mixed noise sampler at the user's request.
  Restore upstream index sampling; remove the sampler option, recipe and tests.
  Keep CFG schedules and video sigma-bin loss weights unchanged.
- Post-removal verification: 238 tests passed (75 H3 cases), `make quality` and
  `git diff --check` passed. HEAD remains the upstream base; no commit or push.
- Interface consolidation: expose one CFG-calibrated FM objective, with
  `training_cfg_curvature_power` in [0, 2] instead of a loss-mode selector.
  Default 2 preserves the inverse-CFG formula; 0 removes curvature attenuation.
  Documentation retains the algebraic equivalence rather than claiming novelty.
- Consolidation red/green: 12 selected tests failed before implementation;
  245 focused tests now pass (82 H3 cases and 163 LoRA/converter cases).
  The full suite stops at E2E collection with `ModuleNotFoundError: exec_scripts`;
  this does not establish a full-suite pass or accelerator validation.
