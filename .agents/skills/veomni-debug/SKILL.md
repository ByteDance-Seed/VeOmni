---
name: veomni-debug
description: "Use this skill for ANY bug, error, crash, wrong output, loss divergence, gradient explosion, test failure, CUDA error, distributed training hang, checkpoint load failure, or unexpected behavior. Covers both quick fixes (clear root cause) and complex debugging (unclear cause). Trigger: 'fix bug', 'fix error', 'broken', 'crash', 'doesn't work', 'fails with', 'loss NaN', 'training hangs', 'FSDP error', 'OOM'."
---

## Quick Path vs Full Protocol

| Situation | Path |
|-----------|------|
| Clear error, obvious root cause, fix in <15 min | **Quick Path** (below) |
| Root cause unclear, multiple hypotheses | **Full Protocol** (Phase 1–5) |
| Distributed training issue (hang, wrong loss, sharding) | **Full Protocol** |
| Numerical accuracy / loss divergence | **Full Protocol** |
| 2+ failed fix attempts | **Full Protocol** |

### Quick Path

1. Reproduce the error. Read the full traceback.
2. Check `.agents/knowledge/constraints.md` for known pitfalls.
3. Write a reproducer test if feasible.
4. Minimal fix — root cause only, don't touch surrounding code.
5. Verify: reproducer passes, `pytest tests/<module>/` passes, no regressions across modalities.
6. Run `/veomni-review`, `make quality`, commit.

If not resolved in 15 min → switch to Full Protocol.

---

## Full Protocol

### Before You Start

Use TodoWrite to track all phases:

```
Phase 1: Investigate <symptom>       -> in_progress
Phase 2: Pattern analysis            -> pending
Phase 3: Hypothesis & test           -> pending
Phase 4: Implement fix               -> pending
Phase 5: Knowledge capture           -> pending
```

### Phase 1: Root Cause Investigation

1. Read the FULL error message / symptom. Don't skim. Extract 2-3 keywords.
2. **Check constraints first**: Read `.agents/knowledge/constraints.md` — many issues are known constraint violations.
3. Reproduce consistently. If you can't reproduce, you don't understand it.
4. `git log --oneline -10` — what changed recently?
5. Trace data flow backward through the call stack.
6. **Distributed training specifics**:
   - Check if error appears on all ranks or just rank 0.
   - FSDP2: verify sharding plan matches model structure (`veomni/distributed/parallel_plan.py`).
   - Sequence parallel: check that attention inputs are properly split/gathered (`veomni/distributed/sequence_parallel/`).
   - MoE: verify expert routing and load balancing (`veomni/distributed/moe/`).

### Phase 2: Pattern Analysis

1. Find a **working** example (previous commit, different config, reference implementation).
2. Compare **completely** — diff line by line, not skim. Include config YAML, environment vars, and launcher scripts.
3. Identify ALL differences between working and broken code.
4. Check dependencies — different transformers version? Different PyTorch version?
5. **If a package version upgrade is suspected**, create isolated uv environments to bisect:
   ```bash
   # Create a separate env with the old version
   uv venv .venv-old
   VIRTUAL_ENV=.venv-old uv sync --extra gpu --dev
   VIRTUAL_ENV=.venv-old uv pip install transformers==4.57.3  # or whichever old version

   # Create a separate env with the new version
   uv venv .venv-new
   VIRTUAL_ENV=.venv-new uv sync --no-group transformers-stable --extra transformers5-exp --extra gpu --dev
   ```
   Run the same reproducer in both envs to confirm the version is the root cause. This avoids polluting the main `.venv/`.

### Phase 3: Hypothesis and Testing

1. Form ONE specific, falsifiable hypothesis.
2. Design a MINIMAL experiment (change one thing only).
3. Run the experiment. Record the result.
4. If wrong, update understanding and form new hypothesis. No random guess-and-check.

**Red flags — STOP and restart from Phase 1:**
- "Let me just try changing X and see what happens"
- "Quick fix for now, clean up later"
- "It probably works, let me move on"

**Verification gate** — before acting on a conclusion, check:
- Does the evidence actually support this cause, or just correlate?
- Could a different root cause produce the same symptoms?
- What observation would disprove this hypothesis? Have you looked for it?
- If confidence < 80% or the evidence is ambiguous, launch a verification subagent (see Appendix).

### Phase 4: Implementation

1. Write a failing test that demonstrates the bug (if feasible).
2. Implement a SINGLE targeted fix addressing the root cause.
3. Verify: test passes, training runs correctly, no regressions.
4. Check for collateral — did the fix break other modalities or trainers?
5. Before committing: run `/veomni-review` skill.

### Phase 5: Knowledge Capture (mandatory)

**Do this immediately after the fix is verified.** Knowledge decays fast.

- [ ] **New hard constraint?** → add to `.agents/knowledge/constraints.md`
- [ ] **Architecture insight?** → add to `.agents/knowledge/architecture.md`
- [ ] **New test needed?** → add to `tests/` for regression prevention
- [ ] **Docs outdated?** → update `docs/` if the fix changes API behavior, config semantics, or usage patterns

If none apply, explicitly note "no new knowledge to capture."

---

## Three-Strike Rule

If 3 consecutive fix attempts fail:
- **STOP fixing symptoms.**
- Question whether the underlying approach/architecture is wrong.
- Step back and re-examine: are you solving the right problem?
- Report to user with analysis before continuing.

## Common Pitfalls

- **FSDP2 + gradient accumulation**: gradients must be accumulated in the unsharded space — accumulating sharded gradients produces wrong results.
- **DCP checkpoint format**: model state dict keys must match exactly between save and load — renamed parameters break checkpoint loading silently.
- **Multi-modality data collators**: text-only collators crash on multimodal data and vice versa — always check `data_collator` type matches the dataset.
- **Sequence parallel**: attention outputs must be gathered before loss computation — partial outputs produce incorrect loss values.
- **Patchgen**: model patches in `veomni/models/transformers/*/` are auto-generated — editing generated files directly will be overwritten.

## Domain-Specific Checklists

Include the relevant checklist when investigating.

### Distributed Training Correctness
- [ ] Is the loss identical (within tolerance) between 1-GPU and multi-GPU runs?
- [ ] Are ALL model parameters sharded correctly? (check parallel_plan)
- [ ] Is gradient clipping applied in the correct coordinate space?
- [ ] For sequence parallel: are attention masks split consistently across ranks?
- [ ] For MoE: are expert assignments deterministic across runs with the same seed?

### Numerical Correctness
- [ ] Is there a reference implementation showing the SAME numbers?
- [ ] Are ALL weights loaded? (check logs for missing/unexpected keys)
- [ ] Is the comparison fair? (same inputs, same dtype, same parallelism)
- [ ] Could there be a dtype mismatch? (float32 vs bfloat16 in computation)
- [ ] Are there NaN/Inf values being silently masked or replaced?

## Appendix: Verification Subagent

When confidence is low or evidence is ambiguous, launch a subagent to challenge your conclusion:

```
You are a critical reviewer. Your job is to find flaws in the following conclusion.

## Conclusion Under Review
<the specific claim or decision>

## Evidence Presented
<the data, logs, experiments supporting the conclusion>

## Your Task
1. Does the evidence actually support the conclusion, or just correlate?
2. Generate 2+ alternative explanations consistent with the same evidence.
3. What specific observation would DISPROVE this conclusion? Has it been checked?
4. Was the experiment controlled (one variable changed at a time)?

## Output
Verdict: CONFIRMED / CHALLENGED / INSUFFICIENT_EVIDENCE
Findings: [issues found, counter-hypotheses, missing evidence]
```
