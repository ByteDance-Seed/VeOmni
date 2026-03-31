---
name: veomni-debug
description: "Use this skill when you encounter ANY error, crash, wrong output, loss divergence, gradient explosion, test failure, CUDA error, distributed training hang, checkpoint load failure, or any unexpected behavior during VeOmni development. Do NOT attempt ad-hoc fixes before invoking this — the four-phase protocol is mandatory. Trigger scenarios: pytest fails, loss becomes NaN, training hangs, FSDP sharding error, model weight mismatch, data collator crashes, sequence parallel output mismatch, MoE routing failure, OOM during training."
---

## Before You Start: Create Debug Todos

**Immediately** use TodoWrite to create todos for all phases. This ensures Phase 5 (knowledge capture) stays visible as a pending task throughout the entire debug session.

```
Phase 1: Investigate <symptom>       -> in_progress
Phase 2: Pattern analysis            -> pending
Phase 3: Hypothesis & test           -> pending
Phase 4: Implement fix               -> pending
Phase 5: Knowledge capture           -> pending
```

Mark each phase as you progress. Do NOT mark Phase 5 complete until knowledge is captured.

## Phase 1: Root Cause Investigation

1. Read the FULL error message / symptom. Don't skim. Extract 2-3 keywords.

2. **Check constraints first**: Read `.agents/knowledge/constraints.md` — many issues are known constraint violations.

3. Reproduce consistently. If you can't reproduce, you don't understand it.

4. `git log --oneline -10` — what changed recently?

5. Trace data flow backward through the call stack.

6. **Distributed training specifics**:
   - Check if error appears on all ranks or just rank 0.
   - FSDP/FSDP2: verify sharding plan matches model structure (`veomni/distributed/parallel_plan.py`).
   - Sequence parallel: check that attention inputs are properly split/gathered (`veomni/distributed/sequence_parallel/`).
   - MoE: verify expert routing and load balancing (`veomni/distributed/moe/`).

## Phase 2: Pattern Analysis

1. Find a **working** example (previous commit, different config, reference implementation).
2. Compare **completely** — diff line by line, not skim. Include config YAML, environment vars, and launcher scripts.
3. Identify ALL differences between working and broken code.
4. Check dependencies — different transformers version? Different PyTorch version?

## Phase 3: Hypothesis and Testing

1. Form ONE specific, falsifiable hypothesis.
2. Design a MINIMAL experiment (change one thing only).
3. Run the experiment. Record the result.
4. If wrong, update understanding and form new hypothesis. No random guess-and-check.

**Red flags — STOP and restart from Phase 1:**
- "Let me just try changing X and see what happens"
- "Quick fix for now, clean up later"
- "It probably works, let me move on"

## Phase 4: Implementation

1. Write a failing test that demonstrates the bug (if feasible).
2. Implement a SINGLE targeted fix addressing the root cause.
3. Verify: test passes, training runs correctly, no regressions.
4. Check for collateral — did the fix break other modalities or trainers?
5. Before committing: run `/veomni-review` skill.

## Phase 5: Knowledge Capture (mandatory — do it NOW, not later)

**Do this immediately after the fix is verified.** Knowledge decays fast.

Check each item:

- [ ] **New hard constraint?** ("this pattern causes a bug") -> add to `.agents/knowledge/constraints.md`
- [ ] **Architecture insight?** -> add to `.agents/knowledge/architecture.md`
- [ ] **New test needed?** -> add to `tests/` for regression prevention

If none apply, explicitly note "no new knowledge to capture."

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
