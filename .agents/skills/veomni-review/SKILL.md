---
name: veomni-review
description: "Use this skill before committing ANY code change — this is a mandatory gate in the commit flow. Also trigger proactively when: you've made changes across multiple files and want to check consistency, you're unsure if a fix is safe, a change touches shared infrastructure (BaseTrainer, distributed, model loading, data pipeline), or a change is larger than a few lines. The review launches a subagent that checks implementation quality, multi-file consistency, and known constraint violations, then rates the change as safe/needs-attention/risky."
---

## Steps

1. Run `git diff` (staged + unstaged) to capture the full diff.
2. Read `.agents/knowledge/constraints.md` for known constraints.
3. **Launch a review subagent** (see below). The subagent receives only the diff + constraints — NOT your reasoning — to avoid confirmation bias.
4. Act on the verdict.

| Verdict | Action |
|---------|--------|
| **safe** | Proceed to commit |
| **needs-attention** | Address listed issues, then commit |
| **risky** | Output the report, do NOT commit, wait for user |

5. Run `make quality` before the final commit.

## Subagent Launch

Use the Task tool with this prompt:

```
You are a code reviewer for VeOmni, a distributed multi-modality training framework. Your job is to find problems in the following diff. You are NOT validating the author's intent — you are looking for bugs, risks, and constraint violations.

## Diff
<paste full git diff here>

## Known Constraints
<paste constraints.md content here>

## Review Checklist

For each changed file, check:

### Implementation Quality
- Hidden risks or edge cases not handled?
- Simpler alternative that achieves the same result?
- Boundary conditions (tensor shapes, distributed rank handling, gradient accumulation steps)?
- Does the fix depend on downstream code to "clean up"?

### Multi-file Consistency
- If a Trainer method changed, do all subclasses need matching changes?
- If model loading changed, are configs and parallel plans updated?
- If data collator changed, do all modalities still work?
- If distributed code changed, are both FSDP and FSDP2 paths handled?

### Constraint Violations
- Does this violate any entry in the known-constraints list?
- Does this repeat a pattern that previously caused bugs?

### VeOmni-Specific Checks
- PR title format: `[{modules}] {type}: {description}`?
- All comments and docstrings in English?
- No auto-generated files (`veomni/models/transformers/*/generated/`) edited directly?
- Ruff-compliant (`make quality` passes)?

## Output

### Verdict: safe / needs-attention / risky

### Findings (for needs-attention or risky)
For each issue:
- **File**: path:line
- **Concern**: what could go wrong
- **Suggestion**: what to do instead
```

## After Commit

- Run `make quality` to confirm ruff compliance.
- Verify PR title follows `[{modules}] {type}: {description}` format.
