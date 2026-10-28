---
name: veomni-review
description: "Pre-PR code review gate. Run before opening a pull request, and again before pushing a substantive update to an open one — not per commit. Required when the PR's branch diff touches Python under veomni/, tasks/ or tests/, or CI workflows, pyproject.toml, uv.lock, docker/ or configs/. Also trigger proactively for runtime or configuration changes that span multiple files, touch shared infrastructure (BaseTrainer, distributed, model loading, data pipeline, ops dispatch), or have uncertain safety. Docs, comments and .agents/-only changes use the self-check below. The review launches a subagent that checks implementation quality, multi-file consistency, and known constraint violations, then rates the change as safe/needs-attention/risky."
---

## When this gate applies

**Once per PR, not once per commit.** The unit that lands is the pull request,
so that is the unit worth reviewing: the reviewer sees the whole change instead
of a slice of it, and a branch of ten commits costs one review rather than ten.
Run it again before pushing a substantive update to an open PR — not for a
typo fix or a rebase.

Commits stay cheap: `make quality` and your own verification still gate every
commit, and nothing stops you invoking this mid-branch when a change worries
you. It is the *obligation* that moved, not the option.

| The PR's branch diff touches | Review |
|--------|--------|
| Python under `veomni/`, `tasks/`, `tests/` | Required |
| `.github/workflows/`, `pyproject.toml`, `uv.lock`, `docker/`, `configs/` | Required |
| Docs, comments, or `.agents/` knowledge and skills only | Skip — self-check instead: verify every repo path, config key and version you assert actually exists, and that every `#L<line>` code link lands on the symbol it names |
| A clean, exact revert or reapplication of a previously approved diff, with no additional or conflict-resolved changes | Skip |

Partial reverts and reapplications with extra edits or conflict resolutions
must follow the normal review gate; prior approval does not cover those changes.

Skipping means skipping the subagent, not skipping verification. Say which
branch you took, so the reader knows a review happened or why it didn't.

## Steps

1. Capture the diff the PR will actually contain — the whole branch against the
   base it targets, not the working tree:
   ```bash
   git diff <base>...HEAD          # <base> is the branch the PR targets
   ```
   Use three dots: it diffs from the merge base, so commits that landed on
   `<base>` after you branched do not show up as your changes. `<base>` is
   usually `main`, but for a stacked PR it is the branch below yours — diffing
   against `main` there would hand the reviewer every PR under you as well.

   If you still have uncommitted work you want included, add it too, and note
   that a plain `git diff` sees neither staged nor untracked files:
   ```bash
   git add -N .                    # intent-to-add: new files become visible, stages no content
   git diff HEAD
   ```
   `git diff HEAD` covers staged and unstaged tracked files; `git add -N` is
   what makes a brand-new module or test visible at all, and it respects
   `.gitignore` so build artifacts stay out.
2. Read `.agents/knowledge/constraints.md` for known constraints.
3. **Launch a review subagent** with your agent's subagent/task mechanism (see
   the prompt below). The subagent receives only the diff + constraints — NOT
   your reasoning — to avoid confirmation bias. Point it at the diff command
   rather than pasting, so it reads the current state.
4. Act on the verdict.

| Verdict | Action |
|---------|--------|
| **safe** | Open or update the PR |
| **needs-attention** | Address the listed issues, then open or update the PR |
| **risky** | Output the report, do NOT open the PR, wait for the user |

5. Run `make quality` before pushing. (It gates every commit anyway, but this
   is the last chance before the diff is public.)

## Anti-Pattern: "AI Self-Inflicted Corner-Case & Defensive Bloat"

A recurring failure mode in AI-authored PRs is the **"self-inflicted defense
loop"**: during self-review, an agent invents hypothetical edge cases ("What if
an `asyncio` task or background thread calls this hook across steps?", "What if
a user's dataset has a key matching our internal private key?", "What if
`metadata_collate_func` secretly flips `input_ids`?", "What if a sample has 0
tokens?", "What if rank 0 has a `torch._neg_view` bit flag and rank 1 does
not?") and then adds a unit test plus hundreds of lines of locks, SHA-256
hashing, preflight collectives, and tensor-cloning guards for each imagined
scenario — ballooning a 40-line change into 3,000+ lines.

Both the outer agent and the review subagent must enforce two rules:

1. **Do NOT invent hypothetical corner cases that violate VeOmni's runtime contract**:
   - Training is **synchronous and single-threaded per rank** (`forward -> backward -> step`). Do not ask for thread locks, `asyncio` task ownership checks, or epoch tokens on trainer/model/callback state.
   - Ranks in the same SP/DP/EP group run the **same config, same model code, and (within an SP group) slice the same deterministic micro-batch**. Do not ask for cross-rank SHA-256/pickle preflight collectives to verify that ranks agree on config flags or batch metadata.
   - Internal hooks (`metadata_collate_func`, model/callback hooks) and collators are **trusted in-process code**, not adversarial inputs. Do not clone tensors before/after hooks to check for mutation or guard against user keys colliding with private `_veomni_*` keys.
   - Dataset preprocessing already filters empty samples and builds monotonic `position_ids` (`0..L-1`) per sample. Do not demand provenance state machines or dynamic-programming solvers for 0-token samples or mid-sample `position_ids == 0` resets.
   - Activations and gradients in LLM/VLM/DiT training are real floating-point tensors. Do not demand handling for `complex64`, conjugate views, rank-asymmetric C++ `_neg_view` flags, deliberate FP16 pre-sum overflow to `NaN`, or manual `untyped_storage().resize_()` aliasing.
   - Do not demand defenses for hostile OS/Python environments (custom `sys.meta_path` finders, broken consecutive namespace packages, processes surviving `SIGKILL`, source files mutated mid-CI run, all-zeros `/proc/sys/kernel/random/boot_id` UUIDs) or wrong Python types (`bool` passed as `int`, generators passed as lists) on internal helpers already behind typed config dataclasses.
2. **Flag speculative defensive bloat and synthetic tests as `needs-attention` (for deletion)**:
   - When a diff contains defensive scaffolding, state machines, extra collectives, or synthetic unit tests for any of the hypothetical scenarios above (or meta-tests that parse AST/source text or test another test's `pytest.skip`), flag them as `needs-attention` and require their **removal**.
   - Never promote a test-invented corner case into `.agents/knowledge/constraints.md` or `.agents/knowledge/testing.md`.

## Subagent Launch

Launch a subagent with this prompt. Use whatever the running agent calls it —
`Task`, `spawn_agent`, or an equivalent — and give it read-only access to the
repo so it can verify claims against the actual files.

```
You are a code reviewer for VeOmni, a distributed multi-modality training framework. Your job is to find problems in the following diff. You are NOT validating the author's intent — you are looking for bugs, risks, and constraint violations.

## Diff
<paste full git diff here>

## Known Constraints
<paste constraints.md content here>

## Review Checklist

For each changed file, check:

### Implementation Quality
- Hidden risks or edge cases in *real* training workflows not handled? (Do NOT invent hypothetical corner cases outside VeOmni's runtime invariants — see "Anti-Bloat & Proportionality" below.)
- Simpler alternative that achieves the same result with substantially less code?
- Boundary conditions (tensor shapes, distributed rank handling, gradient accumulation steps)?
- Does the fix depend on downstream code to "clean up"?

### Anti-Bloat & Proportionality (CRITICAL)
- Does the diff add speculative defensive machinery or synthetic unit tests for scenarios that cannot happen in VeOmni's synchronous distributed training runtime? Flag any of the following as `needs-attention` and recommend **deleting** them:
  - Thread locks, `asyncio` task/thread-ID ownership checks, or epoch tokens on synchronous trainer/callback/op state.
  - Cross-rank SHA-256 digests, `pickle` serialization, or extra preflight `all_gather` / `all_reduce` collectives defending against SP/DP ranks having mismatched configs or corrupted batch metadata.
  - Tensor-cloning guards checking whether internal hooks (`metadata_collate_func`, etc.) mutated inputs, or custom subclasses defending against user dataset keys colliding with private internal keys.
  - Elaborate solvers or provenance tracking for 0-token samples, mid-sample `position_ids` resets, or multi-row packed batches (PackingCollator always packs into 1 row).
  - Handling or testing `complex64`, `.conj()`, rank-asymmetric `torch._neg_view`, deliberate FP16 overflow to `NaN`, custom `sys.meta_path` finders, processes surviving `SIGKILL`, or `isinstance(x, bool)` on internal `int` helpers.
  - Meta-tests that unit-test a test helper's own AST visitor, use `inspect.getsource` to match substrings in trainer methods, or test another test function's `pytest.skip`.
- Is the diff size proportional to the problem solved? If a 40-line fix is accompanied by 500+ lines of defensive scaffolding and synthetic tests, require stripping it down to the essential change.

### Multi-file Consistency
- If a Trainer method changed, do all subclasses need matching changes?
- If model loading changed, are configs and parallel plans updated?
- If data collator changed, do all modalities still work?
- If distributed code changed, are the FSDP2, sequence-parallel and ExtraParallel/EP paths all handled? (FSDP1 no longer exists — a diff that adds an FSDP1 branch is itself a finding.)
- If a trainer lifecycle hook changed, do the composed trainers that override `forward_backward_step()` (`TextDPOTrainer`, `DiTTrainer`) still get it?

### Constraint Violations
- Does this violate any entry in the known-constraints list?
- Does this repeat a pattern that previously caused bugs?

### VeOmni-Specific Checks
- PR title format: `[{modules}] {type}: {description}`?
- All comments and docstrings in English?
- No auto-generated files (`veomni/models/transformers/*/generated/`) edited directly?
- Tests: does the diff extend an existing CI-enumerated test, or add a new file that the workflow owning that path actually lists? Check the owning workflow rather than assuming — `tests/{data,checkpoints,ops,models}/` run wholesale in both unit workflows, `tests/parallel/context_parallel/` runs wholesale only in the GPU workflow, the e2e paths belong to `{gpu,npu}_e2e_test.yml`, and everything else must be listed file by file or it never runs. See `.agents/knowledge/testing.md`.
- Ruff-compliant (`make quality` passes)?

### Documentation
- Does the diff add or change a public API, config field, or extension point (a mixin, hook, callback or base-class method that other modules or models must implement)? Then the same PR must add or update a page under `docs/`. A missing page is a needs-attention finding.
- A new extension point needs a dedicated page that states, for each function: who owns it (must implement / must call / may override / framework-provided), what it does, the exact framework call site, and the observable end effect. The full requirements are in "Documenting an extension point" in `.agents/skills/veomni-develop/SKILL.md`.
- Is every new page in a `docs/index.md` toctree? The docs build runs with `-W`, so an orphan page fails CI.
- Do code references use paths relative to the doc with `#L<line>` anchors, and does each anchor land on the symbol its link text names? Check every anchor into a file this diff changes; line numbers shift silently.
- Do the doc's claims (function names, call order, metric or config keys, defaults) match the code in the diff?

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
