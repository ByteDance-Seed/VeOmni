---
name: veomni-verify
description: "Use this skill when you are about to accept a conclusion, declare something 'normal/expected', attribute a training issue to a specific cause, or propose a workaround — especially after difficult investigation where confirmation bias is likely. Also trigger when: you've been on an issue >1 hour, your confidence is below ~80%, you're about to change >3 files, the evidence is ambiguous, or you are proposing a fix that 'works but you're not sure why'. Launches an adversarial subagent to find flaws in your reasoning before you act on it."
---

## When to Use

At verification gates during debugging, or anytime you're about to act on a conclusion that could be wrong:

- **Design gate** (after root cause analysis): Is the diagnosis correct?
- **Conclusion gate** (after experiments): Does the evidence support this?
- **Implementation gate** (before commit): Is this a proper fix or a workaround?

### Trigger Conditions

Run verification if **any** of these apply:
- You're concluding "this behavior is normal/expected" without a reference proving it
- You're attributing a cause without a controlled experiment
- You're about to use a workaround instead of a root-cause fix
- Your confidence is below ~80%
- You've been on the same issue for >1 hour
- You're about to change >3 files based on your analysis
- The result is ambiguous (could support multiple hypotheses)

## How It Works

Launch a **subagent** whose sole job is to **find reasons your conclusion is WRONG**. The subagent is an adversary, not a validator.

### Step 1: Prepare the Verification Request

Gather:
1. **Conclusion under review**: State it in 1-2 sentences
2. **Evidence**: The data, logs, code, and experiments that led to this conclusion
3. **Context**: Relevant source files, test output, config

### Step 2: Launch Subagent

Use the Task tool with this prompt:

```
You are a critical reviewer. Your job is to find flaws in the following conclusion. You are NOT trying to confirm it — you are trying to BREAK it.

## Conclusion Under Review
<the specific claim or decision>

## Evidence Presented
<the data, logs, experiments supporting the conclusion>

## Your Task

### 1. Evidence Audit
For each piece of evidence:
- Does it actually support the conclusion, or just correlate with it?
- Could this evidence equally support a different conclusion?
- Is there missing evidence that should exist if the conclusion were true?

### 2. Alternative Hypotheses
Generate at least 2 alternative explanations consistent with the same evidence. For each:
- What additional evidence would distinguish it from the original conclusion?
- Has that distinguishing evidence been checked?

### 3. Assumption Check
What assumptions does the conclusion depend on? For each:
- Is it explicitly verified, or just assumed?
- What happens if it's wrong?

### 4. Falsification Test
- What specific observation would DISPROVE this conclusion?
- Has anyone looked for that observation?
- Design one concrete experiment that could falsify the conclusion.

### 5. Methodology Check
- Was the experiment controlled? (one variable changed at a time)
- Is the baseline correct? (right config, right model, right data)
- Could the result be a measurement artifact?

## Output

### Verdict: CONFIRMED / CHALLENGED / INSUFFICIENT_EVIDENCE

### Findings
For each issue found:
- **Issue**: What's wrong
- **Evidence**: What contradicts or undermines the conclusion
- **Suggestion**: What to do instead or what to check next

### Counter-hypothesis (if CHALLENGED)
<An alternative explanation that fits the evidence better>

### Missing Evidence (if INSUFFICIENT_EVIDENCE)
<What data is needed before this conclusion can be accepted>
```

### Step 3: Act on the Verdict

| Verdict | Action |
|---------|--------|
| **CONFIRMED** | Proceed. |
| **CHALLENGED** | Address counter-arguments before proceeding. |
| **INSUFFICIENT_EVIDENCE** | Gather the missing evidence before proceeding. |

## Domain-Specific Checklists

Include the relevant checklist in the subagent prompt when applicable.

### Distributed Training Correctness
- [ ] Is the loss identical (within tolerance) between 1-GPU and multi-GPU runs?
- [ ] Are ALL model parameters sharded correctly? (check FSDP wrap policy)
- [ ] Is gradient clipping applied in the correct coordinate space?
- [ ] For sequence parallel: are attention masks split consistently across ranks?
- [ ] For MoE: are expert assignments deterministic across runs with the same seed?

### Numerical Correctness
- [ ] Is there a reference implementation showing the SAME numbers?
- [ ] Are ALL weights loaded? (check logs for missing/unexpected keys)
- [ ] Is the comparison fair? (same inputs, same dtype, same parallelism)
- [ ] Could there be a dtype mismatch? (float32 vs bfloat16 in computation)
- [ ] Are there NaN/Inf values being silently masked or replaced?

## Rules

- The subagent sees **only the conclusion and evidence**, not your internal reasoning.
- Do NOT skip verification just because you're confident.
- A CONFIRMED verdict does not mean the conclusion is definitely correct — stay alert for contradicting evidence.
