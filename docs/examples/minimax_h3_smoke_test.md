# MiniMax H3 four-GPU smoke test

Date: 2026-10-10. Scope: the paired-cache and CFG-calibration revisions for PR #1258.

## Summary

Forward loss parity and functional training passed for FSDP2 with SP disabled
and enabled. All 12 cases completed three optimizer updates each (36 updates
in total), with finite losses and gradients and nonzero LoRA parameter changes.
**Raw gradient parity passed for SP1 but failed for SP2.** This report does not
claim full numerical equivalence or pretrained-model quality validation.

## Setup and comparison method

| Item | Configuration |
| --- | --- |
| Hardware | One node, four NVIDIA H20 GPUs |
| Distributed modes | FSDP2 SP1/DP4 and FSDP2 SP2/DP2 |
| Model | Randomly initialized H3, 2 layers, hidden size 32, 56 attention heads, head dimension 16 |
| Precision / attention | FP32 / PyTorch SDPA; gradient checkpointing enabled |
| Trainable parameters | Attention and FFN LoRA, rank 4, alpha 4, dropout 0 |
| Optimizer | AdamW, learning rate `1e-3`, weight decay 0, gradient clipping threshold 1 |
| Inputs | Synthetic cached FL2VA and Ref2VA samples; identical batches across DP ranks |
| CFG cases | Disabled; paired negative retaining Qwen visual context; paired negative dropping Qwen visual context |
| CFG settings | Scale 4, sigma-dependent schedule, curvature power 2 for both enabled cases |
| Seeds | Model initialization 42; input/noise seed `100 + step` |
| Loss tolerance | `rtol=2e-4`, `atol=2e-5` |

The manual harness exercised the model, condition processing, collator, LoRA,
FSDP2, Ulysses SP, backward pass, and optimizer. It did not run a full
`DiTTrainer` job with a production dataset or checkpoint-resume lifecycle.
Only the attention head count matches the default H3 model; the remaining
dimensions are reduced for correctness testing.

At each step, the unsharded reference on each GPU received the distributed
student's current weights and exactly the same prepared inputs, including
noise and timesteps. Reference weights were refreshed before every comparison;
this isolates same-weight forward/backward parity rather than comparing two
independently optimized long-running trajectories. Trainable gradients were
reconstructed from their shards and compared before clipping. The enabled CFG
cases also verified that the negative forward ran without gradient recording.

The tested source was based on `b2fd747e79741ec75eb9b803dd8a8b4575d4ba64`
with the local paired-cache review revisions. The submitted tracked patch had
SHA256 `e0f37abfae22aa8e78a3faea0ea3af8d73a74832eb31ab16091c961941436586`.
The H3 attention head loop and output-gather implementation were unchanged
from upstream: neither experimental SP fix was included. Subsequent changes
only updated documentation and removed the excluded head-boundary unit tests.
The manual distributed harness and infrastructure logs are not included in
this commit; the results below are a recorded run, not a new automated CI gate.

## Results

| Check | FSDP2 SP1 | FSDP2 SP2 |
| --- | --- | --- |
| Completed cases / optimizer updates | 6 / 18 | 6 / 18 |
| Maximum absolute loss error against same-weight unsharded reference | 0 | `1.35899e-5` |
| Forward loss parity | Passed | Passed |
| Finite loss and trainable gradients | Passed | Passed |
| Nonzero LoRA parameter changes | All cases | All cases |
| Raw gradient norm / reference gradient norm | 1.0 | `0.499945–0.500062` |
| Maximum relative L2 gradient error | 0 | `0.500055` |
| Raw gradient parity | Passed | Failed |

Gradient parity required both elementwise closeness (`rtol=2e-3`, `atol=2e-5`)
and relative L2 error below `2e-3`. SP2 ran in diagnostic mode: it recorded
gradient mismatches while continuing the functional checks. A successful
process exit therefore does not mean that SP2 gradient parity passed.
The approximately one-half gradient scaling remains unchanged and is outside
this revision's fixes. Forward loss parity alone does not establish equivalent
optimization or long-run training behavior.

### Three-step loss traces

Each cell lists steps 1, 2, and 3. Values are rounded to six decimal places.

| Task | CFG negative policy | SP1 loss | SP2 loss |
| --- | --- | --- | --- |
| FL2VA | CFG disabled | 2.819008, 2.061306, 2.626013 | 2.819002, 2.061311, 2.626028 |
| FL2VA | Keep visual | 2.819701, 2.062173, 2.626988 | 2.819696, 2.062172, 2.626987 |
| FL2VA | Drop visual | 2.822703, 2.062828, 2.626234 | 2.822707, 2.062838, 2.626226 |
| Ref2VA | CFG disabled | 2.815687, 2.061507, 2.623690 | 2.815696, 2.061498, 2.623685 |
| Ref2VA | Keep visual | 2.819896, 2.062820, 2.624957 | 2.819901, 2.062831, 2.624953 |
| Ref2VA | Drop visual | 2.818861, 2.063056, 2.623514 | 2.818863, 2.063050, 2.623515 |

The maximum corresponding-step difference between the SP1 and SP2 traces was
`1.47820e-5`. This differs from the same-weight reference error above because
SP1 and SP2 each applied their own optimizer updates between comparisons.
Three steps with changing synthetic inputs are insufficient to assess convergence.

## Additional checks and limits

- Local H3 packing/CFG unit suite: 93 passed after removing the excluded
  head-boundary cases.
- GPU implicit-device-sync suite: 8 passed, 18 deselected. These used the
  existing small fixtures, separately from the 56-head distributed matrix.
  PyTorch notes that its synchronization debug mode does not detect every
  synchronizing operation.
- Ruff lint/format and documentation task-path checks passed.

These tests do not cover full-size pretrained H3 generation quality, real
Qwen/VAE encoding on GPUs, mixed precision, multi-node networking, throughput,
long-horizon convergence, or checkpoint-resume parity. In particular, synthetic
keep-visual/drop-visual layouts validate the training interface, not the
relative visual quality of those negative-conditioning policies.
