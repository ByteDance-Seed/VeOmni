# Conditional validation methods

The entry point defines when each check applies. Read the relevant sections
here when numerical parity, convergence, resume, or backend evidence is needed.
These methods do not turn every integration into a full training migration.

## Numerical parity after a port or semantic adaptation

Match weights, inputs, masks, dtype, and execution mode. For stochastic
objectives, supply the same sampled noise/timesteps or compare deterministic
intermediates; the same seed need not produce identical random draws across
frameworks. Compare component boundaries when full-model execution is costly.

Record tensor names/shapes, max absolute and relative error, the relative-error
denominator convention, and predeclared tolerances. Establish a native reference
before evaluating a numerically different backend kernel. If the source cannot
run, mark the required comparison blocked and state what narrower checks prove.

## Weight conversion

Check complete source/target accounting, names, shapes, and deliberate dtype
changes. Exercise strict target loading and functional behavior after conversion.
An identity mapping needs loading verification but does not require creating
a converter. Export validation applies when an export consumer is in scope.

## Training and convergence

For training integration, verify finite loss, gradients for expected trainable
parameters, an actual optimizer update, and a short run through the real data
and trainer path. Do not demand a decreasing loss on each stochastic step.

If loss decrease is an acceptance requirement, use a controlled repeated fixture
or another justified experiment. Record the loss series and define the rule
before running, for example comparing initial and final window medians. Report
configuration, steps, and observations. A successful toy overfit is not evidence
of production quality.

## Fresh-process resume

When resume is requested or checkpoint/state handling changes, compare an
uninterrupted N+1-step run against an N-step save followed by a new process
resuming for one step. Record:

- saved/restored global step and applicable state components;
- parameter fingerprints before save and after load;
- next-batch identity and restored RNG state;
- next-step loss and parameter update, with declared tolerances;
- optimizer, scheduler, scaler if used, and data cursor restoration.

Loading weights alone does not prove resume. Exercise the restored state in the
next optimizer step through the intended VeOmni trainer/checkpoint path.

## Backend and performance

Run on each backend and topology actually claimed. CPU checks cannot establish
GPU/NPU kernel or distributed correctness. For requested or affected performance,
record device/runtime, model size, precision, batch/sequence sizes, topology,
warmup, synchronization, measurement interval, throughput, and peak memory.
Compare equivalent workloads; separate startup from steady-state time.

## Reports

For a substantial end-to-end task, adapt the E2E report template. Use the Ascend
experience template when such a report is requested or backend work needs a
durable diagnostic record; NPU support alone does not require every report field.

Each applicable check needs commands and observations. Use N/A with a reason for
inapplicable sections, and BLOCKED for missing prerequisites of a required check.
Replace unresolved placeholders in delivered sections. Reproduction commands
must name documented inputs and not depend on hidden shell state.
