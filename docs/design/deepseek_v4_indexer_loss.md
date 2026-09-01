# DeepSeek-V4 Lightning Indexer KL objective

The Lightning Indexer decides which compressed KV entries CSA attention may
attend to, and until this change nothing trained it: `DeepseekV4Indexer.forward`
returned integer top-k indices, so no gradient reached its 10.75M parameters per
CSA layer — 226M across DeepSeek-V4-Flash's 21 CSA layers, frozen while
everything beneath them moved.

`dsa_indexer_loss` trains it with DeepSeek-V3.2 eq. (4): the KL from the real CSA
attention distribution, restricted to the candidates the indexer itself selected,
to `softmax(index_score)`. The teacher is recomputed in the forward by
`sparse_mqa_target_fwd` from the TileLang attention's own log-sum-exp, summed over
CSA layers, normalised per query token, scaled by `dsa_indexer_loss_coef` and added
to the total loss.

The objective requires sequence parallelism switched off. Ulysses and context
parallelism are the only two modes that enable it and the gate refuses both, for
different reasons recorded below — so an accepted run holds each sequence whole on
one rank, and the per-token normalisation above is already the global one.

The user-facing surface — the two flags, the four metrics, the sparsity regime the
objective needs to be the paper's objective at all, and the measured effect on a
43-layer run — is in
[`docs/usage/arguments.md`](../usage/arguments.md#the-lightning-indexer-kl-objective-dsa_indexer_loss).
This document records what the code cannot tell you on its own: the constraints it
enforces and why, the decisions that took more than one attempt, and the failure
modes that produce a plausible loss curve instead of an error.

That last category is the whole shape of this feature. Almost every way to get
this wrong — a teacher missing the sink, a KL that never reaches the indexer under
reentrant checkpointing, a coefficient of zero that still decays 226M parameters,
a top-k that never binds — leaves a metric that decreases and a run that finishes.
The guards are therefore loud refusals and tests rather than review discipline.

## The path a KL takes

Five call sites in three functions, because the objective changes return arities:

1. `DeepseekV4Indexer.forward` detaches `hidden_states` / `q_residual` and returns
   `index_score` alongside the indices. The scores are already gathered at the
   selected slots by `pytorch_extract_topk_scores`, so the student side of eq. (4)
   is `softmax` over its last axis and the indexer's existing backward supplies
   the rest.
2. The CSA compressor routes them out through `CompressedCandidates.indexer_scores`.
3. `DeepseekV4Attention.forward` asks the attention interface for the teacher over
   the trailing compressed slice, computes the per-query KL and the
   zero-information reference beside it, and returns both as 0-d sums over *local*
   query rows.
4. `DeepseekV4DecoderLayer.forward` passes them up; `DeepseekV4Model.forward` sums
   over the CSA layers and counts them.
5. `DeepseekV4ForCausalLM.forward` takes the per-token mean, folds `coef * kl` into
   the loss, and reports the four metrics. The mean is written as a *local* one
   behind an `sp_enabled` branch that the gate above makes unreachable; see the
   context-parallelism note below for why that branch is there at all.

## Why the teacher is not a softmax over the compressed logits

The target is the *real* CSA attention probability, and V4's CSA denominator has
three parts: the compressed entries, the original-KV sliding window, and the
attention sink. Per head `h`,
`Z_h = exp(a_h) + Σ_{j∈C} exp(c_{h,j}) + Σ_{r∈W} exp(w_{h,r})`; the target
per-head normalises against that full denominator, sums over heads, and
L1-normalises once over the compressed support.

Megatron-LM substitutes `Z_h → Σ_{l∈C} exp(c_{h,l})`, which forces every head to
contribute unit compressed mass before head aggregation even when the head's real
mass sits on its window or sink
([NVIDIA/Megatron-LM#5776](https://github.com/NVIDIA/Megatron-LM/issues/5776),
open since 2026-07-13). Their fused path inherits it structurally: FlashMLA's
`lse_indexer` covers only the first `indexer_topk` entries and `attn_sink` feeds
neither LSE, so no available LSE carries the window and the sink.

This tree's does. `sparse_mqa_fwd` folds the sink into `sumexp` before writing the
LSE, and the indices it consumes span window and compressed entries together, so
its LSE is exactly `Z_h` and the target follows with no correction term.

**That makes the LSE contract load-bearing outside attention for the first time.**
It used to be private to the attention forward/backward pair, so *when* the sink
joins `sumexp` was an internal detail; a future change moving it out would keep
attention correct and silently corrupt the teacher.
`test_reference_target_responds_to_sink_and_window` is the guard, and it is the
reason the fp32 reference in the tests is written out longhand rather than
reusing the implementation: a reference derived from the same code cannot
distinguish the correct teacher from Megatron's.

## Constraints the code enforces

| constraint | enforced at | if violated |
|----|----|----|
| `dsa_indexer_loss_coef` finite and non-negative | `TrainingArguments` validation | `ValueError` at launch |
| the model type implements the objective | `check_indexer_loss_supported`, from `build_foundation_model` | `NotImplementedError` at model build |
| `dsa_indexer_implementation == "tilelang"` | `OpsImplementationConfig` validation, and `_indexer_loss_enabled` again | `ValueError` at launch |
| `dsa_attention_implementation == "tilelang"` | `OpsImplementationConfig` validation, and `_indexer_loss_enabled` again | `ValueError` at launch |
| `ulysses_size == 1` and `cp_size == 1` | `_indexer_loss_enabled` | `ValueError` on the first forward |
| at least one `compressed_sparse_attention` layer | `DeepseekV4Model.forward` | `RuntimeError` on the first forward |
| the compressor handed over its scores | `DeepseekV4Attention.forward` | `RuntimeError` on the first forward |
| teacher width == selection width | `DeepseekV4Attention.forward` | `RuntimeError` on the first forward |
| a layer's return arity matches the shared gate | `DeepseekV4Model.forward` | `RuntimeError` on the first forward |
| the top-k actually binds | not enforced | the dense eq. (3) objective, silently |

Every one of these is keyed on the objective being *on*, and a non-positive
coefficient is off. That reading is the same in all three places that make it —
`OpsImplementationConfig`, `check_indexer_loss_supported` and
`_indexer_loss_enabled` — because `arguments.md` documents the coefficient as the
way to disable the term without editing the flag, and a gate that disagreed would
refuse a shared config for enabling something that config had just switched off.

The model-type gate is an allow-list (`INDEXER_LOSS_MODEL_TYPES`) for the same
reason CP's is, and against a specific run: GLM MoE DSA has a Lightning Indexer of
its own and declares the two neighbouring DSA slots, so `dsa_indexer_loss: true`
on a GLM run is a reasonable thing to write and would produce no error, no metric
and no training. It can only be refused from outside the model, because such a run
never reaches DeepSeek-V4's patched forward where the other three gates live.

The last row is not enforceable from here and is the one most likely to bite; it
needs a sequence length, a compression rate and a *per-sample* length
distribution, none of which are in scope where the flag is validated.
`docs/usage/arguments.md` sets out what to check, and
`configs/text/deepseek_v4_indexer_loss.yaml` derives it for a concrete dataset.

## Decisions that took more than one attempt

**The teacher is recomputed in the forward, not harvested from the attention
backward.** The backward kernel already materialises the quantity, and at `H=64`
one CTA owns every head of a query, so harvesting costs roughly half as much
(~0.5% of step time against ~1%). It was rejected on semantics: the target would
exist only during backward, so there is no forward-time scalar to fold into
`loss`. The KL becomes a gradient injected by the attention Function's backward,
`reduce_sequence_parallel_loss` is unreachable, and the injected gradient's scale
has to be maintained by hand against FSDP's `1/world_size` and a manually reduced
global query count — with nothing type-checking that scale and no natural test
catching it drifting, because any scale still trains *something*. The two differ
only in how the target is produced, so switching later stays a contained change.

**The KL is an explicit return value, not an `OutputRecorder` side channel.**
`router_logits` uses that side channel, which works under the default
non-reentrant checkpointing. It breaks under `enable_reentrant=True`: the custom
class swapped in there runs its first forward under `torch.no_grad()` and then
recomputes under `enable_grad`, backpropagating only through outputs that survive
a `requires_grad` filter. A side-channel capture would yield a graph-less tensor —
a plausible decreasing KL with no gradient reaching the indexer at all. Values
returned through `CheckpointFunction` are grafted into the graph normally, so an
explicit return is correct under both modes;
`test_indexer_receives_gradient_under_checkpointing` asserts on the four
projections' gradients under both settings, not on `kl.grad_fn`, because the proxy
is exactly what a refactor could satisfy while delivering nothing.

**One predicate, read by every site that has to agree.** `_indexer_loss_enabled`
answers "is the objective on", `_builds_indexer_kl` narrows it to "does *this*
layer return a KL", and `_split_indexer_output` applies it to the indexer's own
return. Three functions act on the answer, and a copy that goes stale in any one
of them is an arity mismatch — gating the decoder layer on
`_indexer_loss_enabled` alone would three-unpack the two-tuple that every sliding
and HCA layer returns, three of the four layers of the reference checkpoint.
`test_the_shared_gate_predicts_the_attention_return_arity` and
`test_a_layer_that_disagrees_with_the_shared_gate_is_refused` pin it.

The layer gate keys on `layer_type` rather than on `module.compressor.indexer`
existing, because the two fail in opposite directions: `layer_types` comes from
the checkpoint, so a rename of the compressor's attribute breaks the KL loudly at
the attribute access, while keying on that attribute's name would turn the whole
objective into a no-op with no error and no change of arity.

**`dsa_indexer_loss_coef: 0.0` is off, and it has to be decided at the predicate
rather than at the fold-in.** `loss + 0.0 * kl` is the right *value* while still
building the graph, so the backward writes a zero `p.grad` onto every indexer
parameter — and Muon skips only `p.grad is None` while `_apply_ortho` decays
whatever it steps. That is weight decay on 226M otherwise-frozen parameters, at
the full cost of the teacher kernel, for a term the user switched off. Gating at
the predicate makes a zero coefficient cost exactly what `dsa_indexer_loss: false`
costs, which is what the argument's help text has always promised. The refusals
come *after* the coefficient check for the same reason: a user who switched the
term off has not asked for a TileLang indexer, and refusing their run over the
configuration of a feature they just disabled is advice about the wrong thing.

**Detaching the indexer's input is a change this made, not a property it
inherited.** V3.2 §2.1 detaches the indexer input for separate optimisation. Until
the scores came back out of the indexer, the graph was severed only by accident —
the forward returned integer indices, which carry no gradient. From here the
`hidden_states` / `q_residual` detach is the only thing keeping the auxiliary
objective off the language-modelling one, and
`test_the_indexer_objective_moves_only_the_indexer` is what says so.

**The reduction takes a local mean and a local count**, on the unreachable
sequence-parallel branch. The MoE load-balancing loss is the right precedent for
the *fold-in* shape and the wrong one for reduction: it is folded in after
`ForCausalLMLoss` has returned and therefore misses the SP reduction entirely,
which is fine for a router statistic and not for a per-token objective. So the
indexer KL calls `reduce_sequence_parallel_loss` itself, which re-weights by the
local count before dividing by the global one. Handing it the local *sum* instead
trains perfectly well on a single rank and converges to the wrong cross-rank
weighting — invisible to any single-process test of the value, which is why the
shape is settled now rather than when a sequence-parallel mode is admitted.

The token count is load-bearing on both branches: it is all non-padding query
positions, not the LM loss's unmasked-label count, because eq. (4) sums over query
positions with no reference to labels and a query has a meaningful attention
distribution where its label is masked. One detail belongs to the reduction alone
— the count tensor is `.clone()`d for each of the two calls, because
`ReduceLoss.forward` all-reduces it in place, and a shared tensor would divide the
second term by an SP-world-size-inflated count: correct on one rank, wrong on two.

**The loss keeps the layer sum; the metric is a per-layer mean.** Summed,
`training/indexer_kl` is ~21x larger on DeepSeek-V4-Flash than on the 1-CSA-layer
smoke checkpoint at identical per-layer quality, so no two runs with different
layer counts — and no comparison against an upstream number — would mean
anything. The per-layer mean matches Megatron's `avg_indexer_loss`.

**The captured fraction is a ratio of means within a micro-batch.**
`indexer_kl_captured = 1 − kl / uniform` is the only one of the four metrics that
is interpretable alone, and within the forward that forms it, it is only correct
if both terms arrive through identical denominators: the same per-rank token
count, the same SP reduction, the same layer divisor. Averaging a per-row or
per-rank `kl / uniform` instead yields a number that still lands in [0, 1], still
looks plausible, and is dominated by the rows with the smallest reference.
`test_the_captured_fraction_is_a_ratio_of_means_and_touches_no_gradient` separates
the two aggregations, which requires a fixture discriminating enough to tell them
apart — `num_attention_heads: 64`, the axis the teacher sums over, and
`index_topk: 512`, both from the reference checkpoint. At the toy config's 8 heads
the two land 1.3e-04 apart, under that test's own separation guard, which then
fails rather than passing vacuously.

Beyond the forward, the ratio is not preserved: the trainer averages
`indexer_kl_captured` over micro-steps and over the data-parallel group like any
other auxiliary metric, so the logged value is a mean of per-micro-batch ratios
rather than one minus the ratio of the two numbers logged beside it. That is the
aggregation ruled out above for rows and ranks, and it is admissible here for a
reason specific to what is being averaged: each term is already a mean over a
whole micro-batch of rows, so the terms land within a few percent of one another
instead of spanning orders of magnitude, and the Jensen gap is in the third
decimal place. Making it exact would require the auxiliary-metric path to carry a
numerator and a denominator rather than a value, which is a trainer-wide change.

## The all-missing query row, which is the common case

A query whose compressed slots are *all* misses scores every one `-inf`, and
`log_softmax` of such a row is NaN. Masking after the fact is not enough: the
mask hides the NaN from the returned value, but `log_softmax`'s backward computes
`g − softmax * g.sum(-1)` with `softmax = exp(NaN)`, so even the zero gradient
such a row receives comes back NaN — and the indexer's own backward propagates it,
because it forms `grad * relu(logits)` and `NaN * 0` is NaN. The row is therefore
neutralised on the way *in*, before `log_softmax` sees it.

This is not a corner case. The first `compress_rate − 1` positions of every packed
sample have no complete compression window behind them, so a rate-128 HCA
neighbour or a long packed batch hits it on every step.
`test_indexer_kl_terms_gradient_is_finite_when_a_query_sees_no_compressed_slot`
covers it.

## Interactions this change deliberately leaves alone

**Muon's head split already covers the indexer, and will activate silently.** The
indexer and the main attention both name their up-projection `q_b_proj`, and this
is not a collision the head-split machinery is unaware of — `_head_layout_tiers`
is written for it, resolving the indexer's 8192 rows to 64 × 128 from the module's
own attributes rather than 16 × 512 from the config. So a run with
`--train.optimizer.muon_head_split_modules q_b_proj` has been assigning the
indexer's matrix 64 row blocks all along; it was inert only because `p.grad` was
`None` at step time. This objective populates that gradient, which turns
head-split Muon on for the indexer as a side effect of a flag aimed at attention.
The papers describe plain Muon over the whole matrix, so the paper-faithful
routing is the dimensionality-based one `split_muon_adamw_params` already does,
and `configs/text/deepseek_v4_indexer_loss.yaml` leaves
`muon_head_split_modules` unset. Making the exclusion explicit is a separate
change.

**Context parallelism, which the gate refuses today.** DeepSeek-V4's forward has
no context-parallel path on `main`: no rank exchanges compressor halos or gathers
compressed rows, so at `cp_size > 1` each rank treats its sequence shard as a
whole sequence. Attention is already wrong there, and this objective would build
its teacher out of that attention and report a plausible captured fraction over
it, so `_indexer_loss_enabled` refuses `cp_size > 1` alongside Ulysses rather than
inheriting a silent one.

The reduction it would need is nonetheless written and sits behind
`sp_enabled` — unreachable under that refusal, and left in place because CP query
shards are imbalanced by construction: rank `r` of `C` scores its queries against
roughly `(r+1)/C` of the compressed sequence, so both the number of queries whose
top-k binds and the per-query KL magnitude differ systematically across ranks. A
token-weighted global mean is what makes them comparable where a plain average of
per-rank means would not, and that distinction is invisible to every test that
runs on one rank.

What model-side CP support has to bring with it is backward coverage for the
indexer's own collectives. Its `exchange_compressor_halos` and
`all_gather_compressed_rows` have no backward on *any* rank today — uniformly
unreachable from the loss, the one situation that cannot deadlock — because the
indexer's scores were discarded. This objective makes them reachable, so the
indexer's zero-window path becomes hang-class rather than merely untested: a rank
that stops reading a gathered tensor never enters that gather's backward
all-reduce and its peers wait out the NCCL watchdog.

**Out of scope.** The dense warm-up objective (V3.2 eq. 3), which is only needed
to initialise an indexer from scratch and a released Base checkpoint has already
been through. Ulysses, which needs an `all_reduce` of the partial head sum per
layer per micro-batch. FP4 indexer precision — V4 computes the indexer's attention
in FP4; this keeps the existing bf16 path and computes the target and KL in fp32.

## Open questions

**The LM-loss offset is real and only partly in-semantics.** A flag-on run's LM
loss sits 0.5–1.9% above a flag-off baseline on bitwise-identical batches, where
three baselines agree with each other to 0.02%. Two channels produce it: the
global clip coefficient now depends on the indexer's gradient, and a moving
indexer selects different candidates wherever the top-k binds. Neither is a
gradient leak. Lowering the coefficient is the lever against the first channel
only — the indexer's own motion is coefficient-invariant, because Muon
orthogonalises its update and Adam divides by `sqrt(v)`. A per-indexer clip group
or learning rate would address it and is a recipe choice beyond the reference:
Megatron shares both channels and mitigates neither.

**Nobody has measured whether the indexer was selecting *badly*.** The case for
this objective rests on the indexer selecting *aggressively* — on a 128-card 192K
reasoning SFT run the top-k binds for ~85% of tokens and keeps under 7% of
candidates at the tail of the largest slice — plus a pretrained
`indexer_kl_captured` that sits at ~0.96–0.99 from step 1 and stays flat. Flat
because the teacher moves with the LM, not because nothing is being learned; but
the residual's size is not by itself evidence of drift worth correcting. A
read-only probe comparing the attention mass captured by the indexer's top-512
against an oracle top-512 would quantify it, before and after, and needs no
kernel or training change.

**The cost is measured only on one shape.** MFU on the 43-layer run sits inside
the three baselines' own ±2.9% spread, i.e. the objective is free on step time
there. That is one model, one sequence length and one micro-batch size; the
teacher kernel's cost scales with `index_topk` and the CSA layer count, and the
transient `[B, S, top_k]` fp32 target is ~50 MB per layer at `S=24576`.
