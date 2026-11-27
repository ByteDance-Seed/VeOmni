# DeepSeek-V4 Context Parallelism

Context parallelism (CP) is DeepSeek-V4's alternative to Ulysses for scaling
sequence length. Rank `r` of `cp_size` holds tokens `[r*L, (r+1)*L)` where
`L = S / cp_size`, keeps all attention heads, and computes attention only for its
own queries. The KV it hands to the sparse kernels — full-resolution and every
compressed part — is all-gathered, so it is byte-identical to the single-rank
one. That is the decisive property of the design: because the KV buffer is
replicated, every index the attention kernels consume stays global and needs no
remapping. Only the query axis is sharded.

It is selected with `model.accelerator.cp_size > 1` and `ulysses_size == 1`; the
two are mutually exclusive in milestone 1.

The constraints and tradeoffs below explain why this model uses CP instead of
Ulysses.

This document records what the code cannot tell you on its own: the constraints
it enforces and why they exist, the decisions that took more than one attempt,
the failure modes that hang rather than raise, and the questions that were left
open deliberately.

## Scope: what milestone 1 does and does not do

Milestone 1 buys numerical parity with very few moving parts, and accepts more
communication than necessary to get it.

| in | out (milestone 2 or later) |
|----|----|
| Contiguous sharding, `cp_size` in {2, 4} verified | Zigzag / load-balanced sharding |
| All-gather for full-resolution KV, compressed rows and compressor halos | P2P halos; multi-hop halos |
| DeepSeek-V4 only, via an allow-list | Any other model |
| GPU; Ascend/NPU with the eager or CANN DSA operators | Device-measured CANN parity under CP |
| `cp_size > 1` with `ulysses_size == 1` | Hybrid CP x Ulysses |
| Training | KV cache and decode paths |
| Correctness | Measured performance; `cp_size=8` |

Anyone extending this should know what the contiguous layout is load-bearing
for: windows sort by start token and shards sort by rank, so concatenating each
rank's compressed rows in rank order already reproduces the global compressed
array with no restore permutation, and a local query row `i` is global row
`cp_rank * L + i` with no lookup. A striped or load-balanced layout invalidates
both, in the attention forward, in `shard_packed_compression_metadata` and in
the indexer independently — all three derive the shard's origin from
`cp_rank * local_seq_len` and nothing communicates it.

The safety net for any such change is `tests/parallel/context_parallel/`, which
compares forward output, input gradient and every parameter gradient against a
single-rank baseline at `cp_size` 2, and on CUDA also at 4, packed and unpacked,
with and without the compressor. CUDA CI runs both topologies; NPU CI runs the
two-rank case, where the packed-model branches do not execute. `cp_size=8` has
never been run.

## Constraints the code enforces

| constraint | enforced at | if violated |
|----|----|----|
| `cp_size <= S / max(config.compress_rates)` | `plan_compressor_shard`, called by all three window compressors before their halo exchange | `ValueError` on the first forward |
| `cp_size > 1` requires `ulysses_size == 1` | `ParallelState.__post_init__` and `AcceleratorConfig._resolve_topology` | `NotImplementedError` at launch |
| the model type must implement CP | `check_context_parallel_supported`, from `build_foundation_model` (the omni path reaches it the same way, once per sub-module) | `NotImplementedError` at model build |
| on Ascend/NPU, the model type must be `deepseek_v4` | `check_context_parallel_supported` | `NotImplementedError` at model build |
| on Ascend/NPU, `dsa_indexer_implementation` and `dsa_attention_implementation` must be `eager` or `cann` | `check_context_parallel_supported`, which reads the adapted config's `ops_config` | `NotImplementedError` at model build for any other value |
| explicit `position_ids` under either sequence-parallel mode | `DeepseekV4Model.forward` | `ValueError` on the first forward |
| the attention mask spans the full sequence | `DeepseekV4Attention.forward`, ahead of the KV all-gather | `ValueError` on the first forward |
| no KV cache under CP | `DeepseekV4Attention.forward` | `NotImplementedError` |
| shards are contiguous and equally sized, carrying global `position_ids` | not checkable from one rank; see below | silently wrong results |

Three of these are worth more than a table row.

### Sequence length and compression rate

`cp_size <= S / max(config.compress_rates)`, equivalently `L >= R` for every
configured compression rate. The largest rate is the binding one.

The reason is the halo geometry. A window owned by rank `r` can extend up to
`R - 1` tokens past the end of `r`'s shard, and the overlap half of `r`'s first
owned window begins `R` tokens before that window's start. Both are supplied by
`exchange_compressor_halos`, which reads the *immediately adjacent* ranks and
only those. If a shard is narrower than one rate, a window's tail or its
predecessor lands two or more ranks away and no halo carries it.

The practical risk is a short sequence with a wide rate. Production carries its
rates in the checkpoint's HF config, which is not vendored here, and a 512-wide
rate caps `cp_size` at `S/512` — at most 256 ranks at `S=131072` and at most 8
ranks at `S=4096`. There is no launch-time check, so an operator who exceeds it
learns so on step 1 of the job rather than at submission.

That is a deliberate decision rather than an omission, and worth knowing before
anyone "fixes" it. No single argument determines the width the guard checks: the
collator rounds the *actual* packed length up to a multiple of
`sp_size * sp_pad_scale`, while `data.max_seq_len` is a per-sample budget that
`train.pad_to_length` multiplies by `micro_batch_size` and that dynamic batching
need not reach. And the model config carrying `compress_rates` is not in scope
where those arguments are validated — `build_foundation_model` never receives a
sequence length. A launch-time check would therefore have to either re-derive the
collator's length arithmetic, giving a second and weaker copy of this constraint,
or thread a sequence length through model construction for one model's benefit.
The runtime guard checks the width that actually matters and names the caller
that hit it.

Lifting the constraint means multi-hop halos: exchanging with the
`ceil(R / L)` nearest neighbours instead of one, which changes
`exchange_compressor_halos` and the `+R` index rebase in
`rebase_window_indices` together. That is the same change that would replace the
all-gathers with P2P, so it belongs with milestone 2.

### The contiguous-equal-shard contract

Rank `r` receives rows `[r*L, (r+1)*L)` of the global sequence, contiguous and
equally sized, carrying their *global* `position_ids` (for packed data the
per-sample positions, which is what makes them global), while `cu_seq_lens_q`
still spans the whole packed batch. `SequenceParallelCollator` supplies this;
`DeepseekV4Attention.forward`, `shard_packed_compression_metadata` and
`DeepseekV4Indexer.forward` each reconstruct the shard's origin from it
independently.

No single rank can verify it. The one visible part is the attention forward's
mask-length check, which derives the global length as `local_seq_len * cp_size`
and therefore also catches an uneven split. Fabricated positions were a real
defect on this branch: `arange(local_seq_len)` tells every rank its shard starts
at position 0, which keeps every shape self-consistent while every rank above 0
compresses the wrong rows and the indexer's canonical-position check admits the
TileLang kernel on rank 0 alone. Hence the refusal rather than a default.

### Ascend/NPU operator selections

`check_context_parallel_supported` admits DeepSeek-V4 on Ascend/NPU and then
checks the operator pair on the config it was handed, which is why the check
lives there rather than on the model config: a native checkpoint config is
adapted into a generic class that keeps `ops_config` but not
`DeepseekV4Config.validate_build_prerequisites`. Both DSA implementations that
ship for NPU are accepted, and each adapter rebases the shard onto the global
buffer itself:

- `dsa_attention_implementation: cann` sees `L` queries over an all-gathered
  `S_global` KV buffer plus the row where the shard starts. The kernel has no
  offset input and numbers its queries from the KV end, so the adapter narrows
  the buffer to the prefix `[0, query_offset + q_len)` and the local queries end
  at the last row on every rank. Packed shards then become one single-segment
  call per sample the shard intersects. A prefix — or a truncated packed sample —
  that ends inside its first compression window holds no visible compressed row,
  so it runs the eager row with a zero-row compressed table. The compressed
  selection must either span the table it indexes or be 512/1024 wide. Those are
  the kernel's own widths — the vendor wrapper pads anything at most 512 wide up
  to 512 — so a 512/1024-wide selection is served even over a wider table, and the
  shipped `index_topk: 512` always produces one of those widths. The prefix grows
  with the rank, `(cp_rank + 1) * L / rate`, so rank 0 spans its table exactly when
  `L == rate * index_topk` and every later rank exceeds it. In the validation run
  (`max_seq_len: 4096`, `cp_size: 2`, rate 4) that is a 512-row table on rank 0 and
  1024 rows on rank 1, so the two ranks exercise the spanning and the wide-table
  case in the same step; at the shipped `max_seq_len: 8192` the tables are 1024 and
  2048 rows, both wider than the selection, and only the kernel width keeps them
  served. A selection below 512 that does not span its table returned
  nonfinite rows on CANN 9.2.1, so the adapter refuses it; the shipped model only
  makes that call for an `index_topk` other than 512/1024 over a wider table (see
  [`docs/usage/arguments.md`](../usage/arguments.md)).
- `dsa_indexer_implementation: cann` derives causal visibility from the query
  row and accepts only the ranges the shard's global rows imply. CP supplies
  explicit `cu_seqlen_ks` / `cu_seqlen_ke` rebased onto the global sequence, so
  the adapter validates them against that plan on every call, narrows the
  compressed table to the same prefix, and selects per sample on packed shards.
  A range set built for another shard is refused instead of masked from the
  wrong origin.

The rest of the NPU selections are CP-safe and stay on their kernels:
`mhc_implementation: cann` sees per-token tensors with no sequence geometry, and
`moe_implementation: fused_npu` / `cross_entropy_loss_implementation:
chunk_loss` are unaffected. There is no separate CP configuration: a CP run is
[`configs/text/deepseek_v4_npu.yaml`](../../configs/text/deepseek_v4_npu.yaml)
with `cp_size` set above 1 and `ulysses_size` left at 1. That config ships with
`dsa_indexer_implementation: cann` and `dsa_attention_implementation: cann`;
override both to `eager` to reproduce the eager baseline.

The gate validates that selection pair and nothing else: it cannot check that
either kernel is numerically correct on Ascend, and
`cp_size <= S / max(config.compress_rates)` applies unchanged.

Validation beyond configuration is training comparison, not tensor parity. With lr
`3e-4` the optimizer turns a bit-level difference into a trajectory difference
within tens of steps — the eager baseline below, fully deterministic and at the
same seed, already reaches ±4% — so the CANN comparison is stated against the
spread of repeating the same configuration rather than against a fixed percentage.

The earlier eager baseline: `configs/text/deepseek_v4_npu.yaml` overridden to
GSM8K, `max_seq_len=512`, `global_batch_size=32`, lr `3e-4`, seed 42,
`enable_full_determinism: true`, `dp_size=4`, `ep_size=4`, both DSA selections
`eager`, trained for 71 steps at `cp_size=1` and `cp_size=2`; the loss relative
error stayed within ±4%. At that sequence length the compressed table holds at
most 128 rows, so this run exercises a selection that spans its table, not the
kernel's 512/1024 widths.

The CANN comparison is a two-layer CSA+HCA smoke run — `num_hidden_layers=2`,
`layer_types=[compressed_sparse_attention, heavily_compressed_attention]` — at
`max_seq_len=4096`, `index_topk=512`, `global_batch_size=32`,
`micro_batch_size=1`, lr `3e-4`, seed 42, with `enable_full_determinism` off
because sparse FlashMLA does not support it. Two operator families, attention
`cann` with the indexer `eager` and the indexer `cann` with attention `eager`, each
compared `cp_size=1` against `cp_size=2` over steps 1–99. The topologies also
differ in world size: `cp_size=1` ran `world 4 / dp 4 / ep 4` and `cp_size=2` ran
`world 8 / dp 4 / ep 8`, so the comparison varies CP and expert placement
together.

The early steps agree closely: step 1 loss differs by at most 0.002% across all
five runs, and every step in 1–40 stays within 0.11% — 0.11% in the indexer
family, 0.05% in the attention family. Both families enter a gradient-spike and
loss-plateau regime at steps 44–51, the first step whose loss moves by more than
0.5% being 44 and 45 in the two indexer runs and 49 in the attention run, and from
there single steps diverge. Over steps 1–99:

| comparison | p50 | p90 | max |
|----|----|----|----|
| indexer, `cp_size=2` run A vs `cp_size=1` | 0.19% | 2.2% | 3.7% |
| indexer, `cp_size=2` run B vs `cp_size=1` | 0.20% | 1.8% | 2.9% |
| attention, `cp_size=2` vs `cp_size=1` | 0.13% | 1.6% | 2.6% |
| noise floor: indexer run B vs run A | 0.15% | 1.1% | 2.0% |

The 10-step means alternate in sign around +0.2% rather than drifting. Repeating
the same `cp_size=2` configuration already reaches 2.0% on a single step, and
swapping which DSA selection runs `cann` at a fixed topology — both
implementations are separately validated, so this isolates numerics rather than
correctness — moves the loss by sd 0.6–0.9%, the same scale. The `cp_size=2`
spread is therefore run-to-run reproducibility amplified through the spike region
rather than a systematic bias: it is absent in the early steps and shows no
accumulating offset. None of this is a per-tensor bound. `cp_size=8` has never
been run, and an `lr=0` comparison and a `cp_size=1` run at `world 8 / ep 8` are
still outstanding controls.

`tests/parallel/context_parallel/` runs its two-rank eager case on the NPU
suite, so the eager path's rank and local-query/global-KV geometry is checked on
device; the four-rank and packed-model branches stay CUDA-only. The CANN
adapters' CP geometry — the prefix narrowing, the per-sample rebasing and the
eager fallbacks — is checked on any host by
`tests/ops/dsa/test_dsa_deepseek_v4_cann.py`, which records the arguments the
adapters hand their kernels. On Ascend the same file compares the real CANN
indexer against eager selection, causally and on both packed CP shards. The same
file compares the real CANN attention kernel with eager — forward and every input
gradient — on 512-wide selections, one of the two widths the kernel is built for,
over a 75-row table and a 513-row table: the width is exercised both wider and
narrower than the table. In the validation run (`max_seq_len: 4096`,
`cp_size: 2`) the two ranks hold 2048 query rows each, but their narrowed
compressed tables are 512 and 1024 rows: rank 0's 512-wide selection spans its
table while rank 1's does not, so the same step covers both branches; the
513-row parity case covers the wide-table branch on its own.

## Decisions that took more than one attempt

**Window ownership follows the first token.** Window `w` starting at token `s_w`
belongs to rank `s_w // L`, so it is computed by exactly one rank even when its
tokens span two. The alternative — aligning shard boundaries to window
boundaries — would require every packed sample length to be a multiple of the
compression rate, and the packed metadata deliberately supports arbitrary
lengths (`build_packed_compression_metadata` walks
`range(start, end - R + 1, R)` and drops the incomplete tail). Absorbing
straddling windows is what that choice buys, at the cost of two halos and
unequal per-rank window counts. The counts are not communicated: every rank
builds the same global `window_starts`, so every rank can compute every other
rank's count as pure local arithmetic.

**Pad-to-max gather is a performance choice, not a correctness one.** Ranks own
different numbers of windows, so each pads its compressed rows to
`n_max = max_r n_r` before the all-gather and slices the valid rows back out
afterwards. The branch first recorded that `_all_gather` *requires* equal
shapes; that was wrong — its own docstring says shards may differ in length, and
it exchanges shapes before allocating a tensor per rank. The padding stays for
the collective underneath: PyTorch's NCCL `all_gather` coalesces equally sized
outputs into a single `ncclAllGather` and degrades to one broadcast per rank
when they differ, which is `cp_size` launches per compressor and per indexer per
layer. The distinction matters to anyone reading this code: correctness holds
either way, so deleting the padding loses the coalesced collective rather than
gaining simplicity, and the wrong reason is what motivated threading `counts`
through the call in the first place. Unpadding is a deterministic index derived
from `window_starts`, so the gather stays differentiable and the padding rows
receive no gradient.

**The query axis had to be separated from the KV geometry.** Expanding the
sparse contract into candidate slots needs three lengths that coincide under
Ulysses and separate under CP: the number of query rows, the absolute
full-resolution KV row of each sliding candidate, and the offset that lifts a
compressed slot past the full-resolution rows. Under CP the first becomes `L`
while the other two stay `S_global`. `dsa_attention` therefore takes the
full-resolution and compressed KV as separate tensors, so both KV lengths are
read off their own shapes, plus a `query_offset` that rebases a local query
onto its global row. `query_offset` defaults to 0, which is right for every
caller that is neither CP nor decode — so the type system cannot catch a missed
call site. This whole class of error is silent. Shapes stay self-consistent and
the model attends to the wrong rows, so the mitigation is the parity test that
compares each shard's candidate slots against the matching slice of the
full-sequence build.

**Compressed slot values stay global; window token indices are rebased.** Two
index spaces look alike and must not be confused. Attention and indexer indices
— `topk_indices`, sliding-window indices, `range_starts` / `range_ends` —
address the *replicated* global KV, so their values are
untouched under CP and the arrays are only sliced by query row. The compressor's
`window_indices` address the rank's own `kv` / `gate` tensors and so are rebased
by `cp_rank * L` **and** by the halo width, since they index
`[left halo | shard | right halo]`. `window_starts` is rebased by the shard
offset only, because it is used to index `position_ids`, which carries no halo.
`shard_packed_compression_metadata` applies all of this on the packed path; the
unpacked path rebases through `local_window_token_indices`.

## Failure modes that hang instead of raising

**Backward participation under CP is decided by the autograd graph.** A rank
that stops *reading* a gathered tensor never enters that gather's backward
all-reduce, and its peers wait out the NCCL watchdog — ten minutes of silence
rather than an assertion. This has cost the project time twice. First as a real
defect: a rank owning no compression window returned `new_zeros`, which is
detached. Then again while verifying that fix, when a mutation removing the right
halo hung instead of failing — rank 0 was the only rank then reading neither halo
(its left halo is the end-rank zero buffer), so the slab gather dropped out of
its graph, rank 0 completed its backward, and ranks 1 through 3 timed out in
`WorkNCCL(... ALLREDUCE ...)`.

`empty_compressed_rows` is the fix, and its shape is not incidental: it returns
empty slices of *both* `kv` and `gate` so the zero-window result stays attached
to both. Simplifying it back to `new_zeros`, or to a slice of `kv` alone,
reintroduces the hang — the second form subtly, since it leaves the kv halo's
gather reachable and only the gate halo's unreached. All three window
compressors and the packed compression helper call the one copy, because the
original fix reached one of four and missed the rest.

A sturdier design would gather `kv` and `gate` in one collective and slice, or
drop the autograd-function gather for explicit unconditional collectives. Both
touch `veomni/distributed/context_parallel/dsa_cp.py`.

The same property constrains tests: a CP test must finish every collective
before it asserts, or a genuine parity failure on one rank strands the others in
a watchdog timeout instead of reporting the mismatch.

**The TileLang indexer no longer demotes silently.** Selection is
`dsa_indexer_implementation` through a local `VeomniOp`. Choosing `tilelang`
resolves that registry row; there is no `use_tilelang` conjunction that falls
back to the eager scorer. Tests that claim kernel coverage still need a
pass-through counter, because a CP parity check can stay green if the kernel is
never entered. Nothing currently runs the TileLang indexer inside a bf16 CSA
layer under CP — layer parity is float32 and the indexer's own test is a bare
bf16 module — so that pin is what a future test of that combination would need.

## Where the implementation diverged from the design and the plan

Recorded because the reasons were discoveries, not tidying.

| document said | built instead | why |
|----|----|----|
| `sharding.py` holds contiguous slice / restore helpers and a divisibility check `L % max_compress_rate == 0` | window-ownership arithmetic plus the three helpers every window compressor shares | Slicing is already free — the collator's `sp_slice` narrows on `sp_rank`, which resolves to `cp_rank` under CP — and contiguous shards in rank order need no restore permutation. The divisibility check was the wrong constraint: ownership by first token absorbs any sample length, so `L % R == 0` is not required, while `R <= L` is. |
| model-level validation belongs in the DSv4 forward, which raises if CP is enabled and the model is not DSv4 | positive allow-list `check_context_parallel_supported` in `veomni/models/auto.py`, called from `build_foundation_model`, which the omni path also calls — once per sub-module, with that sub-module's own config | The design's own mitigation is self-contradictory: a forward that never runs cannot observe that the model is not DSv4. An allow-list also fails safe as CP spreads — a newly ported model has to be added deliberately instead of the gate having to be remembered. Per-sub-module calls matter because the encoder/decoder branch constructs `SeedOmniModel._from_config` directly and the outer `SeedOmniConfig` model type says nothing about what runs the collectives. |
| the plan scoped the CP admission gate to `parallel_state.py` | both `parallel_state.py` and `AcceleratorConfig._resolve_topology` | The arguments layer held its own model-agnostic `assert cp_size == 1`, which aborts on the CLI and trainer path before `ParallelState` is ever constructed. Two gates now carry the same hybrid-only refusal. |
| no `shard_packed_compression_metadata` | new helper in `packed_utils.py` | Only the module holding the hidden states knows they are one shard, so only it can shard the global metadata — and per-window and per-query arrays shard differently (see the decision on index spaces above). |
| "there is no sequence-length divisibility constraint beyond `S % cp_size == 0`" | `cp_size <= S / max(compress_rates)` | Found during implementation, from the halo geometry. See "Sequence length and compression rate". |

## Open questions

**The two gather implementations disagree about their gradient buffer.**
`_Gather.backward` all-reduces in place after a `.contiguous()`, which is a
no-op when the gradient is already contiguous and therefore leaves the reduce
writing into autograd's buffer; `_GatherConcatSP.backward` clones first. Neither
documents the precondition. `_Gather` is the one that is wrong, but the right
fix is a single non-in-place reduce in both, not a second clone — that is a
`sequence_parallel` concern and out of scope for CP. The history is worth
knowing before touching it: the `.contiguous()` was added, then reverted on the
rationale that no production caller reduces a gathered output with a bare
`.sum()`, which covered the stride-0 broadcast case but not the one that
actually bites — `torch.cat([kv, compressed_kv], dim=2)` handing backward a
narrowed view, contiguous only while `[B, 1, S, D]` collapses at batch 1 — and
was then restored with a batch-2 parity case pinning it. The line buys
contiguity and nothing else.

**The indexer's zero-window path cannot be backward-tested, and that is
temporary.** The indexer emits integer top-k indices and nothing downstream of
its compression is differentiable — its `index_scores` are discarded by the
`topk` on both the eager and TileLang paths, and its parameters receive no
gradient at all today. So its `exchange_compressor_halos` and
`all_gather_compressed_rows` have no backward on *any* rank: uniformly
unreachable from the loss, which is the one situation that cannot deadlock.
DeepSeek's own design gives the indexer an auxiliary loss. The day that lands,
those collectives acquire a backward and the indexer's zero-window path becomes
hang-class overnight; whoever adds the loss must add the backward coverage in
the same change. The unit test on `empty_compressed_rows` is what still holds in
the meantime.

**The omni wrapper under CP is untested.** The gate reads the foundation's model
type, so an omni model with a DeepSeek-V4 foundation is admitted. Whether the
wrapper's encoders — which process their own modality tokens around the
foundation — are CP-safe is unverified, and nothing exercises it. DeepSeek-V4
has no presence under `veomni/models/seed_omni/` today, so this is latent rather
than live.

**Performance is unmeasured.** The projected win over Ulysses (about 465 GB per
step down to about 7.4 GB, with both all-to-alls disappearing) is an estimate
from a production trace and kernel microbenchmarks, not a measurement of this
implementation. Three known costs are accepted by design in milestone 1: the
indexer is load-imbalanced, because every rank scores its queries against the
full global compressed array; a CSA layer pays two extra collectives, since the
compressor and its indexer compress the same windows at different head
dimensions and cannot share a result; and `exchange_compressor_halos`
all-gathers from every rank when only two neighbours are ever read.
`tests/models/transformers/test_model_forward_no_implicit_sync.py` does not exercise CP, so
the CP paths have no automated guard against device-to-host syncs.
