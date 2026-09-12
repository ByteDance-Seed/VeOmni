# Long-context dynamic-batching cold start

## Result

`TextBatchingStrategy` waits for both the token budget and
`dyn_bsz_buffer_size` candidate samples before it packs a micro-batch. At 512K
and 1M, the default 200-candidate gate can remain closed long after enough
tokens have arrived.

This PR adds an opt-in `context_aware` policy for the measured long-context
text path:

| `max_seq_len` | Per-DP-rank minimum candidates |
| --- | ---: |
| 512K | 24 |
| 1M | 24 |

The default stays `fixed / 200`, so existing jobs, including 128K jobs, do not
change. The opt-in policy requires main-process batching, total-token counting,
text data, and MBS1, but it does not restrict DP size.

The value 24 is a minimum gate, not a capacity. Each DP rank continues fetching
when 24 candidates do not yet contain one physical token window. Once both the
candidate-count and token-count conditions are met, the existing ordered greedy
packer emits the batch. The policy deliberately does not wait for 200 candidates
only to chase the final 1-2% of packing utilization.

## DP1 cold-start matrix

The CPU-only probe retained the production stream, transform, dynamic batching,
packing, collator, and SP slice, but did not build a model or run an accelerator
collective.

| Token budget | Buffer | Rank p50 | Rank max | Ready skew | Source fill | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 512K | 8 | 173.46 s median | 192.80 s | <=1.47 s | 99.7480% | Pass |
| 512K | 16 | 349.44 s | 349.87 s | 0.63 s | 99.7480% | Pass |
| 512K | 24 | 576.75 s | 577.66 s | 2.96 s | 99.7480% | Pass |
| 512K | 200 | 5861.24 s | 5869.38 s | 8.57 s | 99.7480% | Pass |
| 1M | 8 | 374.62 s | 381.90 s | 8.36 s | 92.6719% | Underfilled |
| 1M | 16 | 393.97 s | 399.17 s | 6.50 s | 94.3304% | Underfilled |
| 1M | 24 | 589.63 s | 590.88 s | 7.02 s | 99.6184% | Pass |
| 1M | 200 | 5063.80 s | 5067.92 s | 24.13 s | 99.6184% | Pass |

The fixed-200 rows are natural completions, not projections. Ready skew is the
latest minus earliest server-recorded `data_ready` timestamp. Source fill is
selected source tokens divided by the physical token window; it measures
packing utilization, not sample loss.

For the original 512K seed, fixed-8 and context-aware 200-to-8 matched on the
candidate and selected sequences, document boundaries, packed tensors, SP-local
tensors, and source-token counts across 192 rank records. This identity result
is limited to that deterministic first micro-batch.

## DP scaling and shuffle-seed evidence

The scaling probe ran the same CPU data path on an isolated stable Ascend 910B2
host. One process represented each independent DP stream. With fixed U16 x CP2
geometry, DP1/2/4/8 correspond to 32/64/128/256 total ranks; EP does not change
the data-parallel stream count.

Global source fill is:

```text
sum(selected_source_tokens) / (dp_size * max_seq_len)
```

Candidate and selected hashes were distinct across DP ranks, and every rank
satisfied `0 < selected_source_tokens <= max_seq_len`.

### 1M, minimum buffer 24

| Shuffle seed | DP2 | DP4 | DP8 |
| --- | ---: | ---: | ---: |
| 42 | 99.5129% | 98.6507% | 98.7175% |
| 43 | 98.5127% | 99.4050% | 98.7576% |

All six points pass the 98.5% global source-fill gate. Together with the DP1
result above, they support using the same per-rank minimum at DP1/2/4/8. DP
changes the number and tails of independent data streams; it does not multiply
the per-rank candidate gate.

### 512K, shuffle seed 43

| DP | Buffer | Ready p50 | Global source fill | Verdict |
| ---: | ---: | ---: | ---: | --- |
| 1 | 8 | 191.61 s | 93.1940% | Underfilled |
| 1 | 24 | 478.76 s | 98.7411% | Pass |
| 2 | 8 | 244.67 s | 95.5143% | Underfilled |
| 2 | 16 | 392.61 s | 98.2879% | Underfilled |
| 2 | 24 | 567.99 s | 98.2879% | Underfilled |
| 4 | 16 | 351.15 s | 97.9949% | Underfilled |
| 4 | 24 | 547.36 s | 98.4475% | Underfilled |
| 4 | 32 | 696.47 s | 98.4475% | Underfilled |
| 8 | 32 | 667.50 s | 98.6668% | Pass |

The DP2 buffers 16 and 24 produced identical selected hashes and token counts;
the same was true for DP4 buffers 24 and 32. Waiting for more candidates can
therefore increase cold-start time without changing greedy packing output.

The original 512K DP1/buffer-8 result was 99.7480%, while seed 43 produced
93.1940%. Buffer 24 raised the seed-43 DP1/2/4 global fill to
98.7411%/98.2879%/98.4475%. The DP8 point used buffer 32, so DP8/buffer-24 is
not claimed as a direct measurement. The policy nevertheless keeps a single
per-rank minimum instead of encoding an unsupported `buffer * DP` formula.

## Training trade-off

Source fill is packing utilization, not a data-loss check. Samples not selected
by the greedy packer remain in the buffer for later batches, and padding is
excluded by the attention/loss masks. A lower fill therefore does not corrupt
the batch, but at a fixed optimizer-step count it does expose the model to fewer
source and loss-contributing tokens.

The measured policy accepts this trade-off: roughly 1-2% lower global fill on
some 512K streams is preferable to delaying the first forward by tens of
minutes. Under a fixed wall-clock budget, the earlier training start covers
that small per-step utilization gap for the short long-context runs targeted by
this policy. Long runs that require exact token-budget parity should stop by
accumulated source/loss tokens rather than optimizer-step count.

## Configuration

```yaml
data:
  max_seq_len: 1048576
  dyn_bsz_buffer_policy: context_aware

train:
  micro_batch_size: 1
  dyn_bsz: true
  dyn_bsz_runtime: main
  dyn_bsz_count_mode: total
```

`build_dataloader` resolves the policy before registry dispatch so a registered
external builder receives the concrete per-rank minimum of 24.

The experiments cover first-micro-batch preparation only. Whole-step
pre-materialization and periodic forward-only stalls are separate problems.
