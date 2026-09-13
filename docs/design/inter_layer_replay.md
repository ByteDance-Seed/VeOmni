# Qwen3-MoE inter-layer replay

This document describes the NPU piercing implementation for overlapping Qwen3-MoE
layer backward with the previous layer's activation replay under FSDP2.

## Scope

- Model: truncated 8-layer Qwen3-30B-A3B.
- Parallelism: 8 NPU ranks, EP size 8, FSDP2, gradient checkpointing enabled.
- Top scheduled pair: `B7` and `F'6`.
- MoE replay phases: attention, dispatch, experts, and combine.
- The rolling window supports adjacent boundaries down to a configured bottom layer
  and requires `ep_size` to divide `world_size`.

The scheduler keeps the normal `F'7` layer replay, prepares layer 6 attention early,
and advances layer 6 through the following state machine while `B7` executes:

```text
ATTN_READY -> DISPATCH_PENDING -> EXPERT_READY -> COMBINE_PENDING -> GRAPH_READY
```

The backward communication boundaries drive the transitions:

```text
COMBINE(B7) -> DISPATCH(F'6) -> DISPATCH(B7) -> COMBINE(F'6)
```

Layer 6 consumes the normal FSDP backward unshard before its early replay. It does
not issue an additional model-parameter all-gather. The expert FSDP mesh has size 1
in this piercing configuration.

## Cached backward

The second-stage implementation makes the overlapped work authoritative without
keeping one autograd graph alive across `B7`. It cuts the graph at two activation
boundaries:

1. `router_input_proxy` records the router derivative with respect to the MoE input.
2. `expert_input_proxy` records the expert derivative with respect to dispatched tokens.

Attention and the post-attention norm are first evaluated without an autograd graph.
Dispatch uses the detached MoE input, while routing and expert computation use their
respective leaf proxies. After `B7`, layer 6 backward proceeds as follows:

```text
wait combine(F'6)
  -> backward cached router/expert graph
  -> inverse expert dispatch gradient
  -> add router-input gradient
  -> recompute ATTN + post-attention norm
  -> backward ATTN graph
```

The inverse dispatch operation reverses the expert chunk permutation, performs the
reverse AllToAll, and applies token unpermutation with unit routing weights to sum
the top-k expert paths. This reconstructs the complete MoE-input gradient while
preserving the cached gate and expert parameter gradients. It removes the stage-one
full-layer repair replay and its duplicate MoE forward.

## Configuration

```yaml
train:
  gradient_checkpointing:
    enable: true
  inter_layer_replay:
    enable: true
    current_layer: 7
    window_size: 7
    memory_budget_gb: 0
    memory_reserve_gb: 0
    memory_safety_factor: 1.1
    memory_retry_steps: 4
    strict: true
```

`strict` controls behavior only when the replay frame is unavailable before ILP
scheduling starts. It no longer selects a speculative full-layer repair path.
`window_size` is the number of adjacent overlap boundaries. For example, `2`
enables `B7/F'6` followed by `B6/F'5`; `7` covers all boundaries of the truncated
8-layer model.

The memory limits are disabled when both values are zero. When enabled, the
scheduler uses the maximum allocated memory, minimum free memory, and maximum
observed replay estimate across ranks. `memory_retry_steps` controls how often a
fully paused scheduler reevaluates the budget at the top-layer boundary.

The scheduler constructs adjacent layer pairs only. It rejects ChunkMBS,
non-FSDP2 execution, pipeline/tensor/context/Ulysses parallelism, and
`torch.compile`. The current NPU path also requires the `fused_npu` MoE kernel.

Use `qwen3_30b_ilp_dev.sh` for the 8-rank piercing run. The trimmed eight-layer
script enables all seven backward/replay boundaries by default. Extra command-line
options are appended after the defaults, so they can override the window, sequence
length, steps, and profiling settings.

## Validation

### Stage one baseline

The functional run completed forward, backward, optimizer update, DCP save, and HF
checkpoint export. The initial two-step profiler run used:

```bash
./qwen3_30b_ilp_dev.sh \
  --data.max_seq_len 512 \
  --train.global_batch_size 8 \
  --train.max_steps 2 \
  --train.num_train_epochs 1 \
  --train.checkpoint.save_epochs 0 \
  --train.checkpoint.save_hf_weights false \
  --train.profile.enable true \
  --train.profile.start_step 1 \
  --train.profile.end_step 2 \
  --train.profile.trace_dir ./trace/ilp_stage1_valid
```

Rank 0 trace observations for the final strict-mode profiled step:

| Range | Relative start (ms) | Duration (ms) |
| --- | ---: | ---: |
| `ilp::Fprime6::fsdp_prefetch_launch` | 0.000 | 1.788 |
| `ilp::Fprime7` | 2.486 | 17.352 |
| `ilp::Fprime6::attention` | 20.156 | 3.374 |
| `ilp::B7` | 23.659 | 162.310 |
| `ilp::Fprime6::dispatch_launch` | 27.448 | 2.485 |
| `ilp::Fprime6::experts` | 37.541 | 2.998 |
| `ilp::Fprime6::combine_launch` | 43.204 | 2.344 |
| `ilp::Fprime6::speculative_drain` | 186.121 | 0.143 |
| `ilp::B6::strict_replay` | 186.272 | 295.017 |

The three layer 6 speculative phases are nested inside the `B7` host range. The layer 6
FSDP trace contains exactly two parameter all-gathers: one for the original forward
and one for backward replay. There is no third ILP-specific all-gather.

For numerical validation, a full-sequence one-step run with ILP disabled and the
strict ILP run both produced `loss=12.63` and `grad_norm=49.98` on the same batch.
The experimental `strict: false` path produced the same loss but `grad_norm=52.07`.
Per-parameter diagnostics isolated that difference to layer 6 MoE-input/parent-FSDP
gradients; layer 6 gate/expert and all layer 7 gradient norms matched the baseline.

These timings proved functional scheduling and communication launch overlap only.
The strict repair was removed in stage two.

### Stage two cached backward

A two-rank NPU differential test compares the cached path with the ordinary autograd
path in the same process. Output, hidden-state gradient, routing gradient, down-proj
gradient, and gate-up-proj gradient all matched exactly (`rtol=0`, `atol=0`). The
test is located at `tests/ops/test_npu_ep_cached_backward.py`.

An 8-rank full-sequence one-step run completed with `loss=12.63` and
`grad_norm=49.98`, matching the stage-one baseline. A short-sequence rerun showed
small NPU run-to-run grad-norm variation (`56.89` cached versus `56.92` baseline),
while the same-process differential test remained bit-exact.

Rank 0 trace observations for the final cached-backward profiled step:

| Range | Duration (ms) |
| --- | ---: |
| `ilp::Fprime6::fsdp_prefetch_launch` | 1.962 |
| `ilp::Fprime7` | 38.402 |
| `ilp::Fprime6::attention` | 3.162 |
| `ilp::B7` | 165.750 |
| `ilp::Fprime6::dispatch_launch` | 3.049 |
| `ilp::Fprime6::experts` | 3.040 |
| `ilp::Fprime6::combine_launch` | 2.278 |
| `ilp::B6` | 11.757 |
| `ilp::B6::cached_moe` | 4.403 |
| `ilp::B6::dispatch_input_backward` | 1.059 |
| `ilp::B6::attention_resume` | 1.595 |
| `ilp::B6::attention_backward` | 4.197 |

The replay dispatch, experts, and combine launches are nested in the `B7` host
range. Compared with the synchronized prototype, using the collective handle for
the reverse AllToAll reduced `B6` from `346.593 ms` to `11.757 ms`; the synchronized
version waited for unrelated device-stream backlog. The layer still has only its
normal original-forward and backward-replay parameter all-gathers.

These are piercing-run observations, not a final throughput claim. Stable speedup
measurement needs exclusive devices, warmup, and at least 20 identical steps.

### Stage three rolling window

The controller generalizes the single pair into a rolling state machine. When a
cached layer enters backward, it prepares the next lower layer and uses the cached
combine, expert-input, and manual inverse-dispatch boundaries to advance that next
replay. The callback is stored explicitly in the async AllToAll autograd context and
expert-input hook, so it does not depend on `ContextVar` propagation into an autograd
worker thread.

Dispatch split metadata is captured from the original checkpoint forward and stored
in the replay frame. Reusing it avoids token-count synchronization after the device
backward queue has started. The rolling path waits for the dispatch collective Work
but does not perform a full device-stream synchronization before enqueueing expert
compute.

Final `window_size=2` rank 0 trace observations:

| Range | Duration (ms) |
| --- | ---: |
| `ilp::B7` | 142.033 |
| `ilp::Fprime6::dispatch_launch` | 0.317 |
| `ilp::Fprime6::experts` | 1.922 |
| `ilp::Fprime6::combine_launch` | 1.324 |
| `ilp::Fprime5::attention` | 1.747 |
| `ilp::B6` | 13.582 |
| `ilp::Fprime5::dispatch_launch` | 0.270 |
| `ilp::Fprime5::experts` | 2.536 |
| `ilp::Fprime5::combine_launch` | 1.620 |
| `ilp::B5` | 10.017 |

Layer 5, 6, and 7 each contain exactly two `FSDP::all_gather` events. The trace has
no replay-time `dispatch_prepare` range. A two-rank formal pytest completed with all
cached output and gradient comparisons bit-exact and with the expected
`combine -> experts` callback order.

Functional 8-rank results:

| Window | Sequence | Loss | Grad norm | Peak memory |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 512 | 12.52 | 56.84 | 10.65 GB |
| 2 | 16384 | 12.63 | 49.97 | 25.14 GB |
| 7 | 512 | 12.52 | 56.85 | 11.12 GB |

The long-sequence window result exposes the need for an automatic memory budget:
the equivalent single-pair run peaked at 19.34 GB. Window depth must therefore be
limited or paused using observed activation size and available device memory.

### Stage four pausable memory scheduler

The memory scheduler measures the allocation delta of a completed replay graph and
uses its high-water value as the next replay estimate. At the top backward boundary
it applies the configured safety factor and computes a rank-consistent number of
replay slots for the whole training step:

```text
budget_slots  = floor((budget - max_allocated) / estimated_replay_bytes)
reserve_slots = floor((min_free - reserve) / estimated_replay_bytes)
step_slots    = min(window_size, budget_slots, reserve_slots)
```

Disabled constraints do not limit `step_slots`. Before an estimate exists, at most
one slot is opened to bootstrap measurement. Lower boundaries only consume the
precomputed slots and do not run additional memory collectives.

When no slot remains, the next frame stays `EMPTY`. Its own backward uses ordinary
checkpoint replay and may retry scheduling only when a later top-layer memory probe
permits it. A fully paused scheduler probes every `memory_retry_steps` training
steps; intervening boundaries emit `pause_deferred` and perform no consensus.

Memory metadata is reduced through a dedicated Gloo CPU process group. This keeps
the control-plane collective out of the HCCL/NPU computation stream while still
requiring every rank to use the same slot count. In the final trace, the top memory
check took `4.610 ms`, compared with `62.134 ms` using a layer-boundary HCCL
AllReduce. Deferred checks took `0.013-0.029 ms`.

With a `5.5 GB` budget on the 512-token run, the bootstrap step selected one replay
slot at `4.41 GB` allocated memory. After optimizer state allocation, the next top
probe observed `9.76 GB` and selected zero slots. The run completed normally with
`loss=12.52`, `grad_norm=56.87`, and a one-step peak of `10.58 GB`. The full policy
and NPU differential suite completed with `5 passed`; cached output and gradient
comparisons remained bit-exact.

Twenty-step steady-state observations were collected from 22-step runs by excluding
the first two warmup steps. The elapsed values are coarse because tqdm reports whole
seconds, so the final smoothed iteration rate is included as a second signal.

| Mode | Last-20 estimate | Final rate | Peak memory |
| --- | ---: | ---: | ---: |
| Native checkpoint baseline | 0.25 s/step | 4.10 it/s | 13.73 GB |
| Static `window_size=7` | 0.35 s/step | 2.82 it/s | 28.29 GB |
| Adaptive `5.5 GB`, final policy | 0.35 s/step | 3.44 it/s | 14.71 GB |

The adaptive policy controls memory but does not provide a throughput win at 512
tokens. The static window is about 40 percent slower than baseline in this run; the
adaptive path is about 19 percent slower by final iteration rate. The main remaining
cost is the custom uncached checkpoint path used by paused layers, plus duplicate
attention replay. Performance claims require longer-sequence and repeated-run data.

### Stage five native checkpoint fallback

The memory decision now also controls which checkpoint implementation is installed
for the next training step. The current layer remains on the ILP checkpoint function
as a control anchor. Replay layers covered by the planned slot count use ILP; all
lower paused layers call the original PyTorch non-reentrant checkpoint callable that
was captured before the ILP wrapper was installed. A native layer is recorded for
the current step so the strict scheduler can distinguish an intentional missing ILP
frame from a broken registration.

The slot decision is applied one step later because checkpoint graphs are selected
during forward while the memory probe runs at the top backward boundary. The first
step still bootstraps at most one ILP slot. A later probe can increase the planned
slot count and automatically move those layers back to ILP on the following forward.
Reentrant checkpointing is now rejected explicitly when inter-layer replay is
enabled.

An 8-rank four-step integration run with a `5.5 GB` budget scheduled one bootstrap
slot at `4.41 GB`, then paused at `9.73 GB` after optimizer state allocation. It
completed with `loss=12.52` and `grad_norm=56.87` on the first step. A rank-0 trace
covering the transition contained two native checkpoint markers for each of layers
0 through 5 and one for layer 6. It contained only one custom `Fprime6/B6` pair but
two `Fprime7/B7` pairs, confirming that layer 6 moved from ILP to native checkpoint
while layer 7 remained the control anchor.

Long-sequence performance used the truncated 8-layer model, sequence length 16384,
global batch size 8, and 8 NPU ranks. Each mode ran independently three times for
eight steps. The first two steps were excluded and the elapsed time over the final
six steps was averaged. The tqdm smoothed value is included as a second signal.

| Mode | Steady time, mean of 3 | Final smoothed time | Peak memory | Relative steady time |
| --- | ---: | ---: | ---: | ---: |
| Native checkpoint baseline | 1.333 s/step | 1.860 s/step | 15.61 GB | baseline |
| Paused `window_size=7`, `5.5 GB` budget | 1.333 s/step | 1.863 s/step | 19.98 GB | +0.0% |
| Static `window_size=1` | 1.722 s/step | 2.297 s/step | 42.43 GB | +29.2% |

All three repetitions of the paused mode had the same six-step elapsed estimate as
baseline; their smoothed times differed by about 0.2 percent. The reported paused
peak includes the first bootstrap step and remains a process high-water mark even
after later layers switch to native checkpoint. The static single-boundary ILP path
was consistently slower and used substantially more memory. At this model depth and
kernel implementation, duplicate attention replay and retained replay graphs cost
more than the exposed communication they hide. The next optimization target should
therefore be graph lifetime and duplicate attention compute, not a wider window.

### Stage six single-replay compute parity

The early replay attention now runs with autograd enabled and retains its
`attention_hidden` and post-attention-normalization graph until the replay layer
enters backward. After cached MoE backward reconstructs the MoE-input gradient, the
scheduler applies the residual-output and MoE-input gradients directly to this saved
attention graph. The previous `attention_resume` forward has been removed.

Consequently, an active replay layer has the same theoretical forward operator count
as native checkpointing:

```text
native: original (ATTN + MoE) + checkpoint replay (ATTN + MoE)
ILP:    original (ATTN + MoE) + early replay      (ATTN + MoE)
```

After attention backward is enqueued, the frame immediately releases attention,
router, expert, dispatch, and combine graph references. A toy-layer regression test
checks that prepare plus backward calls `forward_attention` exactly once and still
produces input and parameter gradients.

The 8-rank functional comparison at sequence length 512 produced
`loss=12.52/grad_norm=56.94` for baseline and `12.52/56.93` for ILP. Peak memory was
`13.65 GB` and `13.82 GB`, respectively. In the rank-0 trace, each of the two sampled
steps contained one `Fprime6::attention` and one `B6::attention_backward`; there were
no `B6::attention_resume` events. `B6` host duration was `6.2-9.4 ms`.

The 16384-token, global-batch-8 experiment repeated the active `window_size=1` mode
three times using the same eight-step protocol as stage five:

| Mode | Steady time, mean of 3 | Final smoothed time | Peak memory | Relative to baseline |
| --- | ---: | ---: | ---: | ---: |
| Native checkpoint baseline | 1.333 s/step | 1.860 s/step | 15.61 GB | baseline |
| Stage-five ILP | 1.722 s/step | 2.297 s/step | 42.43 GB | +29.2% |
| Stage-six ILP | 1.611 s/step | 2.053 s/step | 31.18 GB | +20.8% |

Removing duplicate attention improved ILP steady time by about 6.5 percent and its
smoothed time by about 10.6 percent. Immediate graph release reduced peak memory by
about 26.5 percent. The remaining slowdown is runtime overhead rather than excess
model FLOPs: the attention graph lives across the upper-layer backward, and cached
MoE uses manually segmented autograd, collective, stream, and FSDP scheduling.

### Stage seven dual-stream fork-join

The rolling scheduler now executes each adjacent boundary as an explicit fork-join.
After the current layer replay has completed, the autograd worker starts the current
layer backward while a persistent sidecar worker starts the previous layer replay:

```text
F'7
  -> fork(B7 on stream A, F'6 on stream B)
  -> join(B7, F'6)
  -> fork(B6 on stream B, F'5 on stream A)
  -> join(B6, F'5)
  -> ...
```

Execution domains alternate by layer distance. A replay graph built on stream B is
also consumed by that layer's backward on stream B, so the next lower replay must
switch back to stream A. Each domain has a separate EP/HCCL process group. This
prevents concurrent backward and replay AllToAll operations from imposing
incompatible collective ordering on one communicator.

Before a fork, the replay stream waits for the current replay stream and launches
the lower layer's FSDP unshard in its own execution domain. A host barrier releases
the autograd and sidecar branches together. Each branch records an NPU event when
its device work has been enqueued; both streams wait on the opposite event before
the controller advances to the next pair. The join is therefore device-side and
does not introduce a global NPU synchronization.

The `window_size=2` profiler run at sequence length 512 completed four steps without
deadlock. Rank 0 hardware events showed the intended ping-pong schedule:

| Pair | Backward stream | Replay stream | Simultaneous compute kernels |
| --- | ---: | ---: | ---: |
| `B7 + F'6` | 46 | 41 | 342 us |
| `B6 + F'5` | 41 | 46 | 493 us |

Across the two profiled pairs, CANN reported 4.539 ms of communication, of which
1.111 ms overlapped computation. The PyTorch host trace contains consecutive
`fork_join::B7::Fprime6` and `fork_join::B6::Fprime5` ranges; sidecar host ranges
are not emitted by this profiler build, so hardware stream events are the
authoritative concurrency evidence.

Functional 8-rank, one-step runs at sequence length 512 produced `loss=12.52` and
`grad_norm=56.87` for `window_size=1`, and `loss=12.52` and `grad_norm=56.86` for
`window_size=2`. The targeted distributed test suite completed with `9 passed`.

The exclusive 16384-token, global-batch-8 experiment repeated `window_size=2` three
times for eight steps. Its final tqdm smoothed time averaged 2.173 s/step and peak
memory averaged 44.90 GB. Under the same protocol, the native checkpoint baseline
was 1.860 s/step and 15.61 GB. The fork-join prototype is therefore 16.8 percent
slower by the smoothed rate and uses 29.29 GB more peak memory. This stage proves
the requested device concurrency and join ordering, but it is not yet a throughput
optimization.

### Stage eight native top-layer autograd

The current layer no longer uses `_InterLayerCheckpointFunction`. Its wrapper calls
the original VeOmni non-reentrant checkpoint function and only composes the native
`context_fn`. When the native recompute context exits after `F'7`, the controller
starts `F'6` on the sidecar thread, alternate NPU stream, and alternate EP/HCCL
process group. `B7` remains entirely inside the original autograd and FSDP hook
path. The sidecar joins when autograd reaches the custom layer 6 checkpoint node.

PyTorch non-reentrant checkpoint can terminate replay with its internal
`_StopRecomputationError` after all required tensors have been reconstructed. The
composed context treats that exception as successful replay completion, while other
exceptions do not launch the sidecar.

Two stream-ownership fixes are required for correctness:

1. VeOmni normally registers layer 6 modules as native FSDP backward-prefetch
   targets of layer 7. ILP clears that list because the sidecar is the sole owner of
   the layer 6 unshard. Leaving both enabled caused concurrent FSDP state mutation,
   memory growth to 59 GB, and eventual OOM at sequence length 16384.
2. Layer 6 cached backward is enqueued on the replay stream. Before its custom
   Function returns input gradients to native autograd, it records a completion
   event, makes the autograd stream wait on that event, and records the returned
   gradients on the autograd stream. Without this handoff, the 16384-token run
   produced a corrupted first-step gradient norm of about 257000; the corrected
   value is 50.55.

The final rank 0 profiler sampled two steps. Both contained
`native_checkpoint::F7`, `Fprime6::fsdp_prefetch_launch`, and the native join marker,
while custom `Fprime7` and `B7` markers were absent. B7 compute used stream 46 and
F'6 used stream 41, with 319 us of simultaneous compute kernels in the measured
pair. Main-autograd and sidecar AllToAll operations were submitted from different
host threads and executed on HCCL hardware streams 5 and 187, respectively.

The 512-token smoke run completed with `loss=12.52`, `grad_norm=56.84`, and
10.52 GB peak memory. A 16384-token smoke run completed with `loss=12.71`,
`grad_norm=50.55`, and 12.41 GB peak memory. The targeted CPU and NPU differential
suite completed with `10 passed`.

The exclusive 16384-token experiment repeated each eight-step run three times:

| Mode | Final smoothed time | Peak memory | Relative to baseline |
| --- | ---: | ---: | ---: |
| Native checkpoint baseline | 1.860 s/step | 15.61 GB | baseline |
| Stage-six custom B7 | 2.053 s/step | 31.18 GB | +10.4% |
| Stage-eight native B7 | 1.973 s/step | 30.18 GB | +6.1% |

Restoring native B7 improved the ILP smoothed time by 3.9 percent and reduced peak
memory by about 1 GB. The remaining overhead is now concentrated in the layer 6
segmented replay graph, Python sidecar enqueue, alternate communicator, and the
cross-stream gradient handoff rather than the B7 autograd implementation.

An equivalent 512-token baseline profile attributes the remaining gap. The second
steady profiler step was 312.63 ms for baseline and 330.47 ms for ILP, a 17.84 ms
increase that closely matches the long-run relative slowdown.

| Trace metric | Baseline | Native-B7 ILP | Delta |
| --- | ---: | ---: | ---: |
| Device compute span | 107.64 ms | 108.03 ms | +0.39 ms |
| AllToAll kernel time, 48 calls | 17.81 ms | 18.08 ms | +0.28 ms |
| Event-wait time | 90.68 ms | 98.18 ms | +7.50 ms |
| Device free gaps | 157.62 ms | 168.07 ms | +10.45 ms |
| Communication not overlapped | 47.25 ms | 54.07 ms | +6.82 ms |
| Host autograd evaluate ranges | 150.32 ms | 159.51 ms | +9.18 ms |

The operator counts and aggregate compute time are effectively unchanged. The
319 us of simultaneous B7/F'6 kernels is too small to offset the event waits and
enqueue gaps. The layer 6 cached path also divides backward into cached MoE,
inverse dispatch, and attention backward ranges totaling 11.62 ms of host time.
The independent communicator prevents collective-order deadlock but does not
improve the communication-hidden ratio in this trace: it remains about 10 percent.

After moving the current layer to native autograd, the unreachable custom-current
backward branch and the out-of-window fallback were removed. The uncached replay
path, alternating execution domains, and lower-layer fork-join remain because they
are used by memory pause/resume and `window_size > 1`.

### Stage nine phased replay and native checkpoint cache

The `window_size=1` piercing path now keeps layer 6 on a real PyTorch
non-reentrant checkpoint. The controller exposes that checkpoint's holder cache,
runs the split `F'6` sidecar under PyTorch's recomputation saved-tensor hook, and
marks the cache complete for the active autograd graph task. When B6 starts, native
autograd consumes those cached tensors directly. It no longer enters
`_InterLayerCheckpointFunction.backward` or invokes a nested autograd engine.
Larger replay windows retain the previous custom path until the same cache handoff
is extended to rolling layers.

Native B7 MoE backward captures phase callbacks in the forward autograd nodes, so
the callback remains available on the autograd worker thread. After B7 expert
backward kernels are enqueued, the main stream records an event. F'6 launches
attention and dispatch first, then its expert stage waits for that event. The
sidecar continues to use the alternate EP communicator.

The 512-token profile verifies the intended accounting and removes the segmented
B6 host range:

| Metric | Baseline | Stage eight | Stage nine native cache |
| --- | ---: | ---: | ---: |
| Steady profiler step | 312.632 ms | 330.474 ms | 311.749 ms |
| Device tasks | 2177 | 2168 | 2176 |
| Aggregate compute time | 115.991 ms | 116.211 ms | 116.148 ms |
| GroupedMatmul calls | 64 | 64 | 64 |
| Cross-stream GroupedMatmul overlap | 0 | 0 | 0 |
| Custom B6 host range | none | 10.797 ms | none |

The equal GroupedMatmul count is evidence that F'6 is not executed a second time.
The native checkpoint early-stop protocol divides the calls between the sidecar
and main stream, but their sum remains identical to baseline. The targeted test
suite completed with `11 passed`; the 8-rank smoke run completed with
`loss=12.52` and `grad_norm=57.00`.

The 16384-token result does not preserve the short-profile improvement. Three
exclusive eight-step runs produced final smoothed rates of 2.12, 2.13, and
2.13 s/step, averaging 2.127 s/step with a 0.005 s standard deviation. This is
14.3 percent slower than the 1.860 s/step baseline and 7.8 percent slower than
stage eight. The expert-level event removes possible GroupedMatmul contention but
serializes too much of the long-sequence critical path. A profitable scheduler
must move the boundary into individual grouped-matmul backward calls, or rely on a
backend resource scheduler, instead of waiting for the whole B7 expert phase.

### Stage ten: readiness-aware GMM ownership

Stage ten moves the ownership boundary into each `GmmFunction.backward` call.
The B7 forward captures a scheduler binding in its autograd context, while the
F'6 sidecar enters a replay-forward binding. Both paths publish an NPU event after
each grouped matmul and wait on the event that owns the next ticket. For Qwen3-MoE
the strict order is `B0 -> F0 -> B1 -> F1`, where each backward ticket covers both
the input-gradient and weight-gradient grouped matmuls launched by one
`GmmFunction.backward` call.

Strict alternation exposed a readiness problem rather than a compute conflict.
At 512 tokens it increased the profiler step to 323.962 ms because B1 waited
7.761 ms for F'6 dispatch and permutation to make F0 runnable. The scheduler now
uses a readiness handshake: it interleaves when F0 has reached its GMM gate by the
time B1 requests ownership, and otherwise permits `B0 -> B1 -> F0 -> F1`. The
fallback reduced the same profiler step to 312.854 ms and avoids putting a host
wait on the critical path.

The native checkpoint path also captures B6's dispatch plan during the original
forward and reuses it in the sidecar. This removes a duplicate token-count
all-gather from F'6. With that cache enabled, the 512-token profiler step was
301.189 ms. The eight-rank smoke run completed with `loss=12.52` and
`grad_norm=56.88`; the combined targeted suites complete with `12 passed`.

Long-sequence results still show that communication, not GMM arithmetic, is the
dominant remaining limit. Three exclusive 16384-token runs each reported
2.09 s/step. This is 12.4 percent slower than the 1.860 s/step baseline, but
1.7 percent faster than stage nine.

| 16384-token profiler metric | Baseline | Stage ten | Delta |
| --- | ---: | ---: | ---: |
| Compute union | 896.573 ms | 896.110 ms | -0.463 ms |
| Aggregate AllToAll | 395.771 ms | 661.803 ms | +266.032 ms |
| Communication union | 897.247 ms | 1047.896 ms | +150.649 ms |
| No-device-task intervals | 813.708 ms | 951.870 ms | +138.162 ms |
| Final sidecar join | none | 0.204 ms | +0.204 ms |

The alternate communicator contributes 124.070 ms of AllToAll, while primary
AllToAll grows from 395.771 ms to 537.733 ms. A separate process group prevents
collective-order deadlocks, but it does not provide separate HCCL engines or
physical network bandwidth. Concurrent collectives therefore slow both groups.
The unchanged compute union and 0.204 ms final join also show that B6's native
autograd handoff and Python-side completion wait are no longer the primary cost.

### Stage eleven: communication tickets

Stage eleven adds lifecycle callbacks around MoE backward collectives and uses
them to issue four ordered communication tickets:
`B7 combine -> F'6 dispatch -> B7 dispatch -> F'6 combine`. The native
`window_size=1` path keeps separate primary and replay EP communicators, but the
ticket protocol prevents their AllToAll operations from running concurrently.
The replay dispatch starts only after B7 combine backward completes; B7 dispatch
waits for replay dispatch completion; replay combine then waits for B7 dispatch.

Using one EP communicator for both autograd and the sidecar was also evaluated.
It completed ordinary smoke steps, but hung inside HCCL after profiler parsing
introduced rank-dependent host delays. Local thread events cannot make two host
threads globally consistent collective issuers, so that variant is not retained.

The retained two-communicator scheduler passes the deterministic ownership test,
the 2-rank NPU differential test, and the 8-rank smoke run. The combined targeted
suites complete with `13 passed`; the smoke run reports `loss=12.52` and
`grad_norm=56.93`.

At 512 tokens, communication becomes cheaper but the end-to-end step regresses:

| Metric | Stage ten | Stage eleven | Delta |
| --- | ---: | ---: | ---: |
| Steady profiler step | 301.189 ms | 320.723 ms | +19.534 ms |
| Compute union | 107.920 ms | 107.789 ms | -0.131 ms |
| AllToAll aggregate | 20.508 ms | 16.649 ms | -3.859 ms |
| Communication union | 66.228 ms | 53.617 ms | -12.611 ms |
| No-device-task intervals | 132.896 ms | 161.744 ms | +28.848 ms |
| B7 dispatch ticket wait | none | 1.052 ms | +1.052 ms |

Both GMM decisions in the steady step selected `defer`: F'6 dispatch had not
produced runnable expert input when B7 requested the second GMM. Serialization
therefore removes communication contention but creates a larger empty interval
before replay compute. Three exclusive 16384-token runs produced 2.12, 2.10, and
2.13 s/step, averaging 2.117 s/step. This is 1.3 percent slower than stage ten and
13.8 percent slower than the 1.860 s/step baseline. The next optimization must
move replay preparation earlier rather than tighten the communication fence.

### Stage twelve: dispatch-window replay preparation

Stage twelve starts the F'6 sidecar immediately before F'7 launches its forward
dispatch. The launch callback is scoped to the native checkpoint recomputation,
so the original forward is unchanged. F'6 can enqueue attention, routing, and
token permutation while F'7 is in its dispatch and expert phases, but its AllToAll
still waits for the stage-eleven `B7 combine` ticket. Token permutation is split
from `npu_ep_dispatch_async` so it can execute before that ticket without adding
an operator or changing the cached dispatch plan.

Launching at the beginning of F'7 recomputation was evaluated first. It made the
GMM grant ready, but competing F'6 and F'7 attention/MLP kernels delayed F'7 and
increased the steady profiler step to 327.147 ms. Moving the launch to the F'7
dispatch boundary retains the interleave grant with less AI Core contention:

| 512-token metric | Stage ten | Stage eleven | Stage twelve |
| --- | ---: | ---: | ---: |
| Steady profiler step | 301.189 ms | 320.723 ms | 313.568 ms |
| Compute union | 107.920 ms | 107.789 ms | 108.165 ms |
| AllToAll aggregate | 20.508 ms | 16.649 ms | 19.383 ms |
| Communication union | 66.228 ms | 53.617 ms | 66.305 ms |
| No-device-task intervals | 132.896 ms | 161.744 ms | 146.088 ms |
| GMM decision | no marker | defer | interleave |

The trace still contains 64 GroupedMatmul tasks, matching baseline arithmetic.
The targeted CPU and NPU differential suites complete with 13 tests, and the
eight-rank smoke run reports `loss=12.52` and `grad_norm=56.86`.

The short-profile improvement does not extend to the long-sequence case. Three
exclusive 16384-token, eight-step runs produced 2.14, 2.14, and 2.12 s/step,
averaging 2.133 s/step. This is 0.8 percent slower than stage eleven, 2.1 percent
slower than stage ten, and 14.7 percent slower than the 1.860 s/step baseline.
F0 readiness is no longer the blocker. The remaining critical path is dominated
by AI Core contention during early preparation, per-GMM event handoff gaps, and
FSDP all-gather/reduce-scatter traffic that is outside the communication ticket
state machine.

### Stage thirteen: FSDP collective tickets

Stage thirteen installs FSDP2 custom communication wrappers on layers 7 and 6.
The wrappers preserve the original allocator and collective implementation while
publishing lifecycle markers and NPU events for all-gather and reduce-scatter.
If the replay prefetch launches a real all-gather, F'7 dispatch waits on its
completion event; if parameters are already unsharded, the trace records
`F6_all_gather_not_needed` and does not add a cross-stream fence. A later replay
all-gather similarly waits for the B7 reduce-scatter event.

The current `window_size=1` cache path does not launch a layer 6 all-gather at the
sidecar boundary. Its parameters are already available, so an initial variant
that always created a fallback event only added a stream round trip. The retained
implementation records B7 and F6 reduce-scatter calls and avoids that event when
there is no collective to order.

The final 512-token profile reports a 100 percent GMM interleave hit rate (one
interleave, zero defers). The F6 all-gather ticket is not needed; the host wrapper
ranges are 0.223 ms for B7 reduce-scatter and 0.244 ms for F6 reduce-scatter.
Neither range is a device wait. B7 and F6 reduce-scatter operations are already
sequential in the native FSDP order.

| 512-token metric | Stage twelve | Stage thirteen | Delta |
| --- | ---: | ---: | ---: |
| Steady profiler step | 313.568 ms | 316.223 ms | +2.655 ms |
| Compute union | 108.165 ms | 107.904 ms | -0.261 ms |
| Communication union | 66.305 ms | 68.996 ms | +2.691 ms |
| No-device-task intervals | 146.088 ms | 146.778 ms | +0.690 ms |
| AllToAll aggregate | 19.383 ms | 18.088 ms | -1.295 ms |
| All-gather aggregate | 26.635 ms | 29.284 ms | +2.649 ms |
| Reduce-scatter aggregate | 18.364 ms | 18.814 ms | +0.450 ms |

The step delta follows global all-gather variability rather than added compute or
idle time. One discarded profile illustrates the noise: its final root
reduce-scatter took 39.747 ms instead of about 12 ms and increased the step to
336.241 ms. The targeted suites complete with 14 tests, and the eight-rank smoke
run reports `loss=12.52` and `grad_norm=56.99`.

Three exclusive 16384-token runs produced 2.14, 2.13, and 2.12 s/step, averaging
2.130 s/step. This is statistically neutral relative to stage twelve's 2.133
s/step and remains 14.5 percent slower than baseline. The result narrows the next
performance target: the current FSDP collectives are naturally ordered, while
early attention/GMM event gaps and the broader primary-communicator traffic remain
on the critical path.

### Stage fourteen: rolling native checkpoint cache

The native non-reentrant checkpoint cache now covers every replay layer in the
configured window. When B7 consumes the completed F'6 cache, its output hook
launches F'5 with a layer-6 schedule; B6 remains in VeOmni's native autograd and
joins F'5 before continuing. FSDP backward prefetch is disabled for every owner
layer in the replay window because each rolling sidecar owns the corresponding
lower-layer unshard.

With all production wrappers using native checkpoint frames, the unreachable
`_InterLayerCheckpointFunction`, nested autograd backward, cached-gradient
handoff, and legacy lower-layer fork-join paths were removed. Memory pause and
resume now skip a sidecar while leaving the native checkpoint path intact rather
than selecting a second checkpoint implementation.

The targeted CPU and NPU differential suites complete with 14 tests. Eight-rank,
one-step runs at sequence length 512 report `loss=12.52`, `grad_norm=56.94` for
`window_size=1`, and `loss=12.52`, `grad_norm=56.97` for `window_size=2`.

### Stage fifteen: resource-class FIFO ownership

HCCL and GroupedMatmul now use separate FIFO ownership queues. A queue release
records an NPU event on the owner's stream; the next ticket waits for that event
on its own stream before submitting work. The HCCL queue covers MoE AllToAll and
the window's FSDP all-gather/reduce-scatter wrappers. GMM retains the readiness
decision from stage ten, but both the interleave and defer orders are represented
as FIFO reservations rather than separate event arrays.

FSDP ticket roles are resolved from the active backward schedule instead of being
fixed to layer 7 and layer 6. Thus layer 6 is the replay owner during `B7/F'6` and
the current owner during `B6/F'5`; every layer in a rolling window installs the
same wrapper. The bottom layer has no lower sidecar and therefore does not install
MoE/GMM scheduling callbacks during recomputation.

The final 512-token profile shows 46 primary-communicator and two sidecar
AllToAll operations. There are zero cross-group interval overlaps, proving that
the event-backed FIFO removes concurrent HCCL use across the two communicators.
GroupedMatmul remains at 64 tasks, equal to baseline arithmetic.

| 512-token metric | Stage thirteen | Stage fifteen | Delta |
| --- | ---: | ---: | ---: |
| Steady profiler step | 316.223 ms | 305.836 ms | -10.387 ms |
| Compute union | 107.904 ms | 108.045 ms | +0.141 ms |
| Communication union | 68.996 ms | 61.722 ms | -7.274 ms |
| No-device-task intervals | 146.778 ms | 143.800 ms | -2.978 ms |
| AllToAll aggregate | 18.088 ms | 19.068 ms | +0.980 ms |
| All-gather aggregate | 29.284 ms | 19.986 ms | -9.298 ms |
| Reduce-scatter aggregate | 18.814 ms | 20.001 ms | +1.187 ms |

Two earlier repeated profiles measured 310.191 ms and 306.378 ms before the
bottom-layer callback was removed. Communication variability remains visible,
especially in all-gather duration, but every retained trace has no primary/replay
AllToAll overlap. The targeted suites complete with 16 tests. Eight-rank smoke
runs report `loss=12.52`, `grad_norm=57.04` for `window_size=1`, and
`loss=12.52`, `grad_norm=56.86` for `window_size=2`.

The short-profile improvement does not extend to long sequences. Three exclusive
16384-token, eight-step runs produced 2.15, 2.14, and 2.15 s/step, averaging
2.147 s/step with 41.37 GB peak memory. This is 0.8 percent slower than stage
thirteen's 2.130 s/step and 15.4 percent slower than the 1.860 s/step baseline.
At 16K, strict FIFO removes HCCL/HCCL contention but inserts longer replay GMM and
communication ownership intervals into the main-backward critical path. The next
scheduler must preserve per-resource event handoff while avoiding replay admission
when it would stall ready primary work.

### Stage sixteen: deterministic admission policy

Admission is now attached to fixed backward readiness boundaries instead of host
thread arrival order. Replay AllToAll has one deterministic admission window after
`B-combine` completes and before `B-dispatch` requests HCCL ownership. Every rank
therefore retains the same logical order:
`B-combine -> F'-dispatch -> B-dispatch -> F'-combine`. Once a ticket is admitted,
the device-event FIFO remains non-preemptive.

GMM uses a narrower readiness rule. F' GMM0 may reserve the slot after B GMM0 has
released ownership only when the replay thread is already waiting and B GMM1 has
not requested a ticket. If B GMM1 reaches the queue first, the deterministic order
is `B0 -> B1 -> F'0 -> F'1`; replay can no longer overtake a ready B1 as it did in
stage fifteen.

Optional FSDP collectives are not members of the generic MoE HCCL FIFO. An FSDP
unshard may be skipped when parameters are already resident, so reserving it in a
mandatory FIFO can leave an unpublished predecessor event. FSDP keeps its semantic
all-gather/reduce-scatter event tickets, while the generic HCCL queue contains only
the mandatory MoE AllToAll sequence.

Two rejected admission protocols sharpened this design:

1. A Gloo all-reduce at every `B-combine` computed a global AND of replay readiness.
   A rolling 512-token profile completed at 335.442 ms with 1.086 ms of consensus
   host time and 169.972 ms of device-free intervals, but repeated profiled runs
   could stall after rank-dependent profiler parsing. Runtime control collectives
   inside autograd callbacks are therefore not retained.
2. Strict HCCL primary priority used
   `B-combine -> B-dispatch -> F'-dispatch -> F'-combine`. It completed a
   512-token profile at 323.293 ms, 17.457 ms slower than stage fifteen, because
   moving replay behind B-dispatch increased device-free intervals to 164.255 ms.

The deterministic scheduler suite completes with 15 tests and lint passes. The
targeted NPU cached-backward test also passes after the devices became available.
An eight-rank, two-window smoke run reports `loss=12.52`, `grad_norm=56.94`, and
10.65 GB peak memory.

Two independent retained-policy profiles show that the admission semantics work,
but do not yet improve short-sequence throughput:

| 512-token metric | Stage fifteen | Stage sixteen run 1 | Stage sixteen run 2 |
| --- | ---: | ---: | ---: |
| Steady profiler step | 305.836 ms | 326.614 ms | 409.198 ms |
| Compute union | 108.045 ms | 108.155 ms | 108.241 ms |
| Communication union | 61.722 ms | 66.880 ms | 59.254 ms |
| No-device-task intervals | 143.800 ms | 158.382 ms | 251.869 ms |
| GMM admission | interleave | opportunistic | defer-primary |
| B7 GMM ownership wait | 1.046 ms | 0.157 ms | 0.132 ms |

The readiness-aware GMM rule removes almost 0.9 ms of host-visible ownership wait
when compared with stage fifteen. The remaining regression is not GMM arithmetic
or direct GMM overlap: both traces retain 64 GroupedMatmul tasks and primary B1
cannot be overtaken once ready. It is the non-preemptive HCCL admission window.
B-dispatch cannot request ownership until F'-dispatch has both reached and consumed
its fixed slot, so replay host-readiness jitter expands into device-free gaps. The
second run demonstrates the sensitivity: compute is unchanged and communication is
shorter, while task-free time grows by 108.069 ms relative to stage fifteen.

The first exclusive 16384-token, eight-step run reports 2.17 s/step after the
compile warmup and 41.37 GB peak memory. This is 0.9 percent slower than stage
fifteen's 2.15 s/step representative run and remains within the observed
run-to-run variance. Two additional repetitions were not started after unrelated
Ray/VLLM and MindSpeed jobs occupied the two eight-device partitions; contended
measurements are intentionally excluded.

### Stage seventeen: device-ready replay deadline

The top B7/F'6 pair now publishes a replay-ready NPU event after attention,
routing, dispatch metadata, and dispatch-input permutation have been submitted.
B7 snapshots that event at the start of B-combine. At B-combine completion, a
one-element HCCL MIN on a dedicated control communicator computes the all-rank
decision. The scheduler then reserves both dispatch tickets in one critical
section: either `F'-dispatch -> B-dispatch` when every rank was device-ready at
the deadline, or `B-dispatch -> F'-dispatch` when any rank missed it. Host thread
arrival can no longer change the collective order after admission.

The control collective is intentionally limited to the configured top pair.
Applying it again to B6/F'5 allowed all ranks to finish both replay collectives
but stalled the rolling native FSDP tail. Lower rolling pairs therefore retain
the deterministic replay-first policy from stage sixteen until the FSDP control
plane is separated. An earlier prototype that sampled only a host ThreadEvent
could admit replay while its NPU preprocessing was still pending; its no-log
eight-rank run stalled, while instrumentation delays changed the decision to
defer and completed. Requiring a completed NPU event removes that timing
dependency.

The scheduler suite completes with 17 tests and lint passes. The two-rank NPU
cached-backward differential test passes. A clean eight-rank, two-window smoke
run reports `loss=12.52`, `grad_norm=56.97`, and 10.65 GB peak memory.

#### Stage-seventeen performance validation

An exclusive eight-NPU A/B profile used the same model, data, physical devices,
global batch, and profiler window; only inter-layer replay was toggled. The two
profiled steps average 578.667 ms for baseline and 593.094 ms for LIP, a 14.427 ms
or 2.49 percent regression. The latest steady step makes the communication
bottleneck more visible:

| 512-token metric | Baseline | Stage seventeen | Delta |
| --- | ---: | ---: | ---: |
| Steady profiler step | 566.660 ms | 587.458 ms | +20.798 ms |
| Compute union | 194.875 ms | 194.456 ms | -0.419 ms |
| Communication union | 190.338 ms | 209.899 ms | +19.562 ms |
| Compute/communication overlap | 20.812 ms | 24.056 ms | +3.244 ms |
| No compute/communication task | 202.259 ms | 207.159 ms | +4.900 ms |
| AllToAll aggregate | 68.279 ms | 98.244 ms | +29.965 ms |
| ReduceScatter aggregate | 43.604 ms | 94.495 ms | +50.892 ms |
| GroupedMatmul count | 128 | 128 | 0 |
| GroupedMatmul aggregate | 14.987 ms | 15.414 ms | +0.427 ms |

Both top-pair admission decisions deferred replay, so the retained trace contains
no successful replay-before-primary dispatch admission. The two admission MIN
collectives add two AllReduce calls and their host ranges total 1.948 ms. More
importantly, one sidecar-communicator AllToAll lasts 52.975 ms while a
primary-communicator FSDP ReduceScatter lasts 50.374 ms; their device intervals
overlap for 47.958 ms. Baseline's longest communication operation is only
12.805 ms. Independent communicators therefore preserve collective ordering but
do not isolate the physical HCCL links. Stage sixteen's removal of optional FSDP
collectives from the mandatory FIFO reopened cross-communicator contention.

The long-sequence result is worse. At 16384 tokens, baseline stabilizes at
1.81--1.83 s/step and completes all 16 available steps with 15.61 GB peak memory.
The first LIP run completes eight steps at 2.20 s/step. A repeated run reaches
2.05--2.12 s/step before slowing to 2.60 and 3.85 s/step at steps 12 and 13, then
rank 2 fails while allocating 610 MiB with 58.62 GiB already allocated and only
353 MiB free. Thus stage seventeen is 12--21 percent slower in its initially
stable 16K interval and is not long-run memory stable.

The evidence ranks the remaining causes as follows:

1. Cross-communicator HCCL contention and rank launch skew are the dominant
   short-profile regression. Replay AllToAll and the actual FSDP ReduceScatter
   need a late-bound, device-event ownership handoff rather than communicator
   separation alone.
2. The device-ready deadline is too early or too strict: it deferred two of two
   opportunities and therefore paid scheduling cost without exposing the intended
   communication/compute overlap.
3. Control consensus and fork/join coordination add about 5 ms of task-free time.
   They matter after the HCCL expansion is removed but are not the first-order
   20.8 ms regression.
4. Long-sequence sidecar graph, detached input, dynamic-shape workspace, and
   cross-stream allocation lifetimes require a per-step memory audit. Baseline's
   successful 16-step run rules out the input batch alone as the OOM cause.

Extra arithmetic and GMM competition are not supported as primary causes in this
trace: baseline and LIP execute the same 128 GroupedMatmul tasks, compute union is
slightly lower under LIP, and aggregate GMM time changes by less than 0.5 ms.

### Stage eighteen through twenty: replay lifetime and guarded pause

The custom native checkpoint frame originally formed a strong reference cycle:
`ReplayFrame -> native checkpoint frame -> recompute closure -> replay holder ->
ReplayFrame`. The first lifetime fix broke that cycle at sidecar join, but the 16K
run still reproduced the same step-13 OOM. A guarded run then exposed a second
lifecycle bug: when the memory policy paused a sidecar, native backward completed
without consuming its replay frame, so the next forward rejected the stale frame.

Cleanup now has two explicit boundaries. Sidecar join releases only attention,
dispatch, expert, and combine temporaries. A hook on the replay layer's input runs
after that layer's native backward, marks the frame consumed, releases the native
checkpoint graph and detached inputs, breaks the closure cycle, and removes the
frame from the controller. This applies equally to scheduled and paused replay.
The scheduler suite completes with 18 tests, including immediate reference-count
reclamation without `gc.collect`; the two-rank NPU cached-backward differential
test remains bit-exact.

An unguarded 16384-token run still fails at the deterministic high-memory batch at
step 13, proving the dominant peak is a sidecar/native-recompute overlap for a
specific dynamic shape rather than only a Python reference leak. With an 8 GB free
reserve and a 1.5 replay safety factor, the policy schedules the first eight
sidecars, then pauses at 34.10 GB allocated and 15.43 GB free. All 16 available
steps complete; after pause, the rate stabilizes at 1.79--1.82 s/step, effectively
the baseline range. Allocated memory remains at 34.10 GB instead of returning to
the 15.61 GB baseline peak, so dynamic-shape replay allocations or NPU workspaces
remain live even after Python graph retirement. Pause currently bounds that growth
but does not eliminate it.

The development launcher now defaults to the validated 8 GB reserve and 1.5 safety
factor, while command-line overrides can still disable or retune the guard. It also
disables the final HuggingFace weight export so profiling runs do not create an
unrelated 74 GB checkpoint.

### Stage nineteen through twenty-one: late-bound HCCL handoff

The retained stage-seventeen trace shows that the 52.975 ms sidecar operation is
replay combine, not dispatch. Its interval overlaps the current FSDP
ReduceScatter for 47.958 ms. The current layer now waits at the actual
ReduceScatter invocation for a device event published after the sidecar's final
HCCL operation. This is late-bound: a skipped FSDP collective reserves no ticket
and cannot leave an unpublished predecessor. A final sidecar event provides the
same handoff if checkpoint early-stop ends recomputation before combine.

The fence removes cross-communicator AllToAll/ReduceScatter overlap, but rank launch
skew can still make replay combine wait inside its alternate communicator. The top
pair therefore executes one rank-consistent control consensus after replay GMM and
immediately before combine. Lower rolling pairs do not repeat this control
collective because that previously stalled the native FSDP tail.

An exclusive four-NPU piercing profile validates the communication mechanism. All
four sidecar AllToAll operations complete in 0.26--0.44 ms, cross-group overlap
with primary ReduceScatter is zero, and the longest ReduceScatter is 10.75 ms.
The same-device A/B remains slower, however:

| EP=4, 512-token latest-step metric | Baseline | LIP | Delta |
| --- | ---: | ---: | ---: |
| Profiler step | 586.345 ms | 600.354 ms | +14.009 ms |
| Compute union | 297.697 ms | 298.143 ms | +0.447 ms |
| Communication union | 112.875 ms | 127.853 ms | +14.977 ms |
| Compute/communication overlap | 26.500 ms | 25.692 ms | -0.807 ms |
| No compute/communication task | 202.273 ms | 200.050 ms | -2.223 ms |
| AllToAll aggregate | 28.230 ms | 35.426 ms | +7.197 ms |
| GroupedMatmul count | 128 | 128 | 0 |

Thus deterministic handoff removes the catastrophic communication expansion but
does not yet create a throughput gain. The remaining 2.39 percent EP=4 regression
is communication cost, including four extra control AllReduce calls per profiled
step and higher AllToAll/all-gather aggregate time. An attempted EP=8 reprofile is
excluded because an unrelated MindSpeed-MM job started on the same physical NPUs
during the profiler window and then occupied almost all HBM.

### Stage twenty-two: exclusive EP=8 validation

An exclusive eight-NPU A/B on physical devices 8--15 validates the guarded
handoff at the target EP size. Both runs use the same 512-token dynamic batches,
global batch size 16, two microbatches per optimizer step, and profiler steps 9
and 10. Replay remains `within_budget` for the full run. Baseline and LIP both
execute 128 GroupedMatmul tasks per profiled step, and their compute unions differ
by less than 1.5 ms, so the LIP path does not add expert arithmetic.

| EP=8 metric | Baseline step 9 | LIP step 9 | Baseline step 10 | LIP step 10 |
| --- | ---: | ---: | ---: | ---: |
| Profiler step | 588.735 ms | 604.139 ms | 578.738 ms | 615.420 ms |
| Compute union | 194.578 ms | 194.922 ms | 193.941 ms | 193.091 ms |
| Communication union | 136.488 ms | 172.438 ms | 134.357 ms | 246.937 ms |
| Compute/communication overlap | 16.132 ms | 22.935 ms | 15.955 ms | 23.726 ms |
| GroupedMatmul aggregate | 15.555 ms | 16.464 ms | 15.021 ms | 16.327 ms |

The LIP regressions are 2.62 and 6.34 percent. The second step contains a
63.531 ms replay-combine control AllReduce. Rank zero reaches the collective
early, and the current B7 ReduceScatter then waits 63.321 ms on the replay HCCL
completion event. This is rank-arrival skew exposed on the main-backward critical
path, not additional GMM work or direct cross-communicator overlap.

Even the step without the long control wait remains slower. Primary-group
AllToAll P95 increases from 0.661 ms to 1.398 ms, its aggregate grows from
33.599 ms to 49.537 ms despite moving four calls to the replay communicator, and
ReduceScatter median increases from 0.678 ms to 1.204 ms. This indicates that the
current replay schedule increases rank launch skew throughout the primary HCCL
sequence. The final event fence prevents unsafe physical-link contention but also
turns late replay completion into a mandatory B7 stall.

A reverse-order repeat is excluded: Ray workers appeared on physical devices
8--11 after launch, reduced observed free memory by about 13 GB, and changed the
steady rate. No numbers from that run are used in the A/B conclusion.

### Stage twenty-two: active-allocation isolation

A ten-step EP=4 memory-only comparison on physical devices 12--15 isolates the
persistent growth from checkpoint wrapping and pause handling:

| 512-token EP=4 path | Final allocated | Peak allocated | Steady growth |
| --- | ---: | ---: | ---: |
| Baseline | 15.67 GB | 24.97 GB | none |
| Active LIP | 17.46 GB | 26.67 GB | about 0.19 GB/step |
| LIP forced to fallback before first replay | 15.67 GB | 24.97 GB | none |
| Active LIP with per-step GC and empty-cache | 17.48 GB | 26.69 GB | about 0.19 GB/step |

The forced-fallback path exactly matches baseline and its two per-step probes stay
fixed at 17.81 and 23.03 GB. The controller, native checkpoint wrapper, paused
frame cleanup, and allocator cache therefore do not cause the growth. It starts
only after sidecar execution.

The rank-zero allocator snapshot strengthens that conclusion. Baseline has
7.84 GB of `active_allocated` blocks on the primary stream. LIP has 10.05 GB;
2.21 GB across 528 active blocks belongs to the alternate replay stream. A
targeted experiment explicitly removed the graph-task entries from native
checkpoint `recomputed`, `recomp_counter`, `is_recomputed`, and holder handles at
the input-gradient hook. It produced the same 0.19 GB/step slope and was reverted.
The remaining owner is therefore inside the sidecar fused-MoE execution or its
cross-stream tensor lifetime, not the native checkpoint frame.

### Cooperative autograd callback replay

Commit `ba1d9930` removes the replay `ThreadPoolExecutor`, futures, barrier, and
phase-level worker waits. A cooperative driver now runs on the native autograd
host thread and advances replay from the existing boundaries in this order:

1. B7 combine before callback launches F'6 attention.
2. B7 combine after callback launches F'6 dispatch.
3. The second `GmmFunction.backward` records a device event and launches F'6
   experts behind that event.
4. B7 dispatch after callback launches F'6 combine.
5. B7 FSDP ReduceScatter finalizes replay and consumes its completion event.

Each phase enters and exits the replay stream, checkpoint saved-tensor hook, and
replay RNG context on the same host thread. This prevents F'6 state from leaking
into native B7 autograd. The deterministic regression suite now has 21 tests,
including same-thread phase order and a hard assertion that cooperative phases
never wait for host-side phase signals. Ruff passes, the two-NPU cached-backward
differential test passes on exclusive devices, and an EP=4 512-token smoke run
reports `loss=12.61` and `grad_norm=75.22`.

An exclusive EP=4 16384-token A/B profile used physical devices 12--15, global
batch four, and the same data and profiler window. The latest complete profiler
step shows that the change removes HCCL inflation but remains slower:

| EP=4 16K metric | Baseline | Cooperative replay | Delta |
| --- | ---: | ---: | ---: |
| Profiler step | 1690.308 ms | 1883.808 ms | +193.500 ms (+11.45%) |
| Compute union | 954.380 ms | 954.641 ms | +0.261 ms |
| Communication union | 621.355 ms | 594.652 ms | -26.703 ms |
| Device task coverage | 1491.877 ms | 1463.817 ms | -28.060 ms |
| No device task | 198.431 ms | 419.991 ms | +221.560 ms |
| AllToAll aggregate | 193.751 ms | 211.319 ms | +17.568 ms |
| AllGather aggregate | 237.612 ms | 214.320 ms | -23.292 ms |
| ReduceScatter aggregate | 186.385 ms | 165.688 ms | -20.697 ms |

The five cooperative callbacks occupy about 280 ms of host time: attention
7.630 ms, dispatch 60.352 ms, experts 101.565 ms, combine 109.794 ms, and final
join 1.046 ms. Dispatch, experts, and combine are dominated by NPU
`empty_tensor`, GroupedMatmul, and asynchronous AllToAll enqueue paths. Running
these calls synchronously inside native B7 callbacks delays subsequent B7 kernel
submission and creates the extra 221.560 ms device-idle interval. FIFO device
event waits total less than 0.5 ms, and compute is unchanged, so neither GMM
competition nor HCCL contention is the primary regression in this version.

An ILP-first reverse-order run without profiling confirms the direction: its
completion progress reports 2.02 s/iteration versus 1.72 s/iteration for the
following baseline run. These cumulative progress values include warmup and are
not used as the primary percentage estimate.

### Asynchronous replay launcher

Commit `db22c57f` replaces the cooperative driver with a persistent single-worker
mailbox. Native B7 callbacks publish deterministic phase commands while the
worker performs replay NPU submission. The main autograd thread waits only before
B7 dispatch, where shared-communicator ordering requires F'6 dispatch to have
been submitted first, and before B7 ReduceScatter for final replay completion.
Resource-class FIFO tickets and the device-event handoffs remain unchanged.

The scheduler and cached-backward suites pass 19 tests. An exclusive EP=4 smoke
run on physical devices 4--7 reports `loss=12.61`, `grad_norm=75.16`, and
4.56 seconds/iteration. The matching 16384-token profiler comparison shows that
moving replay submission off the autograd host thread is necessary but not
sufficient:

| EP=4 16K metric | Baseline | Async launcher | Delta |
| --- | ---: | ---: | ---: |
| Profiler step | 1636.588 ms | 1891.685 ms | +255.097 ms (+15.59%) |
| Compute union | 952.880 ms | 953.537 ms | +0.657 ms |
| Communication union | 633.529 ms | 653.557 ms | +20.028 ms |
| Device task coverage | 1504.675 ms | 1524.400 ms | +19.725 ms |
| No device task | 131.912 ms | 367.286 ms | +235.374 ms |
| AllToAll aggregate | 184.965 ms | 239.685 ms | +54.720 ms |
| AllGather aggregate | 260.441 ms | 212.262 ms | -48.179 ms |
| AllReduce aggregate | 1.696 ms | 35.067 ms | +33.371 ms |
| ReduceScatter aggregate | 186.427 ms | 166.543 ms | -19.884 ms |

Phase publication now costs only 0.03--0.06 ms, but B7 waits 65.024 ms at the
dispatch boundary for replay dispatch host/runtime submission. Finalization
waits total 0.833 ms, FIFO waits total 0.400 ms, and launcher join costs 0.221 ms.
The remaining regression is therefore not Python mailbox overhead, extra expert
arithmetic, or the FIFO device-event handoff. It is the host-readiness dependency
introduced by preserving collective order on the shared communicator, followed
by rank launch skew and HCCL inflation.

Two follow-up experiments were deliberately reverted. Publishing replay
attention and dispatch before B7 combine launch increased the step to 1926.114 ms
and the dispatch wait to 71.587 ms. Preallocating the replay AllToAll output
increased the step to 1937.430 ms and left a 67.783 ms dispatch wait. The
`empty_tensor` region seen by the profiler is consequently a symptom of runtime
submission/synchronization, not an ordinary `torch.empty` allocation bottleneck.

### Backward-first dispatch admission

An isolated Ascend stream experiment tested whether a consumer stream could
enqueue `wait_event` before the producer records that event. The normal
record-before-wait control preserved tensor visibility in 100/100 trials. The
wait-before-record order observed stale data in 61/100 trials. An unrecorded NPU
event therefore cannot provide the device-only future needed to preserve replay-
first ordering without host readiness synchronization.

Commit `f6a15646` adopts the deterministic fallback: the shared HCCL FIFO reserves
B7 dispatch before F'6 dispatch. A ready native backward dispatch is submitted
immediately; the replay worker waits for the B7 completion event and submits F'6
after it. This removes the 65 ms `wait_replay_dispatch` boundary while retaining
replay attention overlap before B7 dispatch and replay expert work after it. The
19 regression tests and ruff pass. An EP=4 512-token piercing run completes with
`loss=12.61`, `grad_norm=75.18`, and no collective-order failure or deadlock.

The intended EP=4 16K profile is excluded because unrelated Ray/vLLM workers
expanded on physical devices 4 and 5 during step two, leaving only 72 and 578 MiB
free and causing OOM. An exclusive physical-device 6--7 EP=2 A/B is retained as
a scheduling diagnostic, not as a replacement for the EP=4 performance claim:

| EP=2 16K metric | Baseline step 1 | B-first step 1 | Baseline step 2 | B-first step 2 |
| --- | ---: | ---: | ---: | ---: |
| Profiler step | 2385.658 ms | 2054.088 ms | 1393.422 ms | 5373.302 ms |
| Compute union | 1056.057 ms | 1063.224 ms | 967.406 ms | 974.555 ms |
| Communication union | 650.030 ms | 525.857 ms | 372.886 ms | 258.694 ms |
| Device task coverage | 1613.533 ms | 1496.831 ms | 1271.361 ms | 1196.478 ms |
| No device task | 772.125 ms | 557.256 ms | 122.060 ms | 4176.825 ms |

The first profiled window is 331.570 ms faster and confirms that prioritizing B7
can shorten the critical path when allocation remains healthy. The second window
is dominated by two FSDP gradient `aten::div`/`empty_tensor` stalls, corresponding
to 2.224 and 1.111 second device-idle gaps. ILP finishes with 43.54 GB allocated
and 57.80 GB peak versus baseline's 31.34 and 45.60 GB. Compute changes by only
about 7 ms per window and communication is lower, so the sustained blocker has
moved from dispatch admission to sidecar allocation lifetime. The next stage
must release or reuse replay-stream active blocks before optimizing overlap
further.

A follow-up EP=2 16K run disabled profiling and requested ten steps on the same
exclusive physical devices. It completed four steps, slowed from 4.16 to 6.05
seconds/iteration after warmup, and OOMed while entering step five. Rank zero
reported 59.69 GiB allocated, 59.69 GiB active, 59.99 GiB reserved, and only
224.71 MiB free when a 540 MiB allocation failed. The equality between allocated
and active memory rules out profiler retention and allocator-cache fragmentation
as the main explanation. Together with the earlier EP=4 ten-step slope of about
0.19 GB/step, this confirms that the extra allocation does not settle after the
first few warmup steps; EP=2 reaches capacity before a ten-step plateau can be
observed.

### One-shot host-parallel replay

Commit `c935f4b3` removes the five-phase replay mailbox. B7 remains on native
autograd while a persistent launcher worker receives one start signal at the B7
dispatch boundary and runs the complete F'6 attention, dispatch, experts, and
combine sequence on the alternate NPU stream. B7 performs no replay join. The
only host completion wait is the dependency join at B6 entry, and the resulting
sidecar graph satisfies the native non-reentrant checkpoint frame so B6 does not
repeat F'6.

The physical-resource order is deliberately backward-first:

1. B7 dispatch.
2. B7 FSDP ReduceScatter.
3. F'6 dispatch.
4. F'6 combine.

This permits F'6 attention to overlap B7 dispatch while ensuring that neither
replay HCCL call can delay a ready B7 collective. An initial implementation used
the primary EP process group from both host threads. It completed a one-step
smoke run but repeatedly stalled after profiling started because local FIFO order
does not make concurrent process-group submission safe under cross-rank launch
skew. Commit `7d5bd1a2` restores an independent replay EP communicator while
retaining the device-event physical-resource order. A three-step EP=4 profiling
smoke then completed across all ranks.

The scheduler and cached-backward suites pass 20 tests. The EP=4 piercing run
reports `loss=12.61` and `grad_norm=75.22`. A same-device 16384-token A/B gives:

| EP=4 16K steady metric | Baseline | One-shot replay | Delta |
| --- | ---: | ---: | ---: |
| Profiler step | 1647.759 ms | 1846.600 ms | +198.841 ms (+12.07%) |
| Compute union | 953.608 ms | 953.277 ms | -0.331 ms |
| Communication union | 624.595 ms | 604.509 ms | -20.086 ms |
| Device task coverage | 1494.801 ms | 1474.953 ms | -19.848 ms |
| No device task | 152.958 ms | 371.647 ms | +218.689 ms |
| AllToAll aggregate | 184.788 ms | 200.866 ms | +16.078 ms |
| AllGather aggregate | 247.771 ms | 215.821 ms | -31.950 ms |
| ReduceScatter aggregate | 186.224 ms | 186.230 ms | +0.006 ms |

The one-shot start costs 0.045 ms and B7 FIFO waits total 0.262 ms. B6 instead
waits 140.812 ms for replay completion. Equal compute proves that F'6 is not
duplicated, and lower communication time rules out HCCL contention as the main
regression. Strict B7 priority currently exposes too much replay tail after B7;
only the attention portion has a useful overlap window.

Memory remains a separate blocker. After four steps baseline reports 15.67 GB
allocated and 25.54 GB peak, while one-shot replay reports 27.49 and 37.36 GB.
The approximately 11.82 GB active-allocation difference must be removed before a
long-run performance claim.

### One-shot lifecycle repair and deterministic interleave

The one-shot launcher initially retained its completed replay callable, and the
controller retained the owning schedule and launcher until the next forward.
The join path now drops the callable before waking autograd, detaches the launcher
from its schedule, clears completed device-event tickets, and removes the active
controller references. Unit tests use weak references to require immediate
reclamation instead of depending on cyclic garbage collection.

This Python cleanup was necessary but did not remove the long-sequence growth. A
six-step EP=4 run still reached 33.18 GB allocated, and a continued run OOMed with
59.52 GiB allocated and active. The dominant owner was the asynchronous HCCL
handle: `AsyncCollectiveHandle.wait()` retained its completed `Work` object, which
in turn retained the dispatch or combine input/output storage. Clearing `_work`
at the first wait removes the per-step sidecar storage retention. A regression
test now verifies that a completed Work object is reclaimed immediately.

A fully arrival-ordered device FIFO was also rejected. Different ranks can observe
replay dispatch and B7 ReduceScatter in opposite host arrival order, creating a
cross-communicator event cycle even though each local FIFO is valid. The safe
non-backward-priority order is therefore predeclared identically on every rank:

1. B7 dispatch.
2. F'6 dispatch.
3. B7 FSDP ReduceScatter.
4. F'6 combine.

This gives replay an early communication slot without restoring the previous rule
that all ready B7 communication always wins. Ruff and 23 scheduler/cached-backward
tests pass. The EP=4 16384-token six-step lifecycle run reports 15.67 GB current
and 25.55 GB peak allocation, matching the baseline's 15.67 and 25.54 GB. The old
11.82 GB sidecar allocation is therefore not intrinsic to ILP.

The same-device four-step profile gives the following steady-window comparison:

| EP=4 16K steady metric | Baseline | Lifecycle/interleave ILP | Delta |
| --- | ---: | ---: | ---: |
| Profiler step | 1647.759 ms | 1615.263 ms | -32.496 ms (-1.97%) |
| Compute union | 953.608 ms | 952.104 ms | -1.504 ms |
| Communication union | 624.595 ms | 608.737 ms | -15.858 ms |
| Device task coverage | 1494.801 ms | 1471.899 ms | -22.902 ms |
| No device task | 152.958 ms | 143.365 ms | -9.593 ms |
| AllToAll aggregate | 184.788 ms | 201.119 ms | +16.331 ms |
| AllGather aggregate | 247.771 ms | 219.618 ms | -28.153 ms |
| ReduceScatter aggregate | 186.224 ms | 186.438 ms | +0.214 ms |

The B6 dependency join falls from 140.812 ms in the backward-priority one-shot
profile to 4.684 ms. Equal compute confirms that F'6 still replaces native B6
recomputation. The first retained profile window remains 101.773 ms slower than
baseline, so repeated exclusive runs are still required before treating the
1.97-percent steady-window improvement as a stable throughput claim. The largest
remaining device-side regression is the 16.331 ms AllToAll aggregate increase.

### B7 dispatch event gating

The lifecycle/interleave profile contains one 18.021 ms primary-communicator
AllToAll while normal calls take about 3.3--4.8 ms. Timeline correlation identifies
it as B7 backward dispatch: the one-shot worker starts F'6 attention as soon as the
B7 dispatch host callback fires, and that device work overlaps the still-running
primary AllToAll. Moving the worker start from the callback's `before` boundary to
its `after` boundary alone does not fix the device overlap. The outlier remains
18.046 ms and the steady profile step regresses from 1615.263 to 1620.812 ms, so
host submission order is not the controlling boundary.

The accepted implementation records the B7 dispatch completion event, starts the
worker without blocking the autograd host thread, and inserts an event wait on the
replay stream before F'6 attention. The main B7 AllToAll returns to 4.550 ms, B6
join falls to 0.023 ms in the first retained steady profile, and allocation remains
15.67 GB current / 25.54 GB peak. That single window is 1607.114 ms versus the
1647.759 ms baseline, a 2.47-percent improvement.

A reverse-order, same-device eight-step A/B retains four profile windows per path
and disables checkpoint saving. Its more conservative aggregate is:

| EP=4 16K four-window mean | Baseline | Dispatch-event ILP | Delta |
| --- | ---: | ---: | ---: |
| Profiler step | 1385.178 ms | 1375.044 ms | -10.134 ms (-0.73%) |
| Compute union | 926.684 ms | 924.005 ms | -2.679 ms |
| Communication union | 346.276 ms | 353.852 ms | +7.576 ms |
| Device task coverage | 1239.263 ms | 1241.455 ms | +2.192 ms |
| No device task | 145.915 ms | 133.590 ms | -12.325 ms |
| AllToAll aggregate | 180.738 ms | 189.623 ms | +8.885 ms |
| AllGather aggregate | 122.371 ms | 121.372 ms | -1.000 ms |
| ReduceScatter aggregate | 40.370 ms | 39.608 ms | -0.762 ms |

Three of four paired windows improve; one regresses by 0.418 ms. Median step time
improves by 8.482 ms, or 0.61 percent. B6 replay waits range from 0.713 to 1.104 ms.
The demonstrated result is therefore a small positive throughput gain, not the
single-window 2.47-percent upper observation. The current bottleneck is replay
dispatch on the alternate communicator: its delay is off the B6 critical path but
raises mean AllToAll aggregate by 8.885 ms and limits further improvement.

### Replay-attention ownership handoff

All-rank profiling shows that the slow alternate-communicator dispatch is mostly
an arrival-skew symptom rather than slow HCCL transfer. In two steady windows the
earliest replay rank enters dispatch 30--38 ms before the latest rank, while the
latest rank completes the actual transfer in about 4 ms. The delayed rank changes
with the packed sequence workload.

The preceding trace interpretation also needs one correction. F'6 forward
FlashAttention and B7 FlashAttention backward were nominally placed on different
streams, but the replay task could remain behind the native backward task in the
AI Core scheduler. The slow rank consequently reached replay dispatch only after
the long B7 attention region. Serializing the two kernels in a replay-first order
does not make the slow rank's B7 attention intrinsically faster: after handoff it
still takes about 32 ms. That duration is mainly caused by packed-sequence shape
imbalance, not HCCL or direct dual-FA contention.

The accepted scheduler publishes an event immediately after F'6 attention has
been enqueued. At the B7 dispatch phase callback, the native host thread waits
only for event publication and inserts `wait_event` on the primary NPU stream.
Device execution becomes:

1. B7 dispatch completes.
2. F'6 attention owns the AI Core interval.
3. F'6 dispatch overlaps B7 attention backward.

This avoids unsupported wait-before-record event use and does not synchronize the
host with device completion. In the last all-rank steady window, replay dispatch
arrival skew falls from about 32 ms to about 10 ms. The same window improves from
1666.669 to 1638.618 ms, or 1.68 percent, while memory remains at 15.67 GB current
and 25.54 GB peak.

A same-device eight-step baseline/ILP run retains four rank-zero profile windows
for each path. Every paired window improves:

| EP=4 16K four-window mean | Baseline | Attention-handoff ILP | Delta |
| --- | ---: | ---: | ---: |
| Profiler step | 1381.015 ms | 1362.523 ms | -18.492 ms (-1.34%) |
| Compute union | 926.976 ms | 926.640 ms | -0.336 ms |
| Communication union | 353.211 ms | 348.365 ms | -4.846 ms |
| Device task coverage | 1247.263 ms | 1236.619 ms | -10.644 ms |
| No device task | 133.752 ms | 125.904 ms | -7.848 ms |
| AllToAll aggregate | 183.182 ms | 186.881 ms | +3.699 ms |
| AllGather aggregate | 127.355 ms | 116.680 ms | -10.675 ms |
| ReduceScatter aggregate | 40.161 ms | 41.625 ms | +1.464 ms |

The paired savings are 18.641, 7.132, 23.122, and 25.073 ms; the median saving is
20.882 ms. Equal compute confirms that F'6 continues to replace native B6
recomputation rather than removing work. The remaining AllToAll increase is now
3.699 ms instead of 8.885 ms, and B6 dependency joins remain below one millisecond
in the final window.

### Full backward window

The same replay-attention ownership handoff now rolls over every backward boundary
of the trimmed eight-layer model:

```text
B7/F'6 -> B6/F'5 -> B5/F'4 -> B4/F'3 -> B3/F'2 -> B2/F'1 -> B1/F'0
```

Each replay graph satisfies the next layer's native non-reentrant checkpoint frame
before that layer enters backward, then releases its launcher, schedule, collective
storage, and cached graph before the following pair is admitted. A 512-token EP=4
smoke run completes two optimizer steps with `loss=9.16`, `grad_norm=41.91`, and
15.67 GB current / 23.81 GB peak allocation.

An exclusive EP=4 16384-token comparison runs baseline, one boundary, and all
seven boundaries on the same physical devices. Each path retains four rank-zero
profile windows:

| EP=4 16K four-window mean | Baseline | One boundary | All seven boundaries |
| --- | ---: | ---: | ---: |
| Profiler step | 1412.522 ms | 1381.347 ms | 1357.503 ms |
| Speedup vs baseline | - | 2.21% | 3.90% |
| Compute union | 927.348 ms | 926.404 ms | 925.166 ms |
| Communication union | 371.793 ms | 358.697 ms | 348.761 ms |
| Device task coverage | 1265.998 ms | 1247.095 ms | 1231.492 ms |
| No device task | 146.524 ms | 134.252 ms | 126.011 ms |
| AllToAll aggregate | 183.542 ms | 187.209 ms | 211.264 ms |
| AllGather aggregate | 145.270 ms | 127.194 ms | 91.286 ms |
| ReduceScatter aggregate | 39.964 ms | 41.288 ms | 43.536 ms |

All four full-window samples improve over their paired baseline by 36.210, 106.441,
43.809, and 33.616 ms. The mean saving is 55.019 ms and the median saving is
40.010 ms. Seven `fork_join`, launcher, dispatch FIFO, and ReduceScatter FIFO
markers appear in each steady profile, confirming that all seven pairs execute.
Allocation remains 15.67 GB current and rises from the baseline's 25.54 GB peak to
27.12 GB peak.

The result is positive but deliberately sublinear. Extrapolating the one-boundary
2.21-percent result to seven independent boundaries would predict about 15 percent,
which is not physically available: lower pairs share the same backward critical
path and HCCL fabric. Full-window AllToAll aggregate grows by 27.722 ms, while its
main gain comes from hiding 53.985 ms of AllGather time and removing 20.513 ms of
device-idle time. The next optimization target is therefore selective admission:
enable a lower replay boundary only when its predicted hidden AllGather and idle
tail exceed its replay AllToAll cost.

### Four-stage AI/HCCL complementary schedule

The full-window scheduler now uses all four complementary MoE phases instead of
handing off only replay attention. For every `Bn/F'(n-1)` pair, the device order is:

```text
stage 1: B combine (HCCL)  || F' attention (AI)
stage 2: B experts (AI)    || F' dispatch (HCCL)
stage 3: B dispatch (HCCL) || F' experts (AI)
stage 4: B attention (AI)  || F' combine (HCCL)
```

The replay worker starts once the B-combine call has been submitted. Host threads
wait only for event publication; NPU stream events enforce the AI phase handoffs.
The HCCL FIFO reserves one rank-consistent order before the first collective:

```text
B combine -> F' dispatch -> B dispatch -> F' combine -> B reduce-scatter
```

This order prevents concurrent HCCL ownership while still allowing the opposite
AI stream to run. The backward expert hook publishes the stage-2 completion event,
and the replay expert phase publishes the stage-3 completion event. B remains on
native autograd and each replay graph still replaces, rather than duplicates, the
next layer's non-reentrant checkpoint recomputation.

A fresh same-device EP=4 16384-token A/B run retains four rank-zero profile
windows per path. All four paired windows improve:

| EP=4 16K four-window mean | Fresh baseline | Four-stage ILP | Delta |
| --- | ---: | ---: | ---: |
| Profiler step | 1395.434 ms | 1307.885 ms | -87.549 ms (-6.27%) |
| Compute union | 927.561 ms | 936.756 ms | +9.195 ms |
| Communication union | 357.008 ms | 368.880 ms | +11.872 ms |
| AI/HCCL overlap | 34.834 ms | 117.118 ms | +82.284 ms |
| Device task coverage | 1250.607 ms | 1189.714 ms | -60.893 ms |
| No device task | 144.827 ms | 118.171 ms | -26.656 ms |
| AllToAll aggregate | 187.417 ms | 231.681 ms | +44.265 ms |
| AllGather aggregate | 127.274 ms | 91.392 ms | -35.881 ms |
| ReduceScatter aggregate | 40.114 ms | 42.984 ms | +2.870 ms |

The paired savings are 87.904, 103.718, 96.360, and 62.215 ms. The overlap gain
of 82.284 ms is close to the measured 87.549 ms step saving, which confirms that
the improvement is primarily the intended AI/HCCL complement rather than missing
recomputation. Both traces contain 192 AllToAll calls across four steps, or 48 per
step, so ILP adds no semantic replay AllToAll. The seven launcher, dispatch FIFO,
ReduceScatter FIFO, and join markers appear in every sampled step.

The price is higher instantaneous liveness and slower overlapped communication:
peak allocation is 28.39 GB versus 25.55 GB, and AllToAll aggregate grows by
44.265 ms even though its call count is unchanged. The added overlap hides that
cost today, but it is now the main limit between the demonstrated 6.27-percent
gain and the higher theoretical target.

### Phase-local replay storage release

The sidecar previously retained explicit references to every phase output until
the next native backward joined the replay. Those references duplicated the
lifetime already owned by the non-reentrant checkpoint frame and autograd saved
tensors. The replay now releases prepared/permuted dispatch storage after dispatch,
the dispatched expert input after expert launch, dispatch/combine state after
combine, and the replay output root after checkpoint capture validation.

On a fresh device-4--7 EP=4 16384-token run, ILP peak allocation falls from
28.25 GB to 27.07 GB while the same baseline remains 25.55 GB. This removes
1.18 GB of the 2.70 GB same-device overhead. Mean step time remains positive at
1316.576 ms versus 1402.460 ms baseline, a 6.12-percent improvement, and AI/HCCL
overlap remains 120.326 ms versus 34.642 ms. The remaining 1.52 GB is dominated
by tensors and operator workspaces that coexist only while the two streams are
actually concurrent. End-of-profile allocator snapshots contain 15.671 GB of
active baseline storage and 15.679 GB of active ILP storage, so there is no
material persistent replay-tensor leak after the phase-local releases. The ILP
snapshot does retain more inactive cached blocks; those affect reserved memory,
not the reported peak allocated-memory delta.

### AllToAll overlap-tax controls

The optimized trace still contains exactly 48 AllToAll calls per step. Per-call
sequence matching localizes the slowdown to the seven complementary boundaries:

| AllToAll region | Baseline | ILP | Delta |
| --- | ---: | ---: | ---: |
| Original forward | 58.270 ms | 60.227 ms | +1.956 ms |
| Top-layer recompute | 7.797 ms | 8.957 ms | +1.160 ms |
| B combine | 24.842 ms | 28.623 ms | +3.781 ms |
| F' dispatch | 32.359 ms | 41.870 ms | +9.511 ms |
| B dispatch | 24.835 ms | 39.403 ms | +14.568 ms |
| F' combine | 27.273 ms | 48.472 ms | +21.198 ms |
| Bottom-layer backward | 7.055 ms | 7.138 ms | +0.083 ms |

This rules out duplicated replay communication. The increase is an overlap tax:
HCCL transfers take longer while the opposite stream consumes shared memory and
device-fabric bandwidth. Two controls were rejected. Routing replay through the
primary communicator increased the same-device AllToAll delta to 52.680 ms and
reduced the speedup to 5.46 percent. Giving the entire replay stream lower
priority produced a 53.900 ms AllToAll delta and only 3.65-percent speedup;
launch-stream priority does not provide useful HCCL preemption on this stack.

Deferring F' combine until the complete B-attention backward is also not a valid
boundary: the native non-reentrant checkpoint join needs the replay graph before
that layer can publish its final input gradient, so the naive wait creates a
cycle. Finer-grained combine throttling must remain before that join and requires
an operator-level attention completion event, not an entire-layer completion
event.

The accepted configuration therefore retains the alternate communicator,
default replay-stream priority, and all four overlap stages. On the fresh
device-4--7 run, overlap grows by 85.684 ms while AllToAll aggregate grows by
52.258 ms and the total step still falls by 85.884 ms. Reducing communication
aggregate by serializing a positive overlap window would optimize the wrong
metric; a future admission rule should disable a boundary only when its measured
critical-path saving is non-positive.

### FlashAttention backward boundary

The attention wrapper now inserts a transparent autograd boundary on the
FlashAttention output only while an ILP schedule is active. During backward this
boundary runs after the output projection and immediately before the FA backward
node. It records an event on the native B stream; the replay stream waits on that
event before submitting F' combine. Every lower replay graph captures its own
native-layer schedule, so the rolling seven-layer window publishes B6 through B1
events to the correct following replay pair rather than reusing the upper pair's
schedule.

An event after the FA backward node was tested first and rejected. It moved F'
combine onto the Q/K/V projection interval, increasing the AllToAll delta to
57.782 ms and reducing the end-to-end speedup to 5.65 percent. The accepted
boundary therefore represents FA-backward admission, not whole-attention or
whole-layer completion:

```text
B output-projection backward
    -> attention boundary event
    -> B FlashAttention backward || F' combine
    -> B Q/K/V projection backward
```

A same-device device-8--11 EP=4 16384-token control compares baseline, natural
stage-4 admission, and the FA-start boundary over four profile windows:

| Four-window mean | Baseline | Natural stage 4 | FA-start boundary |
| --- | ---: | ---: | ---: |
| Step | 1395.738 ms | 1319.621 ms | 1319.585 ms |
| Speedup | - | 5.45% | 5.46% |
| AI/HCCL overlap | 34.091 ms | 119.926 ms | 108.654 ms |
| AllToAll delta vs baseline | - | +48.430 ms | +43.522 ms |
| Peak allocation | 25.55 GB | 27.07 GB | 26.70 GB |

The boundary preserves step time while removing 4.908 ms of AllToAll inflation
and 0.37 GB of transient peak allocation. All paths retain 48 AllToAll calls per
step. A two-step 512-token smoke run also completes with loss 12.61 to 9.15,
normal gradient norms, and 23.81 GB peak allocation.

### Per-boundary AllToAll attribution

The 48-call sequence is deterministic enough to map every profiled AllToAll to
its layer boundary and semantic phase. A checked-in analyzer performs this
mapping for baseline order (`B combine`, `B dispatch`, `F' dispatch`, `F'
combine`) and ILP order (`B combine`, `F' dispatch`, `B dispatch`, `F'
combine`). It reproduces the accepted 43.522 ms aggregate delta exactly:

| Cross-layer stage | AllToAll delta |
| --- | ---: |
| B combine | +3.579 ms |
| F' dispatch | +16.255 ms |
| B dispatch | +6.035 ms |
| F' combine | +16.898 ms |
| Seven boundaries | +42.767 ms |
| Other 20 calls | +0.755 ms |

The largest individual taxes are B6/F'5 combine (+4.116 ms), B4/F'3 combine
(+3.947 ms), B5/F'4 replay dispatch (+3.781 ms), B2/F'1 combine (+3.565 ms),
and B3/F'2 replay dispatch (+3.472 ms). The inflation is therefore distributed
across both replay collectives and their neighboring native collectives rather
than being one duplicated or anomalous call.

One deterministic admission experiment delayed F' dispatch until the first
`GmmFunction.backward` completed on each rank. It passed 38 unit tests and a
two-step EP=4 512-token smoke run, but failed the 16384-token performance gate:
the AllToAll delta rose to 84.750 ms and speedup fell to 3.96 percent. Replay
dispatch, B dispatch, and replay combine respectively inflated by 26.788,
15.894, and 34.863 ms. A local device event is deterministic within one rank,
but the triggering GMM finishes at different times across ranks; the resulting
arrival skew makes early ranks wait in HCCL and delays every following
collective. The experiment was reverted. Future admission decisions must be
rank-consistent and are evaluated against per-rank arrival spread, not rank-0
duration alone.

### Replay stream and communicator ownership

Fresh all-rank traces show that the former execution-domain assignment coupled
two independent choices. Replay communicators alternated correctly so a native
backward never shared collective order with the following replay, but replay
streams alternated as well. Consequently F'5, F'3, and F'1 ran on the native
autograd stream. Their all-rank boundary envelopes improved by only 8.4--9.0 ms,
while the four replays on the sidecar stream improved by 26.1--31.8 ms.

The scheduler now keeps communicator alternation but runs every replay on the
sidecar stream. Reusing that stream requires an explicit allocator-lifetime
handoff. Without it, a 512-token smoke run passed but the 16384-token gradient
norm grew to 8.28e10 and then infinity: F'(k-1) could reuse storage while Bk was
still consuming tensors saved by F'k. At join, the controller now records every
tensor in `native_frame.recomputed[graph_task_id]` on the native backward stream
after that stream waits for replay completion. This keeps the saved storage live
until its actual consumer finishes.

The corrected EP=4 16384-token all-rank run has normal loss and gradients and
reduces rank-0 mean step time from 1417.137 ms to 1286.497 ms, a 9.22-percent
improvement. All seven boundary envelopes are now 44.6--47.4 ms, versus
72.5--75.4 ms in baseline. Peak allocation is 26.94 GB versus 25.54 GB.
AllToAll aggregate inflation rises from 45.925 ms in the alternating-stream
control to 65.111 ms because three additional boundaries now perform real
AI/HCCL overlap. The remaining communication tax is therefore no longer caused
by accidentally serialized replay. Rank arrival spread still dominates replay
dispatch, while post-arrival tails show a secondary physical HCCL/memory-bandwidth
contention cost. Further throttling must preserve the new 9.22-percent critical-
path gain rather than restoring same-stream serialization.

A no-extra-collective admission control started replay attention when the native
B-combine acquired its existing FIFO ticket, while keeping replay dispatch
behind B-combine completion. The intent was to use B-combine as a natural
rank-consistent synchronization anchor. It passed 33 scheduling tests and an
EP=4 512-token smoke run, but failed the 16384-token performance gate. AllToAll
inflation improved by only 1.069 ms, from 65.111 to 64.042 ms, while mean step
time regressed by 13.294 ms, from 1286.497 to 1299.791 ms. AI/HCCL overlap rose
slightly from 183.327 to 184.615 ms, confirming that the earlier start merely
moved replay attention onto B-combine's memory-bandwidth interval; it did not
remove the attention-dependent rank arrival skew at replay dispatch. The code
was reverted. Existing-collective anchoring is not sufficient unless the
admitted operation itself is ready on every rank.

### GMM-internal compute-aware admission experiment

A finer-grained prototype published a replay-dispatch completion event directly
after the asynchronous AllToAll `Work.wait()` and before post-dispatch token
sorting. The first `GmmFunction.backward` consumed that event between its
`grad_input` and `grad_weight` grouped matmuls; the second GMM remained
unrestricted. This preserved the useful replay-dispatch/`grad_input` overlap and
targeted only the roughly 1.2 ms dispatch tail that overlapped `grad_weight` in
the accepted 16384-token trace. The prototype passed 37 scheduling tests and an
EP=4 512-token two-step smoke run with gradient norms 75.20 and 40.66.

It failed the long-sequence gate. An exclusive 16384-token run was unavailable
because an unrelated VLLM job occupied about 30 GB on each device; the attempted
run failed in the baseline CE allocation before profiling. A paired EP=4
12288-token run on the same devices compared commit `25d097ad` with the
prototype. Rank-0 mean profiler step time regressed from 1057.613 ms to 2789.784
ms, AllToAll time increased by 270.465 ms per step, and the candidate gradient
norm peaked at 140.81. Its worst profiled step spent 4325.680 ms device-idle out
of 6551.544 ms, while the last step recovered to 1025.623 ms. This is rank-arrival
amplification, not additional GMM work: an early rank blocks its native autograd
thread at `grad_weight` until the slowest rank reaches replay dispatch.

The code was reverted. A native B GMM must never wait for a local replay
collective completion signal, even when the wait is placed at an exact kernel
boundary. Future compute-aware control must keep native autograd non-blocking
and make replay yield before it enters the collective, or use a backend admission
mechanism whose decision is rank-consistent without adding another collective.
The observed GMM/HCCL contention tail is smaller than the barrier cost it would
take to remove with a completion wait.

### Device-notify feasibility boundary

Ordinary NPU events cannot safely encode a dependency before its producer has
recorded the event: a 100-iteration wait-before-record stress test observed 61
ordering failures. CANN `aclrtNotify` does support this ordering; same-thread,
cross-host-thread, wait-before-record, and record-before-wait stress tests all
pass. The ILP scheduler therefore has an experimental
`VEOMNI_ILP_DEVICE_NOTIFY=1` path for pre-wired device dependencies.

The rolling stream topology imposes an important constraint. B submits its
Python callback on the primary stream, while alternating replay layers also use
that stream. Enqueuing notify wait before notify record on one stream creates a
self-deadlock. The implementation binds native and replay stream handles before
creating notifies, uses notifies only for cross-stream pairs, and retains the
record-before-wait event path for same-stream pairs. Notify handles are created
only for eligible pairs and are destroyed after a device completion event reports
that both sides have consumed them.

This path is intentionally disabled by default. On the same device-12--15 EP=4
16384-token four-window comparison, replacing all eligible stage dependencies
and the rolling ReduceScatter dependency regressed the mean step from 1321.456 ms
to 1345.093 ms. Restricting notify to the four AI/HCCL phase dependencies still
measured 1342.449 ms, a 20.993 ms regression, while overlap changed from 108.767
ms to 107.384 ms and AllToAll aggregate improved by only 0.544 ms. The original
host phase-launcher waits total just 0.215 ms per step, so this partial conversion
cannot recover enough host time to offset its device queue perturbation. A useful
device DAG must remove the seven per-layer launch/join admissions as one unit;
replacing already-short phase publication waits in isolation is counterproductive.

### Full-window device DAG experiment

A full-window prototype replaced the seven rolling launch/join admissions with one
persistent replay-worker launch. It assigned every replay layer a distinct NPU
stream and EP communicator, connected all phase dependencies with
wait-before-record `aclrtNotify` edges, and performed one host wait after the
worker had captured all seven checkpoint graphs. A window-wide device FIFO was
also required for MoE AllToAll and FSDP ReduceScatter ownership; pair-local FIFOs
deadlocked when packed-sequence imbalance allowed different ranks to enter
different communicators.

The prototype passed 40 CPU scheduling tests and an EP=4 512-token two-step NPU
smoke run. The smoke produced loss 12.61 to 9.16, gradient norm 75.17 to 41.64,
and 23.81 GB peak allocation. It did not pass the 16384-token performance gate:
the first training step did not complete under the stronger rank timing skew.
The strongest remaining cycle candidate is the set of eagerly submitted replay
FSDP AllGathers, which are outside the window-wide MoE/ReduceScatter FIFO.
Serializing only the later collectives does not make the complete device graph
rank-consistent.

There is therefore no valid throughput result for this prototype, and the code
was rolled back. The experiment establishes that host admission is currently a
correctness boundary, not just a 0.325 ms scheduling cost. A future full device
DAG must include FSDP AllGather, MoE AllToAll, and FSDP ReduceScatter in one
rank-consistent admission topology while keeping unshard storage bounded; simply
pre-enqueueing all replay layers is not viable.

### Memory-parity replay admission

The full seven-pair 16384-token window peaked at 26.83 GB versus 25.54 GB for
native FSDP. The scheduler now explicitly reshards the current expert FSDP
module at its GMM backward boundary before admitting replay experts, so its
parameter lifetime cannot extend into the next expert phase. The measurable
full-window peak, however, moved to replay dispatch: it materialized an
expert-major sorted tensor while the current expert backward still owned its
large temporaries. The dispatch path now releases its HCCL ticket after
`Work.wait()`, defers sorting until the current expert boundary, and fences the
following native dispatch with a device event. This reduced the full-window peak
from 26.81 GB to 26.57 GB.

The remaining peak came from replay work at the bottom of the window after more
FSDP ReduceScatter gradient buffers had accumulated. A deterministic tail
admission sweep measured 26.20 GB with six replay pairs and 25.34 GB with five.
The accepted eight-layer Qwen3-MoE piercing configuration therefore uses
`window_size=5`: layers 7 through 2 retain ILP replay, while the final two layer
pairs use native non-reentrant checkpoint recomputation.

A fresh same-device EP=4 eight-step comparison, with profiler memory sampling
disabled and the first two profiler steps excluded, measured 1408.493 ms for
baseline and 1326.731 ms for memory-parity ILP. Peak allocation was 25.55 GB for
both runs. Memory-parity ILP therefore reduced step time by 5.80 percent and
improved throughput by 6.16 percent without increasing peak memory. The maximum
reported gradient norm was finite at 77.63.

### EP8 throughput recovery

The full seven-pair window was also evaluated with EP=8 on devices 4 through 11,
16384-token packed sequences, two epochs, and a six-step profiler window. The
first two profiler steps were excluded as warmup. Native FSDP averaged
1725.686 ms per steady step; ILP at commit `1ababdba` averaged 1566.883 ms.
This is a 9.20 percent step-time reduction and a 10.13 percent throughput gain.
Every captured ILP step was faster than its baseline counterpart. Peak allocation
increased from 15.61 GB to 16.75 GB, so this configuration restores the throughput
target but is not a memory-parity result.

A CANN fused `GMM -> AllToAllv` replay-combine prototype was validated separately
for uniform and non-uniform routing. Forward output, expert-input gradient, and
weight gradient matched the standard path exactly, and a Qwen3-MoE EP=8 smoke run
completed with native non-reentrant checkpoint metadata. It nevertheless regressed
the unfused ILP steady step from 1541.381 ms to 1694.960 ms, or 9.96 percent. The
fusion couples the second expert GMM to combine communication and removes the
scheduler's freedom to place them against complementary native phases. The fused
prototype was therefore reverted; operator fusion is not used by the accepted
10.13 percent path.

### Historical-runtime EP8 rerun

The EP=4 throughput-optimal runtime at `dace8d57` was restored on `lip_dev` by
reverting the backward-boundary expert reshard and replay-dispatch lifetime
fence. The restored runtime files are byte-identical to `dace8d57`; later
experiment records remain in this document. The focused validation suite passed
with 51 tests and four NPU-only skips.

Two exclusive EP=8, 16384-token A/B rounds were then run on devices 4 through
11. Each run used eight training steps, profiled steps 2 through 8, and excluded
the first two profiler samples as warmup:

| Round | Baseline | Historical runtime ILP | Step reduction | Throughput gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1691.274 ms | 1593.539 ms | 5.78% | 6.13% |
| 2 | 1702.566 ms | 1572.900 ms | 7.62% | 8.24% |
| Mean | 1696.920 ms | 1583.219 ms | 6.70% | 7.18% |

Peak allocation was 15.61 GB for baseline and 17.33 GB for the historical
runtime ILP, a 1.72 GB increase. The ILP path retained 48 AllToAll calls per
captured step. Its second-round absolute step time was only 0.38 percent slower
than the previously accepted 1566.883 ms result, so there is no evidence of a
large ILP execution regression. Most of the difference from the former 10.13
percent throughput claim comes from baseline variance: that comparison used a
single 1725.686 ms baseline, 1.69 percent slower than the new two-round baseline
mean. The historical runtime also adds 0.58 GB over the accepted EP=8 ILP peak,
which increases allocator pressure and ILP run-to-run variance. The repeatable
EP=8 claim for this restored version is therefore 6.13--8.24 percent, with a
two-round mean of 7.18 percent.

### Selected EP8 release runtime

The historical-runtime rerun above is retained as a variance study, but it is
not the selected release configuration. `lip_dev` restores the validated
`1ababdba` runtime behavior at commit `a640cfb5`, including expert reshard at the
backward boundary and the replay-dispatch memory-lifetime fence. The accepted
EP=8, 16384-token result remains a 10.13 percent throughput gain, with peak
allocation increasing from 15.61 GB to 16.75 GB, or 1.14 GB. The slower fused
replay-combine prototype remains reverted.

### 24-layer EP8 scaling

A separate checkpoint was trimmed from the complete 48-layer weight set to
layers 0 through 23. It contains 9435 tensors in six safetensor shards and sets
`num_hidden_layers=24`. A two-step EP=8, 512-token smoke comparison completed
for baseline and the full 23-boundary ILP path. Both produced loss 12.17 then
9.15; baseline gradient norms were 84.20 and 41.47, while ILP measured 84.40 and
41.49. Both peaked at 32.22 GB.

The formal EP=8, 16384-token run used devices 8 through 15, eight training
steps, and the same six-sample profiler window as the eight-layer experiment.
The first two profiler samples were excluded:

| Configuration | Steady step | Step change | Throughput change | Peak allocation |
| --- | ---: | ---: | ---: | ---: |
| Native FSDP | 4332.324 ms | - | - | 34.21 GB |
| ILP, top 7 boundaries | 3785.530 ms | -12.62% | +14.44% | 34.22 GB |
| ILP, all 23 boundaries | 12402.065 ms | +186.27% | -65.07% | 35.56 GB |
| ILP, full depth with 7-boundary chunks | 3462.795 ms | -20.07% | +25.11% | 35.55 GB |
| ILP, all 23 with completion backpressure | 3478.301 ms | -19.71% | +24.55% | 35.56 GB |

All five configurations retained 144 AllToAll calls per captured step, so the
full-window regression is not duplicated replay. Baseline and the seven-pair
window accumulated roughly 0.94--1.15 seconds of AllToAll per steady step. The
23-pair window inflated the same collectives to 5.20--7.10 seconds as rank
arrival skew and physical HCCL contention compounded across the deeper replay
window.

The first deep-window repair used deterministic chunk admission. Seven
consecutive boundaries used ILP, then one boundary fell back to native
non-reentrant checkpoint recomputation to drain both replay execution domains.
This established that unfinished device pairs, rather than a semantic eighth-
layer boundary, caused the jump. It was only a diagnostic workaround: the
number seven came from the previously validated eight-layer window and is not a
model-independent scheduling rule.

The chunked run retained 144 AllToAll calls per step and reduced their steady
aggregate duration to 0.97--1.26 seconds. Its saved EP dispatch plans avoided
one count-128 token-count metadata AllGather at each replay boundary, reducing
total AllGather calls from 97 to 76 per step; FSDP parameter AllGather counts
were unchanged. Every sampled chunked step was faster than its baseline
counterpart.

The accepted repair replaces chunking with completion-backed admission. Each
rolling join records one event after the native stream and replay stream have
both reached the pair boundary. Before launching the next sidecar, the host
queries that event and synchronizes only while it remains unfinished. No replay
boundary is skipped, no extra collective is added, and the number of queued
unfinished pairs is bounded independently of model depth. The same state
machine therefore covers 8, 24, 48, or more layers.

All 23 replay boundaries were active in the accepted run. AllGather fell to 74
calls per step, while AllToAll remained at 144 calls and accumulated 0.95--1.38
seconds. The 23 completion-backpressure markers per sampled step spent
0.71--1.33 seconds waiting for useful pair completion instead of allowing that
work to accumulate across deeper layers. The generic path is only 0.45 percent
slower than the empirical chunk workaround and retains a 24.55 percent
throughput gain over native FSDP. Peak allocation is 1.35 GB above baseline.

Layer selection is also model-derived rather than tied to the original
eight-layer experiment. The default `current_layer=-1` and `window_size=0`
resolve to the final decoder layer and every available boundary. At each
boundary, the replay layer is derived as `backward_layer - 1`. Unit coverage
verifies the same resolution for 8, 24, and 48 layers, while explicit partial
windows remain supported. The 24-layer runner contains no decoder-layer
indices; its startup log resolves the model to layers 0 through 23 before
installing the scheduler.

### 48-layer EP16 scaling and memory boundary

The complete Qwen3-MoE 30B checkpoint was restored to all 48 decoder layers and
trained with full parameters on 16 NPUs. The comparison used FSDP2, EP=16,
16384-token packed sequences, global batch size 16, eight optimizer steps, and
rank-0 profiling from steps 2 through 8. The first two captured samples were
excluded, leaving four steady samples for each reported mean. Baseline and ILP
used the same checkpoint, input batches, optimizer, and profiler settings.

| Configuration | Active replay boundaries | Steady step | Step reduction | Throughput gain | Peak allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native FSDP | 0 | 8876.243 ms | - | - | 34.50 GB |
| ILP, full window (`window_size=47`) | 47 | 7544.779 ms | 15.00% | 17.65% | 35.76 GB |
| ILP, bounded window (`window_size=20`) | 20 | 8298.971 ms | 6.50% | 6.96% | 34.70 GB |

The full window therefore exchanges 1.26 GB of additional peak allocation for
the highest measured throughput. The 20-boundary window reduces that premium
to 0.20 GB while retaining a 6.96 percent throughput gain.

An initial profiler-disabled three-step scan was not sufficient to establish
memory parity: it reported 34.49 GB through `window_size=20`, but later packed
batches raised the peak. The scan was therefore repeated for all eight steps
at representative boundary points, with a fresh eight-step profiler-disabled
baseline control:

| Configuration | Active layer range | Eight-step peak | Versus matching baseline |
| --- | --- | ---: | ---: |
| Native FSDP, profiler disabled | - | 34.49 GB | - |
| `window_size=1`, profiler disabled | `[46, 47]` | 34.52 GB | +0.03 GB |
| `window_size=12`, profiler disabled | `[35, 47]` | 34.56 GB | +0.07 GB |
| `window_size=18`, profiler disabled | `[29, 47]` | 34.57 GB | +0.08 GB |
| `window_size=19`, profiler disabled | `[28, 47]` | 34.56 GB | +0.07 GB |
| Native FSDP, formal profiler run | - | 34.50 GB | - |
| `window_size=20`, formal profiler run | `[27, 47]` | 34.70 GB | +0.20 GB |

The minimum nonzero window already exceeds its matching baseline by 0.03 GB,
or about 0.09 percent. Consequently, the current implementation has no
nonzero `window_size` that can guarantee strict peak-memory parity in this
configuration. `window_size=0` means automatic full-depth selection rather
than disabling replay, so strict parity requires disabling ILP. If allocator-
rounding-level tolerance is acceptable, `window_size=1` is the closest active
configuration measured here.

Communication counts explain what the speedup does and does not come from. All
three formal runs execute 288 MoE AllToAll operations and 49 FSDP
ReduceScatter operations per captured step. FSDP parameter AllGather counts
are also unchanged: every run has 96 calls with count 1,196,304 and one call
with count 38,895,744. ILP therefore does not obtain this result by skipping
parameter materialization or expert data exchange.

The count-128 EP token-count metadata AllGather is the only removed collective.
It falls from 96 calls per baseline step to 49 with all 47 replay boundaries,
and to 76 with the 20-boundary bounded window. Each accepted replay uses
the dispatch plan saved by the original forward instead of rebuilding that
plan during checkpoint recomputation, so the reduction is exactly one metadata
AllGather per active boundary. Total AllGather counts are consequently 193,
146, and 173 for baseline, full-window ILP, and bounded-window ILP.

The remaining gain comes from moving the selected recompute forward work off
the serial native-backward critical path and admitting it against complementary
AI Core/HCCL phases of the preceding layer. Completion-backed admission keeps
unfinished pairs bounded as depth increases, while saved forward inputs and
dispatch plans let the completed replay replace, rather than duplicate, native
non-reentrant checkpoint recomputation. The full window exposes more overlap
opportunities and removes more metadata collectives; the 20-boundary window
retains 6.96 percent throughput improvement with a 0.20 GB formal-run peak
premium.

Reproduction artifacts are `qwen3_30b_48l_ep16_baseline.sh` and
`qwen3_30b_48l_ep16_ilp.sh`. Formal logs are
`log_qwen3_48l_baseline_r2_ep16_16k.txt`,
`log_qwen3_48l_ilp_w47_ep16_16k.txt`, and
`log_qwen3_48l_ilp_w20_ep16_16k.txt`; their corresponding trace directories use
the same stem under `trace/`. The complete memory controls use the `_mem8.txt`
suffix, while the preliminary three-step scans use `_mem.txt`.

## Remaining work

1. Repeat exclusive EP=4 16384-token A/B in the opposite launch order and report
   the combined median and confidence interval.
2. Measure every native B phase against baseline and enforce a less-than-one-
   percent regression guard for dispatch, attention, and ReduceScatter.
3. Evaluate a completion-credit window larger than one without reintroducing
   unbounded device queue growth. Credits must be returned by completed pair
   events, not decoder-layer indices, and must preserve collective order across
   ranks.
4. Attribute the remaining 1.52 GB transient peak to individual NPU operator
   workspaces and test workspace reuse without extending tensor lifetimes.
5. Add runtime counters for four-stage event waits and join tails so performance
   regressions are visible without a full profiler capture.
6. Add a multi-node control-group fallback and policies for non-MoE layers and
   `ep_size < world_size`.
