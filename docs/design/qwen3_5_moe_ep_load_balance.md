# Qwen3.5-MoE expert-parallel load balancing

## Scope and configuration

Temporary expert replicas can redistribute a Qwen3.5-MoE per-forward EP hotspot.
This changes fused execution, not the router auxiliary loss or persistent model
structure. The public switch `train.moe_ep_load_balance.enabled` defaults to
`false`; `max_replicas_per_rank` defaults to `1`.

Enabled mode requires full, non-LoRA Qwen3.5-MoE SFT, compatible merged experts,
and the following settings:

| Owner | Requirement |
| --- | --- |
| `model.accelerator` | `ep_size > 1`, resolved `dp_replicate_size == 1`, `ep_outside=false` |
| `model.accelerator.fsdp_config` | `fsdp_mode=fsdp2`, `offload=false` |
| `model.accelerator.offload_config` | `enable_activation=false`, `enable_async_activation=false` |
| `model.ops_implementation` | `moe_implementation=fused_triton` on CUDA or `fused_npu` on Ascend |
| `train.checkpoint` | `load_path=null` (resume unsupported) |

The expert count must be divisible by EP size, and the positive replica-slot
count cannot exceed the original local expert count. Invalid configuration,
module layouts or plans fail before replica communication; there is no eager or
alternate-backend fallback. HSDP, EP-outside, CPU/activation offload, resume,
LoRA, eager/Quack MoE and other model families are unsupported. Multi-node and
other SP/EP topology combinations remain untested. Ordinary checkpoint saving
and `model.accelerator.gradient_checkpointing` remain allowed.

[The validation report](../performance/qwen3_5_moe_ep_load_balance.md) records
GPU observed-mean non-degradation, native NPU precision/load redistribution,
and the remaining NPU throughput regression. Measurements retain their own
source, hardware and software identities; they are not universal guarantees. Executable opt-in, comparison and
profiling recipes belong in the [training guide](../examples/qwen3_5.md#qwen35-moe-ep-load-balance-experiments).

## Ownership and forward lifecycle

[`VeOmniModelRuntime`](../../veomni/models/model_runtime.py) owns attachment in
its training constructor, after EP/FSDP2 wrapping and weight loading but before
optimizer construction, inside `use_parallel_state(self.model_name)`. It
attaches one private controller per compatible merged expert module. The
trainer owns job callbacks, not model construction or attachment.

1. Native top-k yields logical expert IDs and routing weights.
2. The controller gathers actual selected-ID histograms across its EP group.
3. The integer planner chooses replicas that strictly improve the
   descending-sorted physical rank-load vector, lexicographically.
4. Dispatch assigns concrete occurrences to physical aliases, transfers
   temporary weights and invokes the backend's fused EP execution.
5. Backward returns replica weight gradients to the original owner before its
   optimizer update.

The [planner/controller](../../veomni/distributed/moe/ep_load_balance.py) owns
the immutable per-forward plan. Shared dispatch validation owns group identity,
merged-weight layout and plan invariants; CUDA/NPU dispatchers own backend
execution. No expert forward hook replaces routing, and no cross-forward plan
or weight-value cache is kept. Router replay is a test seam after native top-k, not a
production routing policy.

The objective admits `[100,100,0,0] -> [50,100,50,0] -> [50,50,50,50]`:
the first move is useful even though max-minus-min spread has not yet changed.
Equal sorted profiles are rejected. Search stops exactly when the rank-load
range is at most one, the integer-optimal profile for a conserved token total.
This is a token-load objective, not a communication-cost or speedup guarantee.

## Dispatch invariants

Let `E` be logical experts, `R` EP size, `L=E/R` original rows per rank,
`K` replica slots per rank and `S=L+K` physical stride.

- Logical expert `e` maps to `floor(e/L)*S + (e mod L)`.
- Replica slot `s` on rank `r` maps to `r*S + L+s`, `0 <= s < K`.
- With no replica, dispatch retains the original IDs and expert count.

Validation checks integer IDs, local histogram agreement, selected-occurrence
conservation, split totals, unique replica slots, alias formulas and matching
group/expert dimensions. Native per-token top-k uniqueness is retained;
across-token repetitions remain valid.

CUDA gathers in logical expert order, applies a one-to-one physical occurrence
permutation for all-to-all, and restores logical order before combination.
Stable physical sorting alone does not preserve top-k output or duplicated-input
gradient summation order. Reused chunk-row maps only copy values; they perform
no floating-point reduction. NPU and disabled dispatch keep their native paths.

## Temporary weights and backward invariants

Merged gate/up and down rows move together through asynchronous P2P on the EP
group. Temporary rows are tensors, not parameters: parameter/state-dict names,
optimizer groups/state and checkpoint format are unchanged. P2P send and receive buffers are contiguous even for transposed GEMM
operands. CUDA returns ordered FP32 gradient prefixes; NPU returns replica
inputs/upstream gradients to reconstruct native owner GEMM operands.

CUDA replica weight gradients stay FP32 inside the fused backward boundary
until the final owner cast. Independent FP32 partial sums would still change
the original reduction tree. Replicas instead continue consecutive token-prefix
accumulators in original order; owners continue with the suffix. Quotas align
to the least common multiple of weight-gradient `BLOCK_K` values, and every
segment uses the original owner's token count for kernel selection. Histogram
collection also checks tile-configuration agreement across EP ranks.

Affected owner suffixes are computed only once. The ordered down-projection
chain submits its first stage before independent gate derivatives/input-gradient
work, then waits; down communication completes before gate/up communication.
Pending handles retain operands and buffers, support idempotent waits and drain
in `finally` when independent work raises.

These rules preserve the tested arithmetic order, not unrestricted bitwise
identity across every dtype, shape or backend. FP32 P2P doubles low-precision
gradient bytes, and ordered chains serialize segments. Better load balance
does not establish net throughput or memory benefit.

### Experimental memory path

`train.moe_ep_load_balance.optimize_memory=true` requires balancing enabled
and defaults to `false`. It changes storage/compute boundaries, not planning,
token order, replica quotas or the CUDA ordered FP32 weight-gradient chain.

- CUDA grouped GEMMs read original and replica weight tensors in one launch,
  avoiding their full weight concatenation while retaining the existing tile
  selection and accumulation instructions. Strided original weights retain
  the legacy concatenation path; that layout does not receive the memory optimization.
- NPU reuses its native grouped matmul over original and replica token
  segments. Segment boundaries come from the plan's existing CPU counts.
  Only activations are concatenated; empty segments keep zero gradients
  without launching an empty native GMM. A shared autograd boundary returns
  replica X/dY prefixes in original sender/token order and computes each owner
  weight gradient with the original native grouped matmul. Independent FP32
  partial sums are not numerically interchangeable with that reduction.
  Original owner counts are retained in pinned CPU storage for asynchronous
  device upload; batched P2P posting preserves peer and operand order.
- Each layer owns a bounded idle-storage pool (two tensors total). Transfers
  refresh weights every forward. Live/retained autograd graphs retain their
  leases; CUDA/NPU events and allocator stream tracking protect later reuse.
  Concurrent graphs can allocate more than two live tensors. No-grad buffers
  can return after submission; later reuse waits on their recorded events.

The pool is not a parameter, optimizer state, checkpoint entry or weight-value
cache. It introduces no DeepEP/MoonEP dependency, VMM mapping or global
allocator change. CPU/Gloo tests exercise storage lifetime, empty/frozen
paths and gradient return. Actual CUDA/NPU validation is reported separately
in the evidence report; untested combinations remain untested. NPU segmentation adds launches and activation
copies, so lower weight-copy volume does **not** establish faster training or
lower peak memory. See the [hardware comparison protocol](../examples/qwen3_5.md#experimental-memory-candidate).

## Telemetry and artifacts

Logical router histograms sum every distinct rank-local token slice over the
cached DP+SP/FSDP group. Physical plans are already EP-global: only EP-local
rank 0 contributes them, while every rank joins monitoring collectives. This
prevents EP-size inflation of absolute moved-token/replica/heatmap counts.
Physical telemetry counts executions, including activation-checkpoint
recomputation forwards; its absolute totals are not unique training-token counts.

For physical loads `x` over `R` ranks, imbalance is zero if their sum is zero;
otherwise it is `mean(abs(R*x/sum(x)-1))`. Moved fraction is
`moved_tokens/total_routed_tokens`, or zero for an empty denominator.
Acceptance telemetry keys are:

- `moe/ep_rank_imbalance_before/avg`
- `moe/ep_rank_imbalance_after/avg`
- `moe/ep_active_replicas/sum`
- `moe/ep_moved_tokens/sum`
- `moe/ep_moved_token_fraction/avg`

The callback requests heatmaps only for an enabled WandB image sink.
`MoERouterMonitor.compute_metrics(..., render_heatmaps=False)` retains scalar
values, collectives, cadence and reset; direct callers render images by default.

The [comparison reporter](../../scripts/profile/compare_moe_ep_load_balance.py)
accepts strict schema-version-1 envelopes, flat JSON, per-step JSONL or plain
loss/gradient logs. It rejects nonfinite/misaligned curves and inconsistent
critical metadata. Precision uses `abs(delta) <= atol + rtol*abs(baseline)`;
undefined Pearson correlation is unavailable with a reason. Timing, throughput,
P2P and memory sections require explicit fields on both sides. Missing
measurements remain unavailable, never zero or inferred values.

## CI proof and rollback

Unit workflows run planner/executor/dispatch/config/monitor/reporter and E2E
gate contracts without accelerator training. The separate GPU/NPU E2E
workflows select the exact matched-training node in strict prerequisite mode
(`VEOMNI_REQUIRE_EP_BALANCE_E2E=1`). CUDA requires four NVIDIA SM70+ devices
and its declared imports; NPU requires four devices, `torch_npu` and the
`triton` import supplied by `triton-ascend`. Missing prerequisites fail those
jobs. Ordinary local invocation skips before fixture creation.
Forced hotspots, two steps and blocking launches prove function/limited
precision, not throughput. NPU's separate two-rank alias/gradient node is
launched explicitly with torchrun in unit CI.

Rollback is `train.moe_ep_load_balance.enabled=false` and
`train.moe_load_balance_monitor_interval=0`; checkpoints need no conversion.
Set `train.moe_ep_load_balance.optimize_memory=false` first if it was enabled.
To roll back only the memory candidate, leave balancing enabled and set this
memory flag to `false`.
Code rollback must remove only scoped feature changes, not reset unrelated
work or hand-edit generated modeling files.
