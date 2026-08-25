# Qwen3.5-MoE expert-parallel load balancing

## Scope

VeOmni can optionally reduce a per-forward Qwen3.5-MoE expert-parallel hotspot by placing temporary copies of selected expert weights on less-loaded EP ranks. The default remains disabled:

```yaml
train:
  moe_load_balance_monitor_interval: 0
  moe_ep_load_balance:
    enabled: false
    max_replicas_per_rank: 1
```

This is a fused-EP execution feature, not the router auxiliary loss and not a persistent model transformation. Enabling it currently fail-fast requires full Qwen3.5-MoE SFT with FSDP2, resolved `dp_replicate_size == 1` (no HSDP), `ep_outside=false`, no FSDP CPU offload, no activation offload, no checkpoint resume (`train.checkpoint.load_path=null`), `ep_size > 1`, and either CUDA `fused_triton` or Ascend `fused_npu`. Multi-node, other SP/EP topologies, eager MoE, Quack MoE, and other model families remain outside the validated support surface.

## Runtime path and ownership

After model loading, Qwen3.5's normal `ParallelPlan` shards the original expert dimension across EP ranks and FSDP2 wraps the model. Only then does `BaseTrainer` attach one private load-balancer controller to each compatible merged `Qwen3_5MoeExperts` module. Attachment validates the real EP process group, the global/local expert-row relationship, and the fixed replica-slot count.

For every fused MoE forward:

1. The native Qwen3.5 router produces weights and logical top-k expert IDs.
2. The planner computes a histogram from those actual selected IDs, then gathers the histogram across the existing EP group. It does not predict load from a prior step or add a second routing policy.
3. A deterministic greedy/water-filling plan selects only replicas that strictly reduce the physical rank-load spread. It conserves selected occurrences and assigns concrete occurrences of a hot logical expert to each replica.
4. Dispatch rewrites only those selected IDs into a private physical-alias namespace, transfers temporary weights, and runs the existing fused EP dispatcher.
5. The custom autograd boundary returns every replica gradient to the original expert owner before the optimizer sees the original parameter gradient.

The generic trainer owns lifecycle and validation; the planner owns the immutable per-forward plan; CUDA and NPU dispatchers own backend-specific fused execution. The monitor observes routing and physical-plan telemetry. No expert forward hook replaces routing. The synthetic E2E uses the existing router-replay seam, after native top-k, and leaves Qwen3.5's own gather and renormalization of routing weights in place.

## Physical alias namespace

Let `E` be the number of logical experts, `R` the EP size, `L = E / R` the original experts per rank, `K` the fixed `max_replicas_per_rank`, and `S = L + K` the physical stride.

- Original logical expert `e` maps to physical ID `floor(e / L) * S + (e mod L)`.
- Replica slot `s` on target EP rank `r` maps to `r * S + L + s`, for `0 <= s < K`.

The namespace keeps every rank's original rows first and fixed replica slots second. It exists only for the current forward. When the plan creates no replica, dispatch retains the original logical IDs and expert count.

The plan contains rewritten IDs, EP input/output splits, per-sender counts for every local physical row, replica ownership, moved-token counts, and before/after physical rank loads. Validation checks integer IDs, local histogram agreement, token conservation, unique target slots, alias formulas, and consistent group/expert dimensions before communication starts.

## Temporary weights and gradients

Qwen3.5's merged expert has two parameter tensors that must move together: the merged gate/up projection and the down projection. The dispatcher validates both tensors first, then starts two asynchronous point-to-point transfers over the same EP group. A target rank receives each copied row into a fixed temporary tensor and concatenates original rows followed by replica rows.

The concatenation is a custom `torch.autograd.Function`. During backward, a target rank sends the gradient of each temporary row back to its source rank. The source adds that contribution to the corresponding original local expert gradient. The temporary tensor is not an `nn.Parameter`; the custom function returns no gradient for it.

Consequently, model parameters, optimizer parameter groups/state, state-dict keys, and checkpoint format do not change. Checkpoint saving remains structurally unchanged. DCP resume with `train.moe_ep_load_balance.enabled=true` is intentionally rejected at configuration time via `train.checkpoint.load_path`. This statement describes the design and CPU/gloo structural tests only; it is not an accelerator checkpoint-resume support claim.

## Configuration and failure behavior

`train.moe_ep_load_balance.enabled` defaults to `false`; `max_replicas_per_rank` defaults to `1`. Enabling the feature requires:

- `train.accelerator.ep_size > 1`;
- `train.accelerator.fsdp_config.fsdp_mode=fsdp2`;
- resolved `train.accelerator.dp_replicate_size == 1` (HSDP unsupported);
- `train.accelerator.ep_outside=false`;
- `train.accelerator.fsdp_config.offload=false`;
- `train.accelerator.offload_config.enable_activation=false`;
- `train.checkpoint.load_path=null`;
- full, non-LoRA training;
- `model.ops_implementation.moe_implementation=fused_triton` on CUDA or `fused_npu` on NPU;
- compatible merged Qwen3.5 expert modules after EP sharding/FSDP2 wrapping;
- a positive replica count no greater than `num_experts / ep_size`; and
- a global expert count divisible by the EP size.

Invalid settings raise during argument validation or attachment. The implementation does not silently select eager MoE or another backend. HSDP, `ep_outside`, FSDP CPU offload, activation offload, and DCP resume are explicit enabled-only fail-fast boundaries rather than fallback paths. Checkpoint saving and ordinary gradient checkpointing remain allowed because they do not change the runtime load-balance contract. Qwen3.5's GatedDeltaNet backend is independent of MoE balancing: CUDA packed/varlen training uses the fused FLA path. The NPU toy E2E pins gated RMSNorm, causal conv1d, and chunk GDN to the executable `npu` kernels, which import `triton` from the `triton-ascend` distribution. It deliberately avoids the `npu_ascendc` chunk-GDN backend and its additional `fla_npu` dependency. `dyn_bsz=false` selects the non-dynamic batching path for both matched NPU runs; it is not an eager-kernel fallback.

## Telemetry

Set `train.moe_load_balance_monitor_interval=N` to emit an interval every `N` optimizer steps. Logical expert counts and physical plan data are reduced across the cached DP+SP/FSDP group that contains distinct token slices. Replicated EP siblings are deliberately excluded; reducing them would duplicate identical router counts.

For a physical rank-load row `x` with `R` ranks, normalized imbalance is

```text
imbalance(x) = 0                                      if sum(x) = 0
             = mean_r(abs(R * x[r] / sum(x) - 1))   otherwise
```

For each layer, moved-token fraction is `moved_tokens / total_routed_tokens`, or zero when the denominator is zero. The monitor publishes per-layer values, cross-layer `max`/`avg` values for imbalance and fraction, and `sum`/`max`/`avg` values for replicas, moved tokens, and total routed tokens. The frozen aggregate acceptance keys are:

- `moe/ep_rank_imbalance_before/avg`
- `moe/ep_rank_imbalance_after/avg`
- `moe/ep_active_replicas/sum`
- `moe/ep_moved_tokens/sum`
- `moe/ep_moved_token_fraction/avg`

Before/after physical-rank heatmaps are media values; scalar sinks must not serialize them as numbers. Every rank participates in the interval collectives, while only the formatting/logging rank publishes scalar/media output.

## Synthetic accelerator E2E

`tests/e2e/test_qwen3_5_moe_ep_load_balance.py` skips before dataset or checkpoint creation unless at least four compatible devices and required packages are available. CUDA must be NVIDIA rather than ROCm, each of the four selected devices must be SM70 or newer, and the `triton`, `fla`, `flash_attn`, and `liger_kernel` imports must resolve. NPU requires both `torch_npu` and the `triton` import provided by `triton-ascend`. It creates one seed-0 CPU-fp32, native-stacked toy checkpoint and reuses one dummy dataset for exactly two matched four-process FSDP2/SP=1/EP=2 jobs.

A stateless router-replay manager forces unique top-k IDs `[0, 1]` for every Qwen3.5 router. Both experts belong to EP rank 0 in the 16-expert, EP=2 toy. Baseline and enabled jobs use the same replay manager, checkpoint, data ordering, seed, dtype, batch sizes, gradient-checkpointing setting, backend, and two-step schedule. Only feature enablement, output directory, and run label differ. The temporary wrapper reuses `TestVLMTrainer`, replaces only its log sink, and restores the replay singleton in `finally`.

The test requires finite, equal-length loss and gradient-norm curves within `rtol=atol=0.1`; positive candidate replica/move telemetry; no step whose after-imbalance exceeds before-imbalance; and at least one strict improvement. `CUDA_LAUNCH_BLOCKING=1`, two steps, and a forced hotspot make this functional/precision coverage only. It provides no throughput, overlap, or memory conclusion.

Both accelerator E2E workflows invoke only the exact real-hardware test node `test_qwen3_5_moe_ep_load_balance_matched_precision_and_telemetry` in a step separate from the existing SP/EP grid and set `VEOMNI_REQUIRE_EP_BALANCE_E2E=1`. In that strict CI mode, every unsupported device type, insufficient device count, ROCm, insufficient CUDA capability, or missing-package prerequisite is a test failure rather than a skip, so a dedicated accelerator job cannot pass without executing the hardware proof. Static workflow/backend and local-policy gate tests remain in ordinary pytest without the strict variable; this prevents their intentional skip assertions from inheriting the hardware job's policy. A CPU or Mac still skips the real E2E before fixture materialization during ordinary local invocation, and that skip is not accelerator evidence.

## Comparison artifacts

`scripts/profile/compare_moe_ep_load_balance.py` compares canonical schema-version-1 JSON envelopes. It also normalizes flat `log_dict.json`, per-step JSONL, and plain loss/gradient training logs. Schema version is strictly integer `1` (not `true` or `1.0`). JSONL tracks whether metadata has been seen so an empty metadata object cannot hide a later change. Shared critical metadata uses type-sensitive equality. The reporter rejects missing, boolean, non-numeric, non-finite, or misaligned curves; each present `step_time_s`, `step_tokens`, `tokens_per_second`, `p2p_bytes`, and `p2p_wait_time_s` series must align with the same run's loss/gradient step count.

Precision reports maximum absolute/epsilon-safe relative error, repository-close and relative-threshold hit rates, and standard Pearson correlation. Correlation is unavailable, with a reason, for fewer than two points or zero variance. Optional timing, throughput, P2P, and peak-memory sections are available only when both artifacts explicitly contain the required values; absent values remain `null` with a side-specific reason. `p2p_bytes` reports warmup totals/means, steady totals/means, candidate-minus-baseline deltas, and candidate/baseline ratios. `p2p_wait_time_s` reports warmup/steady means, candidate-minus-baseline steady delta, and baseline/candidate speedup. Both are canonical per-step inputs exported by profiler post-processing or runtime instrumentation; the reporter does not collect or infer them. Deterministic JSON and Markdown expose the same available aggregates.

## Rollback

For an immediate runtime rollback, set:

```yaml
train:
  moe_load_balance_monitor_interval: 0
  moe_ep_load_balance:
    enabled: false
    max_replicas_per_rank: 1
```

This restores the pre-feature fused EP dispatch without changing the model checkpoint. For a code rollback, remove only the Qwen3.5 EP-balance planner/dispatcher/monitor integrations and their scoped tests/docs; do not edit generated modeling files or destructively reset an unrelated dirty worktree.
