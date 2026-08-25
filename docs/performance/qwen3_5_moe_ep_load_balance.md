# Qwen3.5-MoE EP load-balance evidence report

Status: local structural validation plus four-device Ascend A3 functional, precision, steady-state, memory, and profiler evidence

Date: 2026-08-22 (Asia/Shanghai)

Base revision: `a8a6e5418e435e006e873afce0c8cff31fe350ce`

## What this report establishes

Local validation establishes deterministic planning/executor/monitor behavior on CPU and real two-rank gloo collectives, plus the standalone artifact reporter's parsing and arithmetic. A separate four-device Ascend A3 run establishes the fused-NPU path for the synthetic Qwen3.5-MoE workload described below. CUDA/NCCL and production-scale behavior remain untested.

The four-device synthetic E2E is present in both accelerator workflows, but this Mac skips it before dataset/checkpoint materialization. Dedicated GPU/NPU workflow steps select only the exact real-hardware test node and enable strict proof mode, in which any device, backend, or package prerequisite failure fails the test instead of skipping it. Static/backend and local-policy tests run without that workflow-scoped environment. The A3 evidence is a forced-hotspot toy workload, not a production-scale claim.

## Local environment

- Host: macOS 26.5 arm64.
- Interpreter: Python 3.12.2 from `.venv/bin/python`.
- Packages recorded by the Task 6 baseline: PyTorch 2.13.0, Transformers 5.9.0; `torch_npu` unavailable.
- Accelerators: CUDA unavailable; no Ascend NPU. MPS is outside the distributed evidence scope.
- Distributed local evidence: CPU plus spawned two-rank gloo only.

## Exact local test ledger

| Validation set | Result | What it proves |
| --- | ---: | --- |
| Comparison reporter (`tests/scripts/test_compare_moe_ep_load_balance.py`) | 75 passed | Canonical/flat/JSONL/plain parsing; strict schema/curve/metadata validation; precision/correlation; optional timing/throughput/P2P/memory semantics; deterministic JSON/Markdown including warmup timing and P2P aggregates; CLI exit codes |
| Accelerator workflow/backend/gate contracts | 16 passed | Both dedicated workflows require strict hardware proof; local skip and strict-failure policies; NPU exact Ascend-index install, project-venv child PATH/order/no-resync contracts, `npu` GDN command and `triton-ascend` import gate; CUDA ROCm, per-device SM70, and Liger gates; mocked hardware boundary only |
| Accelerator E2E execution on this host | 1 skipped | Honest pre-materialization gate only; no model, precision, performance, or memory evidence |
| Tasks 1-6 combined CPU/gloo regression rerun after the final support-boundary fix | 113 passed, 3 skipped | Config and enabled-only topology/offload/resume rejection, planner, temporary transfer/autograd, CUDA/NPU dispatch seams, monitor formulas, and real gloo collective structure; accelerator-gated cases remain skips |
| Real two-device Ascend NPU alias/gradient test | 2 ranks passed | Fused-NPU alias forward/backward, routing-weight preservation, finite gate/down gradients, owner-gradient equality, and zero non-owner gradients |
| Real four-device Ascend NPU matched E2E | 1 passed in 103.87 s | Baseline and enabled two-step loss/gradient equality, real replica participation, positive moved-token telemetry, and strict EP-rank load improvement |

The final Task 7 combined rerun is recorded in the task implementation report. Counts above are never converted into accelerator claims.

## Evidence matrix

| Hardware/backend | Functional structure | Precision | Performance | Accelerator memory | Host memory | Status / note |
| --- | --- | --- | --- | --- | --- | --- |
| Apple CPU / gloo | pass | reporter arithmetic pass; no real Qwen3.5 training curve | untested | not applicable | untested | Structural/unit evidence only |
| CUDA / NCCL, fused Triton | untested | untested | untested | untested | untested | Requires the self-hosted four-device E2E and a separate matched steady run |
| Ascend A3 / HCCL, fused NPU | pass on toy EP=2/FSDP2 | 10/10 loss and gradient points exactly equal | 10-step matched run: +0.997% steady tokens/s after 2 warmup steps | +113,223,680 bytes peak allocated | untested | Synthetic forced-hotspot evidence on torch/torch_npu 2.9.0; not production scale |

Enabled-mode fail-fast exclusions: HSDP, `ep_outside`, FSDP CPU offload, activation offload, checkpoint resume, LoRA/PEFT, eager MoE, and Quack MoE. Untested topology/features beyond that fail-fast surface: multi-node and SP values other than the synthetic E2E's SP=1. The existing production Ascend recipe topology is not evidence for these cells.

## Precision acceptance

For every loss and gradient-norm point, the reporter applies the repository envelope

```text
abs(candidate - baseline) <= atol + rtol * abs(baseline)
```

and requires a 100% hit rate for both curves. It also reports maximum absolute error, epsilon-safe relative error, threshold-relative hit rate, and standard Pearson correlation. Fewer than two points or zero variance produce `null` correlation with a reason instead of a fabricated `1.0`.

The accelerator E2E uses `rtol=atol=0.1`, exactly two steps, one seed-0 checkpoint, one dataset/order, and one forced `[0, 1]` hotspot for both baseline and enabled runs. It additionally requires positive active-replica and moved-token telemetry, `after <= before` imbalance at every recorded point, and at least one strict improvement. Because this is a forced two-step run with `CUDA_LAUNCH_BLOCKING=1`, it is not a performance experiment.

## Performance and memory reporting contract

Performance is available only from explicit matched fields:

- `step_time_s` for warmup and steady mean time;
- `step_tokens` with `step_time_s` for aggregate `sum(tokens) / sum(time)`;
- `tokens_per_second` for explicit steady throughput;
- `p2p_bytes` for warmup/steady per-step communication volume, steady total/mean, candidate-minus-baseline deltas, and candidate/baseline ratios;
- `p2p_wait_time_s` for warmup/steady mean wait time, candidate-minus-baseline steady delta, and baseline/candidate speedup;
- `peak_accelerator_memory_bytes`; and
- `peak_host_memory_bytes`.

Every present per-step performance/P2P curve must align with its run's loss and gradient-norm step count. P2P curves come from profiler trace post-processing or explicit runtime export; the reporter itself neither instruments training nor infers communication. The reporter emits candidate-minus-baseline deltas and directionally defined ratios/speedups. Missing data on either or both sides yields `status: unavailable`, `null` numbers, and a side-specific reason. It never substitutes zero or samples system-wide used CPU memory.

## Ascend A3 matched result

The remote host exposed 16 `Ascend910_9382` devices; four idle devices were isolated for this run. The exact software stack was CANN 9.0.0, PyTorch 2.9.0, `torch_npu` 2.9.0, Transformers 5.9.0, and `triton-ascend` 3.2.1. This differs from the repository's torch/torch_npu 2.10 CI target and is therefore recorded as a separate compatibility point.

Both runs used the same seed-0 checkpoint and 80-sample order, bf16 FSDP2, world size 4, EP=2, SP=1, global batch 8, micro batch 1, sequence length 2048, four MoE layers, 16 experts, top-k 2, and the same forced `[0, 1]` hotspot. Profiling and checkpoint saves were disabled. Two warmup steps were excluded from the eight-step steady window.

| Metric | Disabled | Enabled | Delta / result |
| --- | ---: | ---: | ---: |
| Max absolute loss error (10 points) | — | — | 0 |
| Max absolute grad-norm error (10 points) | — | — | 0 |
| Steady tokens/s | 22,835.054 | 23,062.686 | +0.997% |
| Steady derived step time | 0.717560 s | 0.710590 s | -0.006970 s |
| Peak allocated accelerator memory | 18,950,668,288 bytes | 19,063,891,968 bytes | +113,223,680 bytes |
| EP-rank imbalance, before | unavailable | 1.0 at every step | — |
| EP-rank imbalance, after | unavailable | 0.0 at every step | strict improvement |
| Moved expert-token fraction | 0 | 0.5 at every step | replicas active |

A separate all-rank one-step Ascend profiler capture showed no `HcclSend`/`HcclRecv` events in the disabled run. The enabled trace contained the expected bf16 expert-row transfers with shapes `1024x2048` and `2048x512`, totaling 201,326,592 bytes (192 MiB) of unique send volume across four ranks for the captured optimizer step, including forward parameter transfer and backward gradient return. Mean per-rank `AscendCL` SEND+RECV device-event duration was 2.621 ms. This is communication-path evidence, not a step-aligned `p2p_wait_time_s` measurement: the reporter correctly leaves both P2P curves unavailable because only one separately profiled step was captured and HCCL work-handle wait time was not exported explicitly. Process-scoped host peak memory was also not captured.

## Reproduction

Reporter unit tests:

```shell
.venv/bin/python -m pytest -q \
  tests/scripts/test_compare_moe_ep_load_balance.py --tb=short
```

Honest local E2E gate:

```shell
.venv/bin/python -m pytest -q \
  tests/e2e/test_qwen3_5_moe_ep_load_balance.py --tb=short
```

Artifact comparison:

```shell
python scripts/profile/compare_moe_ep_load_balance.py \
  --baseline baseline.json \
  --candidate candidate.json \
  --warmup-steps 20 \
  --rtol 0.1 \
  --atol 0.1 \
  --relative-error-threshold 0.1 \
  --epsilon 1e-12 \
  --json-out comparison.json \
  --markdown-out comparison.md
```

Matched functional, steady-state performance, and all-rank profiler command templates are in the [Qwen3.5 training guide](../examples/qwen3_5.md#qwen35-moe-ep-load-balance-experiments). The functional and performance templates explicitly disable the recipe's profiler; the diagnosis template explicitly enables it on all ranks and excludes those traces from throughput evidence.

## Next required experiments

1. Run the isolated E2E on the declared CUDA self-hosted image and record the exact GPU model, driver/CUDA, PyTorch, Triton/FLA, NCCL, resolved arguments, exit status, and JSON envelope.
2. Repeat the Ascend run on the repository's declared torch/torch_npu 2.10 image and on a production-sized Qwen3.5-MoE checkpoint.
3. Extend the matched accelerator run to 100 or more steps and multiple seeds, retaining the predeclared warmup and explicit tokens/timing fields.
4. Add explicit runtime export for step-aligned `p2p_bytes` and `p2p_wait_time_s`; keep profiler wall-clock timings out of the steady throughput comparison.

## Rollback

Set `train.moe_ep_load_balance.enabled=false`, `max_replicas_per_rank=1`, and `moe_load_balance_monitor_interval=0`. This returns runtime behavior to the original fused EP dispatch without changing checkpoint or optimizer state. Code rollback must remove only scoped EP-balance changes and preserve unrelated worktree changes; any NPU modeling rollback is made in the patch spec and propagated with the repository's joint patchgen command rather than by hand-editing generated files.
