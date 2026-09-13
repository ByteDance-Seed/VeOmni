# Inter-Layer Replay for FSDP2 MoE Training

Inter-layer replay (ILP) overlaps the backward pass of decoder layer `N` with
the activation-checkpoint replay of layer `N-1`. The current implementation is
an Ascend NPU path for Qwen3-MoE with FSDP2 and expert parallelism.

## Execution Model

Native activation checkpointing executes the selected layers serially:

```text
F'7 -> B7 -> F'6 -> B6
```

ILP starts the lower-layer replay as a sidecar and coordinates only shared
physical resources:

```text
F'7 -> [B7, F'6] -> [B6, F'5]
```

The native autograd backward remains authoritative. A completed sidecar replay
replaces the corresponding non-reentrant checkpoint replay; it is not an
additional model forward. Resource-specific FIFO queues and NPU device events
preserve collective order while preventing backward and replay GMM operations
from competing for the same AI Core interval.

## Requirements

- Ascend NPU with `torch_npu` and the CANN environment loaded.
- Qwen3-MoE using `moe_implementation: fused_npu`.
- FSDP2, non-reentrant gradient checkpointing, and expert parallelism.
- `ep_size` must divide `world_size`. ILP creates one replay communicator per
  expert-parallel subgroup, so multiple EP groups are supported.
- `torch.compile` must be disabled.

ILP is opt-in and does not change existing training behavior when disabled.

## Configuration

Add the following section under `train`:

```yaml
accelerator:
  ep_size: 16
  fsdp_config:
    fsdp_mode: fsdp2

gradient_checkpointing:
  enable: true
  enable_reentrant: false

inter_layer_replay:
  enable: true
  current_layer: -1
  window_size: 0
  strict: true
```

The complete example is
[`configs/text/qwen3-moe-ilp.yaml`](../../configs/text/qwen3-moe-ilp.yaml).

| Option | Default | Meaning |
| --- | ---: | --- |
| `enable` | `false` | Enables ILP. Set this to `false` for the baseline. |
| `current_layer` | `-1` | Highest scheduled layer; `-1` selects the final decoder layer. |
| `window_size` | `0` | Number of replay boundaries. `0` means every available boundary, not disabled. |
| `strict` | `true` | Fails if a required replay frame is unavailable instead of silently changing the schedule. |
| `memory_budget_gb` | `0` | Optional allocated-memory admission limit; `0` disables the limit. |
| `memory_reserve_gb` | `0` | Optional free-memory reserve; `0` disables the limit. |
| `memory_safety_factor` | `1.1` | Multiplier applied to observed replay-memory estimates. |
| `memory_retry_steps` | `4` | Probe interval while memory admission is paused. |

`current_layer` defines the first overlap pair as
`B(current_layer) || F'(current_layer - 1)`. `window_size` extends this schedule
toward lower layers. The default `current_layer=-1, window_size=0` selects every
available boundary; for a 48-layer model this is `B47/F'46` through `B1/F'0`.

## Quick Start

Load CANN and activate the VeOmni NPU environment first. Then provide the model
and dataset paths:

```bash
export MODEL_PATH=/path/to/Qwen3-30B-A3B
export TRAIN_PATH=/path/to/train.parquet

NPROC_PER_NODE=16 EP_SIZE=16 WINDOW_SIZE=0 \
  bash scripts/ilp/run_qwen3_moe.sh
```

Use a smaller explicit window to trade overlap coverage for memory:

```bash
MODE=ilp WINDOW_SIZE=20 NPROC_PER_NODE=16 EP_SIZE=16 \
  bash scripts/ilp/run_qwen3_moe.sh
```

Run a baseline with the same command surface:

```bash
MODE=baseline NPROC_PER_NODE=16 EP_SIZE=16 \
  bash scripts/ilp/run_qwen3_moe.sh
```

The runner accepts additional VeOmni CLI overrides after the script arguments.
For example, a short smoke run is:

```bash
MODE=ilp MAX_SEQ_LEN=512 MAX_STEPS=2 NUM_TRAIN_EPOCHS=1 \
  bash scripts/ilp/run_qwen3_moe.sh --train.profile.enable false
```

For a small EP2/FSDP2 correctness comparison against native checkpointing,
reserve two NPUs and run:

```bash
VEOMNI_RUN_ILP_E2E=1 pytest -q tests/e2e/test_inter_layer_replay.py
```

## Baseline/ILP Profiling

The benchmark wrapper runs baseline first and ILP second with identical model,
data, batch, and profiler settings:

```bash
export MODEL_PATH=/path/to/Qwen3-30B-A3B
export TRAIN_PATH=/path/to/train.parquet

NPROC_PER_NODE=16 EP_SIZE=16 WINDOW_SIZE=45 PROFILE_ENABLE=true \
  bash scripts/ilp/benchmark_qwen3_moe.sh
```

The default profiler interval is steps 2 through 8. Exclude the first two
captured steps when comparing steady-state means. The startup log must contain
the resolved layer range, window size, completion backpressure state, and EP
size. Use [`scripts/profile/analyze_ilp_alltoall.py`](../../scripts/profile/analyze_ilp_alltoall.py)
for per-step AllToAll arrival-skew analysis. Pass the actual decoder depth and
window; the analyzer excludes two captured warmup steps by default:

```bash
python scripts/profile/analyze_ilp_alltoall.py baseline.db ilp.db \
  --num-layers 48 --current-layer -1 --window-size 45 --warmup-steps 2
```

## Validated Qwen3-MoE 30B Result

The 48-layer, EP16, 16384-token full-window run measured a 15.00 percent step-
time reduction and a 17.65 percent throughput increase. Peak allocation rose
from 34.50 GB to 35.76 GB. A 20-boundary window retained a 6.96 percent
throughput increase with a 0.20 GB formal-run peak premium. Memory depends on
sequence packing and allocator history, so measure the complete training window
before selecting a production value.

The speedup does not come from skipping FSDP parameter materialization or MoE
token exchange. Baseline and ILP execute the same FSDP parameter AllGather and
MoE AllToAll counts. ILP reuses the dispatch plan saved by the original forward,
removes one EP token-count metadata AllGather per active replay boundary, and
moves selected recomputation work off the serial backward critical path.

## Limitations

- Only Qwen3-MoE decoder layers with the NPU fused MoE implementation are
  supported.
- Every layer in the selected window must be a sparse MoE layer.
- Pipeline, tensor, context, and Ulysses sequence parallelism are rejected when
  ILP is enabled; each of their sizes must remain one.
- Memory-budget admission is a guardrail, not a guarantee of exact peak-memory
  parity. Always run an exclusive baseline/ILP comparison for a new topology.
