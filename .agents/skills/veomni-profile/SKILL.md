---
name: veomni-profile
description: "Use this skill for performance profiling and optimization. Two modes: (1) Analyze existing profile files (Chrome traces, memory snapshots) — write scripts to parse and summarize metrics per user requirements. (2) Generate profiles during development — configure ProfileConfig, run training, collect traces, analyze bottlenecks, and suggest optimizations. Trigger: 'profile', 'performance', 'slow', 'MFU', 'throughput', 'bottleneck', 'memory usage', 'trace', 'optimize training speed'."
---

## VeOmni Profiling Infrastructure

Key components:

| Component | Location | Purpose |
|-----------|----------|---------|
| `ProfileConfig` | `veomni/arguments/arguments_types.py` | Config fields: `enable`, `start_step`, `end_step`, `trace_dir`, `profile_memory`, `with_stack`, etc. |
| `create_profiler()` | `veomni/utils/helper.py` | Builds `torch.profiler.profile` (CUDA) or `torch_npu.profiler` (NPU) with schedule |
| `ProfileTraceCallback` | `veomni/trainer/callbacks/trace_callback.py` | Integrates profiler into the training loop via `BaseTrainer` |
| `VeomniFlopsCounter` | `veomni/utils/count_flops.py` | Analytical FLOPs/MFU computation per model family |
| `EnvironMeter` | `veomni/utils/helper.py` | Step-level throughput metrics (tokens/s, FLOPs, MFU) |
| `merge_chrome_trace.py` | `scripts/profile/merge_chrome_trace.py` | Merge multi-rank Chrome traces for unified viewing |

Output formats:
- **Chrome trace**: `veomni_rank{R}_{timestamp}.pt.trace.json.gz` — viewable in `chrome://tracing` or Perfetto
- **Memory snapshot**: `.pkl` file via `torch.cuda.memory._dump_snapshot` — viewable with PyTorch Memory Viz

---

## Mode 1: Analyze Existing Profile Files

User provides one or more profile files (Chrome traces, memory snapshots, logs). Write scripts to parse and analyze them.

### Steps

1. **Identify file types**: `.json.gz` / `.json` (Chrome trace), `.pkl` (memory snapshot), `.log` / `.txt` (training logs with throughput metrics).

2. **Understand the analysis goal** — ask the user what they want to know:
   - Kernel-level breakdown (which CUDA kernels dominate wall time?)
   - Communication vs computation ratio (NCCL all-reduce, all-to-all, all-gather time)
   - Memory high-water mark and allocation timeline
   - Per-step time breakdown (forward, backward, optimizer, data loading)
   - MFU / hardware utilization
   - Comparison across multiple profiles (e.g. before/after optimization, different parallelism configs)

3. **Write an analysis script** using `torch.profiler` APIs or raw JSON parsing:

   ```python
   import json, gzip
   from collections import defaultdict

   def load_chrome_trace(path):
       opener = gzip.open if path.endswith('.gz') else open
       with opener(path, 'rt') as f:
           return json.load(f)

   def analyze_kernel_time(trace):
       """Group events by kernel name, sum durations."""
       kernel_times = defaultdict(float)
       for event in trace.get('traceEvents', []):
           if event.get('cat') == 'kernel':
               kernel_times[event['name']] += event.get('dur', 0)
       return sorted(kernel_times.items(), key=lambda x: -x[1])
   ```

   Adapt the script to the user's specific analysis goal. Output tables, summaries, or CSV for further processing.

4. **For multi-rank traces**: use `scripts/profile/merge_chrome_trace.py` to merge before analysis, or analyze per-rank and compare.

5. **For memory snapshots**: load with `pickle`, analyze allocation records, identify peak usage and largest tensors.

6. **Present findings**: summarize top bottlenecks, compute/comm ratio, and actionable optimization suggestions.

---

## Mode 2: Generate Profiles During Development

Actively profile a training run to identify performance bottlenecks or validate optimizations.

### Step 1: Configure Profiling

Add or modify the `profile` section in the training YAML config:

```yaml
train:
  profile:
    enable: true
    start_step: 5        # skip warmup steps
    end_step: 10         # capture 5 steps
    trace_dir: ./profile_output
    record_shapes: true
    profile_memory: true  # enable memory snapshot (CUDA only)
    with_stack: true      # capture Python call stacks
    with_modules: true    # annotate with nn.Module names
    rank0_only: true      # profile only rank 0 to reduce overhead
```

Or pass via CLI overrides: `--train.profile.enable=true --train.profile.start_step=5 ...`

### Step 2: Run Training

```bash
source .venv/bin/activate
# Single GPU
python tasks/train_text.py --config configs/text/<model>.yaml

# Multi-GPU (profile will capture per-rank traces)
torchrun --nproc_per_node=8 tasks/train_text.py --config configs/text/<model>.yaml
```

### Step 3: Collect and Analyze

1. Locate outputs in `trace_dir`:
   - `veomni_rank*_.pt.trace.json.gz` — Chrome trace
   - `veomni_rank*_.pkl` — memory snapshot (if `profile_memory: true`)

2. Write analysis scripts as in Mode 1 to extract the metrics the user needs.

3. **Quick analysis shortcuts**:
   - **Kernel time breakdown**: parse Chrome trace events with `cat == 'kernel'`
   - **NCCL communication**: filter events with names matching `nccl` (e.g. `ncclAllReduceRingLLKernel`)
   - **Forward/backward split**: use `with_modules` trace annotations to separate phases
   - **Memory peak**: load `.pkl` snapshot, find max `allocated_bytes`
   - **MFU from logs**: `EnvironMeter` already logs `flops_achieved` and `flops_promised` — grep training logs

4. **For multi-rank comparison**: merge traces with `scripts/profile/merge_chrome_trace.py` or analyze per-rank to find stragglers.

### Step 4: Optimize

Based on findings, suggest and implement optimizations:

| Bottleneck | Typical solutions |
|------------|-------------------|
| Attention kernels dominate | Switch to FlashAttention 3/4 (`veomni/ops/flash_attn/`), check FA is actually active |
| NCCL communication > 30% | Increase compute/comm overlap, adjust FSDP reshard policy, try async SP |
| Memory OOM / high peak | Enable activation checkpointing, reduce micro-batch size, check for memory leaks |
| Data loading stalls | Increase `num_workers`, enable prefetch, check I/O throughput |
| Low MFU (< 40%) | Check dtype (bf16 vs fp32), verify tensor cores are used, check for host-device syncs |
| Uneven per-rank time | Check MoE load balancing, verify data distribution across ranks |

### Step 5: Validate

After optimization:
1. Re-profile with the same config to compare before/after.
2. Verify training correctness is preserved (loss matches baseline).
3. Document the optimization and results.

---

## NPU (Ascend) Profiling

On NPU, `create_profiler()` uses `torch_npu.profiler` instead of `torch.profiler`. Key differences:
- Output format includes AiC (Ascend insight Counters) metrics.
- Memory profiling uses NPU-specific APIs.
- Analysis tools differ — use Ascend Insight instead of Chrome tracing.
- Always guard NPU-specific analysis code with `is_torch_npu_available()`.
