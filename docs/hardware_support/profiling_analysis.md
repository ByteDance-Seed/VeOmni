# Model Optimization - Profiling Collection, Analysis and Optimization Ideas

Performance optimization is a critical step when training models on Ascend NPUs. Performance analysis (Profiling) can effectively identify performance bottlenecks and optimize model training efficiency. This guide will detail how to collect and analyze profiling data, including relevant configurations, tool usage, and typical performance problem analysis methods.

## Profiling Collection Configuration and Description

VeOmni's profiling configuration is located under the `train.profile.*` namespace, defined by the `ProfileConfig` class in `veomni/arguments/arguments_types.py` .

### Configuration Item Description

| Configuration Item | Type | Default Value | Description |
|---------------------|------|---------------|-------------|
| enable | bool | False | Whether to enable profiling |
| start_step | int | 1 | The step to start profiling |
| end_step | int | 2 | The step to end profiling |
| trace_dir | str | "./trace" | Directory to save profiling traces |
| record_shapes | bool | True | Whether to record input tensor shapes |
| profile_memory | bool | True | Whether to profile memory usage |
| with_stack | bool | True | Whether to record stack traces |
| with_modules | bool | False | Whether to record module hierarchy in profiling traces |
| rank0_only | bool | True | Whether to profile only rank 0 |
| npu_offline_analysis | bool | False | Whether to finalize Ascend raw data during training and analyze it offline |

### Configuration Items That May Affect Performance

The following configuration items will impact training performance and need to be set according to the scenario:

- **record_shapes**: Recording tensor shapes increases profiling overhead
- **profile_memory**: Enabling memory profiling adds additional overhead
- **with_stack**: Recording stack traces significantly increases profiling overhead
- **rank0_only**: When set to False, all ranks will be profiled, generating a large number of files and consuming significant disk space and time
- **npu_offline_analysis**: When set to True on Ascend, the training process finalizes raw profiling data without generating the Chrome trace. Parse the finalized `*_ascend_pt` directory later with the torch_npu offline analysis API or `msprof`. This avoids blocking the training step on a large trace export.

### Typical Configuration Method

Add profiling configuration in the model's YAML configuration file:

```yaml
train:
    profile:
        enable: true
        start_step: 5
        end_step: 6
        record_shapes: true
        trace_dir: ./profiling
        npu_offline_analysis: true
```

The same option can be enabled from the command line:

```bash
--train.profile.enable true \
--train.profile.start_step 5 \
--train.profile.end_step 6 \
--train.profile.trace_dir /tmp/veomni_npu_profile \
--train.profile.rank0_only true \
--train.profile.record_shapes false \
--train.profile.profile_memory false \
--train.profile.with_stack false \
--train.profile.with_modules false \
--train.profile.npu_offline_analysis true
```

Use pod-local storage for large Ascend captures, then copy the finalized raw directory to durable storage before parsing it offline.

For distributed Ascend training, use the following options together:

```yaml
train:
  profile:
    enable: true
    rank0_only: true
    npu_offline_analysis: true
    trace_dir: /tmp/veomni_npu_profile
```

VeOmni synchronizes all ranks immediately before and after the final profiler step on Ascend (both online and offline analysis). This prevents non-profiled ranks from entering the next collective while rank 0 finalizes its capture. Prefer `npu_offline_analysis: true` so that barrier only covers raw finalization rather than a long Chrome/DB export. All ranks must execute the profile callback at the same global step. `start_step` and `end_step` are absolute global steps; VeOmni rebases the remaining schedule after checkpoint resume or a hot update and skips a window that has already elapsed.

Using an `hdfs://` trace directory is supported, but not recommended for large Ascend captures: it still copies the raw directory synchronously before releasing the other ranks and can therefore stall training. Prefer `/tmp` or another pod-local SSD path, then copy the finalized `*_ascend_pt` directory to durable storage from a separate process.

The current trainer callback and `tasks/omni/train_omni_model.py` support this option. Deprecated standalone training entrypoints reject it explicitly instead of falling back to online analysis.

## Profiling Analysis Tool - MindStudio Insight

After configuring the collection script, start the training script to begin performance data collection. Results are output to the specified folder. MindStudio is typically used for visual analysis of profiling data.

Use MindStudio Insight's visualization tools for performance analysis, viewing operator execution time, communication time, memory usage, etc. For details, refer to the [Ascend Tool Official Documentation](https://www.hiascend.com/document/detail/zh/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md).

## Typical Performance Problem Analysis

### 1. Computational Bottleneck Analysis

**Check NPU Utilization:**
- Use TensorBoard or MindStudio Insight to view operator execution time
- Identify operators with long execution times, analyze their input shapes and types to determine if they are computational bottlenecks
- Examine operator call stacks to identify redundant operations
- Identify computationally intensive operations (such as attention, matmul)
- Check for serialization operations causing NPU idle time

### 2. Memory Bottleneck Analysis

**Memory Usage Analysis:**
- Use TensorBoard or MindStudio Insight to view memory usage
- Identify steps with high memory usage, analyze memory allocation and deallocation
- Determine if memory rearrangement exists

### 3. Multi-Machine Multi-Card Communication Bottleneck Analysis

**In Distributed Training:**
- Use MindStudio Insight to view the multi-card communication overview, analyzing computation, communication, and idle time for each card
- Find cards with long communication times, analyze their communication matrices to identify slow cards and links
- Check the time consumption of collective communications such as all-reduce and all-gather
- Analyze if NPU idle waiting is caused by communication

### 4. Data Loading Bottleneck Analysis

**CPU Activity Analysis:**
- View data preprocessing time
- Check if the dataloader is a bottleneck
- Analyze the overlap between data loading and computation
