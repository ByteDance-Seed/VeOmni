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

### Configuration Items That May Affect Performance

The following configuration items will impact training performance and need to be set according to the scenario:

- **record_shapes**: Recording tensor shapes increases profiling overhead
- **profile_memory**: Enabling memory profiling adds additional overhead
- **with_stack**: Recording stack traces significantly increases profiling overhead
- **rank0_only**: When set to False, all ranks will be profiled, generating a large number of files and consuming significant disk space and time

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
```

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
