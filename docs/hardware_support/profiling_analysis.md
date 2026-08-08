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
| npu_analysis_mode | `offline` \| `async` | `offline` | How Ascend trace analysis runs after raw finalization |

### Configuration Items That May Affect Performance

The following configuration items will impact training performance and need to be set according to the scenario:

- **record_shapes**: Recording tensor shapes increases profiling overhead
- **profile_memory**: Enabling memory profiling adds additional overhead
- **with_stack**: Recording stack traces significantly increases profiling overhead
- **rank0_only**: When set to False, all ranks will be profiled, generating a large number of files and consuming significant disk space and time
- **npu_analysis_mode**:
  - `offline` finalizes the raw `*_ascend_pt` capture during training. In a Merlin job, VeOmni automatically starts a sidecar to parse, gzip, and upload a clickable profiling asset through a platform-provided file uploader or `merlin-cli`. It deliberately avoids JSON/base64 SDK uploads for large traces. If no safe uploader is available, the sidecar reports the failure and preserves raw data. This is the default and safest mode for large captures.
  - `async` calls the official torch_npu online handler with `analyse_flag=True, async_mode=True`. Raw finalization and parser submission are synchronous, but Chrome/DB analysis continues in torch_npu's process pool while training advances.

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
        npu_analysis_mode: offline  # offline | async
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
--train.profile.npu_analysis_mode async
```

Use pod-local storage for large Ascend captures, then copy / parse / upload outside the training barrier.

In `offline` mode, VeOmni can spawn a detached postprocess sidecar after raw finalization:

| Env | Effect |
|-----|--------|
| unset (default) | In a Merlin job, auto-spawn sidecar: analyse → gzip → profiling upload through a platform file uploader or `merlin-cli`; associate with the current Trial for the JobRun Profiling tab (JobRun fallback when no Trial is available); otherwise preserve raw data |
| `VEOMNI_UPLOAD_CMD=...` | Auto-spawn sidecar: analyse → run the command on `trace_view.json.gz` (`{trace}` placeholder supported) |
| `VEOMNI_NPU_OFFLINE_MERLIN_UPLOAD=1` | Force Merlin upload through a platform file uploader or `merlin-cli` (job/trial read from Merlin env) |
| `VEOMNI_NPU_OFFLINE_MERLIN_UPLOAD=0` | Disable automatic Merlin upload |
| `VEOMNI_NPU_OFFLINE_POSTPROCESS=1` | Force-spawn sidecar analysis; also copy when `trace_dir` is `hdfs://` |
| `VEOMNI_NPU_OFFLINE_POSTPROCESS=0` | Disable automatic postprocessing; raw data remains pod-local, with no synchronous fallback |

VeOmni waits for an automatically spawned sidecar for up to 300 seconds when training ends. A timeout is non-fatal and leaves the raw local capture in place; for very large captures, use the manual postprocess command below while the pod remains alive.

Manual / external postprocess (recommended when the train pod may exit soon after capture; `--merlin-upload` requires `merlin-cli` on `PATH`):

```bash
python -m veomni.utils.npu_offline_postprocess \
  --raw-dir /tmp/veomni_npu_profile \
  --copy-to hdfs://haruna/.../profile/ \
  --analyse \
  --merlin-upload
```

If only a platform file uploader is available, pass it explicitly with `--upload-cmd '<command>'` instead.

Sidecar logs are written next to the raw directory as `veomni_npu_offline_postprocess.log`. Sidecar startup, analysis, or upload failures never fall back to synchronous work in the distributed barrier; the raw capture remains available for recovery.
NPU profiler initialization, raw finalization, and cleanup failures are also non-fatal: the failing rank disables further profiling, every rank still leaves the paired barrier, and training continues.

For distributed Ascend training, use the following options together:

```yaml
train:
  profile:
    enable: true
    rank0_only: true
    npu_analysis_mode: offline
    trace_dir: /tmp/veomni_npu_profile
```

VeOmni synchronizes all ranks immediately before and after the final profiler step on Ascend. This prevents non-profiled ranks from entering the next collective while rank 0 finalizes its capture. In both modes the barrier covers raw finalization; in `async` it also covers the short process-pool submission, never the full analysis. All ranks must execute the profile callback at the same global step. `start_step` and `end_step` are absolute global steps; VeOmni rebases the remaining schedule after checkpoint resume or a hot update and skips a window that has already elapsed.

Use `/tmp` or another pod-local SSD for `async`: automatic HDFS copy/upload is intentionally rejected or skipped because the background parser may still be writing its outputs. Wait for training to exit, then copy or upload the completed trace.

An `hdfs://` trace directory is supported in `offline` mode. VeOmni captures locally and automatically starts a sidecar to copy the raw directory; it never falls back to copying a large directory inside the distributed finalization barrier. If sidecar startup fails or is disabled, the raw local path is logged for recovery.

Async analysis can compete with training for host CPU and disk bandwidth, and torch_npu waits for its process pool during interpreter exit. Use `offline` for the lowest training interference or very large traces. If the training process is a multiprocessing daemon, VeOmni logs a warning and safely falls back from `async` to `offline` because torch_npu refuses daemon-process analysis.

The former `npu_offline_analysis: true` setting remains as a deprecated alias for `npu_analysis_mode: offline`. Explicit `npu_offline_analysis: false` is rejected because it selected the removed synchronous online parser; choose `async` or `offline` explicitly.

The current trainer callback and `tasks/omni/train_omni_model.py` support both modes. Deprecated standalone training entrypoints reject NPU profiling explicitly because they cannot honor the distributed synchronization contract.

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
