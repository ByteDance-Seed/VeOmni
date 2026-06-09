# Precision Analysis and Troubleshooting Guide

This document guides developers on how to quickly locate and resolve precision issues when migrating and training models on Ascend NPUs. Through standardized processes and tool usage, even first-time users can efficiently achieve precision alignment.

## Understanding Precision Considerations

When working with Ascend NPUs, precision behavior can vary based on multiple factors including model architecture, operator implementations, and training configurations. Instead of rigid thresholds, it's important to focus on identifying and analyzing precision discrepancies that may impact model performance.

### Common Precision Indicators

While specific thresholds depend on the use case, the following metrics can help identify potential precision issues:

- **Loss Consistency**: Stable and expected loss reduction patterns during training
- **Output Similarity**: Comparable model outputs for identical inputs
- **Numerical Stability**: Absence of NaN/Inf values and extreme outliers
- **Convergence Behavior**: Similar training convergence characteristics across runs

Precision analysis should be tailored to the specific model and training objectives, considering factors like model size, complexity, and target performance metrics.

## Precision Anomaly Determination

Precision issues during model training may lead to the following phenomena:

- **Abnormal Loss Curve**: Loss values suddenly increase or fluctuate significantly during training, failing to decrease smoothly.
- **NaN or Overflow**: Model outputs contain NaN (Not a Number) or Inf (Infinity) values.
- **Unmet Precision Requirements**: Loss values show spikes or other anomalies relative to the baseline, exceeding the defined metrics.

## Common Causes of Precision Issues

### Inconsistent Training Environment

Versions of transformers, triton, flash-attn, and related Ascend toolkits can significantly affect training precision. Carefully verify version consistency between GPU and NPU environments.

### Misaligned Hyperparameter Configuration

Ensure complete alignment of training and distributed configuration parameters between GPU and NPU training scripts.

### Unensured Deterministic Computation

#### Enable Deterministic Computation

Enable deterministic computation on NPUs to ensure consistent tensor outputs for the same input across multiple runs.

##### Common Ways to Enable Deterministic Computation in VeOmni

1. Install the mindstudio-probe tool library:

```bash
pip install mindstudio-probe
```

2. Add the following code to the main program:

```python
from msprobe.pytorch import seed_all
seed_all(mode=True)
```

Or use this custom implementation:

```python
# def main():
#     pass

def seed_all(seed=42, mode=True, is_gpu=False):
    print("======================seed_all=============================")
    import random
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)
    if is_gpu:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.benchmark = False
    else:
        import torch_npu
        torch_npu.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)

if __name__ == "__main__":
    import os
    seed_all(mode=os.environ.get('DETERM_COMP', "true") == "true")
    main()
```

3. **Fixed Data Order**: Ensure DataLoader has `shuffle=False` and `num_workers=0` (or fix worker initialization seeds) to guarantee identical batch data reading on both NPU and GPU.

4. **Fixed Random Noise**: For generative models, ensure Noise Tensors generated on NPU and GPU are identical (you can generate and save them on GPU first as .pt files, then load them directly on NPU).

## Precision Issue Localization Process

### Overall Network Loss Comparison

Compare the Loss curves between NPU and GPU.

**Judgment Criteria**:
- If the Loss difference is within an acceptable range (e.g., <1%), precision is basically up to standard.
- If the Loss difference is large, fails to meet precision metrics, or NaN appears, proceed to the following troubleshooting steps.

### Middle Layer Tensor Dump (Binary Search Method)

Use tools (such as msprobe or manual hooks) to capture output tensors from intermediate layers of the model.

**Strategy**: Select key nodes at 1/2, 1/4, 3/4 of the model for dumping.

**Comparison Metrics**: Calculate Mean, Max, Sum, and Cosine Similarity.

**Localization**:
- If the output of layer N is consistent but layer N+1 is inconsistent, the problem lies in the operator of layer N+1.
- If all layers are consistent but Loss is inconsistent, check for implementation differences in the Loss calculation function (e.g., CrossEntropy).

### Single Operator Reproduction and Localization

After identifying the problematic operator, conduct isolated testing.

1. **Input Extraction**: Save the input tensors of the operator on both NPU and GPU as .npy files.

2. **Independent Operation**: Write a script to load the inputs and execute only the operator logic on both NPU and GPU.

3. **Result Analysis**: Compare output results.
   - **Large Error**: Confirm it's an operator implementation issue (possibly incorrect operator precision mode configuration, such as FP16 overflow).
   - **Small Error**: May be caused by accumulated errors from preceding or subsequent operators.

## Tool for Locating Precision Issues - MindStudio

Use the `mindstudio-probe` tool for precision data collection. This document provides examples; for details, please refer to the [Ascend Tool Official Documentation](https://gitcode.com/Ascend/msprobe/).

### Localization Process

#### Confirm Problem Reproduction

Before using tools for analysis and localization, ensure the problem is reproducible. Fix randomness and repeat training to fix the phenomenon as much as possible, ensuring it's an operator-induced problem.

#### Install the Tool

```bash
pip install mindstudio-probe
```

#### Edit Debugger Configuration File

For most cases, using "statistics" as the task is sufficient:

```json
{
    "task": "statistics",
    "dump_path": "/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "async_dump": false,

    "statistics": {
        "scope": [],
        "list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

#### Add Debugger Code

```python
from mindstudio_probe import Debugger
debugger = Debugger(config_path="path/to/config.json")

# Simulated training code
for step, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    debugger.step(step)
```

#### Collect Data

Run the training script until the specified step is completed to obtain the input/output states of each API. Repeat the same operation on GPU to get data from both NPU and GPU.

#### Precision Data Comparison

Configure compare.json:

```json
{
  "npu_path": "/data_dump/step0/rank0/dump.json",
  "bench_path": "/gpu_data_dump/step0/rank0/dump.json",
  "stack_path": "/home/data2/ltt/mirro/mirro4huawei/data_dump/step0/rank0/stack.json",
  "is_print_compare_log": true
}
```

Perform precision comparison:

```bash
msprobe -f pytorch compare -i ./compare.json -o ./output -s
```

#### Locate Error Root Cause

- **Small Comparison Result Error**: Troubleshoot pre-processing and post-processing procedures to determine if it's an accumulated error issue.
- **Large Comparison Result Error**: Find the first operator that doesn't meet precision standards and analyze the error root cause. If it's confirmed to be an operator implementation issue, contact technical engineers for further support.
