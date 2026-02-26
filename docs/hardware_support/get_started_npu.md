# Get Started with Ascend NPU

## Key Updates

2025/12/23: VeOmni supports training on Ascend NPU.

## Installation

Please refer to [Installation with Ascend NPU](../get_started/installation/install_ascend.md).

## Supported Models

| Model                               | Model Size | Support | FSDP1 | FSDP2 | EP | SP | Note |
|-------------------------------------|------------|---------|-------|-------|----|----|------|
| [Qwen3](../examples/qwen3.md)       | 8B         | ✅       |       | ✅     |    |    |      |
|                                     | 30B        | ✅       |       | ✅     |    |    |      |
| [Qwen3 VL](../examples/qwen3_vl.md) | 8B         | ✅       |       | ✅     |    | ✅  |      |
|                                     | 30B        | ✅       |       | ✅     | ✅  | ✅  |      |
| [Wan2.1](../examples/wan2.1.md)     | 14B        | ✅       | ✅     |       |    | ✅  |      |

## Environment Variables

### CPU_AFFINITY_CONF

```shell
export CPU_AFFINITY_CONF=1
```
Enable coarse-grained or fine-grained CPU core binding. This configuration helps prevent thread contention, improves cache hit rates, avoids memory access across different NUMA (Non-Uniform Memory Access) nodes, and reduces task scheduling overhead—collectively optimizing task execution efficiency.
Parameter Settings:

* `0`: Disable the binding function. Default is `0`.
* `1`: Enable coarse-grained kernel binding.
* `2`: Enable fine-grained kernel binding.

### PYTORCH_NPU_ALLOC_CONF

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

`expandable_segments:<value>`: Enable the memory pool extension segment feature.  
* `True`: This configuration instructs the cache allocator to create specific memory blocks with the capability to be extended later. This allows for more efficient handling of scenarios where the required memory size frequently changes during runtime.  
* `False`: The memory pool extension segment feature is disabled, and the original memory allocation method is used. Default is `False`.

## Declarations

The Ascend support code, Dockerfile and image provided in the documentation are for reference only. If you intend to use them in a production environment, please contact the official channels. Thank you.
