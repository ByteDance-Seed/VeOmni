# Ascend Environment Variables

This document describes the key environment variables that can be configured to optimize VeOmni performance on Ascend NPUs.

## CPU_AFFINITY_CONF

```shell
export CPU_AFFINITY_CONF=1
```

Enable coarse-grained or fine-grained CPU core binding. This configuration helps prevent thread contention, improves cache hit rates, avoids memory access across different NUMA (Non-Uniform Memory Access) nodes, and reduces task scheduling overhead—collectively optimizing task execution efficiency.

### Parameter Settings:

* `0`: Disable the binding function. Default is `0`.
* `1`: Enable coarse-grained kernel binding.
* `2`: Enable fine-grained kernel binding.

## PYTORCH_NPU_ALLOC_CONF

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

`expandable_segments:<value>`: Enable the memory pool extension segment feature.

* `True`: This configuration instructs the cache allocator to create specific memory blocks with the capability to be extended later. This allows for more efficient handling of scenarios where the required memory size frequently changes during runtime.
* `False`: The memory pool extension segment feature is disabled, and the original memory allocation method is used. Default is `False`.

## MULTI_STREAM_MEMORY_REUSE

```bash
export MULTI_STREAM_MEMORY_REUSE=2
```

This environment variable enables memory reuse across multiple streams, which can help reduce memory fragmentation and improve memory utilization.

## Other Useful Environment Variables

For additional environment variables and their configurations, please refer to the Ascend CANN official documentation:
https://www.hiascend.com/document/detail/zh/canncommercial/900/index/index.html
