# FAQ: Common Issues and Solutions for Ascend NPU

This document addresses frequently asked questions and common issues encountered when using VeOmni with Ascend NPUs.

## Q: How to resolve memory fragmentation issues on NPU?

### A: Set the multi-stream memory reuse environment variable

```bash
# Enable NPU multi-stream memory reuse
export MULTI_STREAM_MEMORY_REUSE=2
```

This enables the NPU's multi-stream memory reuse feature, which reduces memory fragmentation and improves utilization. Recommended value: 2.

> **Note**: This environment variable is already set by default in `train.sh`

## Q: How to configure multi-node training?

### A: Modify environment variables in train.sh

Below is a **2-node example** (adjust according to your cluster size):

```bash
# Number of nodes (2 nodes in this example)
NNODES=${NNODES:=2}
# Current node rank (0 to 1 for 2 nodes - must be different for each machine)
NODE_RANK=${NODE_RANK:=0}
# Master node address (IP address - must be the same across all machines)
MASTER_ADDR=${MASTER_ADDR:=192.168.1.100}
# Master node port (default works for most cases)
MASTER_PORT=${MASTER_PORT:=12345}
# Number of NPUs per node (A2: max 8, A3: max 16)
NPROC_PER_NODE=${NPROC_PER_NODE:=8}
```

> **Configuration Location**: These parameters are in `train.sh` (lines 9-37).

**Parameter Explanations**:
- `NNODES`: Total number of nodes in your cluster
- `NODE_RANK`: Unique identifier for each node (0 to NNODES-1)
- `MASTER_ADDR`: IP address of the master node (same for all machines)
- `MASTER_PORT`: Communication port (default: 12345)
- `NPROC_PER_NODE`: NPUs per node (A2: max 8, A3: max 16)

**Important Notes**:
- All nodes must communicate via `MASTER_ADDR:MASTER_PORT`
- All nodes need the same configuration files and data paths
- Ensure network connectivity between all nodes

## Q: How to resolve "'liger_kernel' is not supported on Ascend NPU" error?

### A: Set model.ops_implementation parameters in YAML

Configure the operators in your YAML configuration file:

```yaml
model:
  ops_implementation:
    # Attention implementation selection
    attn_implementation: "flash_attention_2"  # Options: "eager", "sdpa", "flash_attention_2", "flash_attention_3", "flash_attention_4", "native-sparse"
    # MoE implementation selection
    moe_implementation: "fused_npu"    # Options: "eager", "fused_npu"
```

> **Configuration Location**: These parameters are in your YAML configuration file (see `arguments.md` lines 127-135).

**NPU Optimized Operators**:
VeOmni automatically detects NPU environment and uses optimized operators:
- `npu_group_gemm`: MoE GroupGEMM operator (`npu_group_gemm.py:1-114`)
- `npu_rms_norm`: RMS normalization operator (`npu_fused_operator.py:20-26`)
- `npu_rotary_mul`: RoPE positional encoding operator (`npu_fused_operator.py:28-52`)

**Note**: NPU automatically selects optimal operator implementations. For Attention, it uses SDPA or CANN built-in operators; for MoE, it uses `npu_group_gemm`.

## Q: How to resolve "'global batch size' should be a multiple of 8/16/32" error?

### A: Ensure proper batch size configuration

Make sure the global batch size meets the multiple requirement:

```
global_batch_size = micro_batch_size × data_parallel_size × gradient_accumulation_steps
```

**Important Notes**:
- If `global_batch_size` is not set, the system automatically calculates it as `micro_batch_size × dp_size`
- Ensure `global_batch_size` can be divided by all parallel dimensions

## Q: How to set NPU device visibility?

### A: Use ASCEND_RT_VISIBLE_DEVICES environment variable

```bash
# Make only NPUs 0,1,2,3 visible
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

**Automatic Detection**:
If not set, the system automatically detects all available NPU devices:

```bash
# Automatically detect number of available NPUs
NPROC_PER_NODE=$(ls -l /dev/davinci* | grep -v "davinci_manager" | wc -l)
```

**Note**: Similar to CUDA's `CUDA_VISIBLE_DEVICES`, this controls which NPU devices are visible to processes.

## Q: How to resolve Transformers version incompatibility issues?

### A: Use compatible Transformers versions

Ensure you're using a compatible Transformers version:

```bash
# Check current Transformers version
python -c "import transformers; print(transformers.__version__)"

# Install using uv
uv sync --locked --extra npu --group dev

# Or install using pip
pip install transformers==5.9.0
```

**Version Recommendations**:
- VeOmni pins Transformers `5.9.0` (see `pyproject.toml`). Other v5 minor
  versions may work but are not exercised in CI.
