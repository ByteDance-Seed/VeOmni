# HDFS FUSE Patch for DCP Consolidation

This document explains the monkey patch for PyTorch DCP (Distributed Checkpoint) safetensors consolidation for HDFS FUSE compatibility.

## Problem Background

When saving HuggingFace format checkpoints using `HuggingFaceStorageWriter` with `enable_consolidation=True`, PyTorch internally consolidates sharded safetensors files by calling `_consolidate_safetensors_files` in `torch.distributed.checkpoint._consolidate_hf_safetensors`.

### The Problem

The original implementation uses `r+b` mode for random write access:

```python
# Original PyTorch implementation in _process_output_file
with open(output_file, "r+b") as output_stream:
    output_stream.seek(0, os.SEEK_END)
    ...

# This fails on HDFS FUSE with:
# OSError: [Errno 95] Operation not supported
```

This is not supported by some distributed file systems like HDFS via FUSE, which only support append-only writes.

## Solution

This patch replaces `r+b` mode with `ab` (append) mode, which is compatible with append-only file systems.

```python
# Patched implementation
with open(output_file, "ab") as output_stream:
    ...
```

### Key Changes

1. **File mode**: `r+b` -> `ab` (append-only)
2. **Seek removal**: No `seek(0, SEEK_END)` needed in append mode
3. **Order guarantee**: Tensors are sorted by `offset_in_file` before writing

## Usage

The patch is applied just-in-time when using the distributed HuggingFace safetensors save functionality:

```python
from veomni.utils.save_safetensor_utils import _save_hf_safetensor_distributed

# Patch is applied automatically inside this function before dcp.save()
```

No manual action is required. The patch is guarded by `_dcp_consolidation_patch_applied` flag to prevent duplicate patching.

## Requirements

- **PyTorch Version**: Requires PyTorch 2.9.x (e.g., 2.9.0, 2.9.1)
- **Tensors must be sorted by offset** before writing (already ensured by the implementation)

## Implementation Details

See `veomni/ops/dcp_consolidation.py` for full implementation.

### Patch Application Flow

1. User calls `_save_hf_safetensor_distributed()`
2. Function applies patch via `apply_dcp_consolidation_patch()`
3. Patch verifies torch version (must be 2.9.x)
4. Patch replaces `_process_output_file` in `torch.distributed.checkpoint._consolidate_hf_safetensors`
5. `dcp.save()` proceeds with patched function

### Guards

- `_dcp_consolidation_patch_applied`: Module-level flag to prevent duplicate patching
- `_REQUIRED_TORCH_VERSION`: Version check to ensure patch compatibility

## When to Update

When upgrading PyTorch to a version other than 2.9.x:

1. Update `_REQUIRED_TORCH_VERSION` in `veomni/ops/dcp_consolidation.py`
2. Verify that `_process_output_file` function signature hasn't changed in the new PyTorch version
3. Test this patch with HDFS FUSE to ensure compatibility
