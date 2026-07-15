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

- **PyTorch Version**: Verified against PyTorch 2.9.1, 2.10.0, and the allow-listed 2.11.0 upstream/CI-wheel implementations. Other patch releases or builds fail the source-hash guard until explicitly verified and added.
- **Tensors must be sorted by offset** before writing (already ensured by the implementation)

## Implementation Details

See `veomni/checkpoint/dcp_consolidation.py` for full implementation.

### Patch Application Flow

1. User calls `_save_hf_safetensor_distributed()`
2. Function applies patch via `apply_dcp_consolidation_patch()`
3. Patch verifies that the torch version and `_process_output_file` implementation are known-compatible
4. Patch replaces `_process_output_file` in `torch.distributed.checkpoint._consolidate_hf_safetensors`
5. `dcp.save()` proceeds with patched function

### Guards

- `_dcp_consolidation_patch_applied`: Module-level flag to prevent duplicate patching
- `_SUPPORTED_TORCH_VERSION_PREFIXES`: Allowed PyTorch version prefixes
- `_SUPPORTED_PROCESS_OUTPUT_FILE_SHA256`: Hashes of verified `_process_output_file` implementations

## When to Update

When upgrading PyTorch beyond the currently supported versions:

1. Verify that the `_process_output_file` signature and implementation remain compatible
2. Add the version prefix and verified implementation hash in `veomni/checkpoint/dcp_consolidation.py`
3. Test the patch with HDFS FUSE before documenting the new version as supported
