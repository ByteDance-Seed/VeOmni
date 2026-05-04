# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: F821
# Reason: This module patches PyTorch internal functions using types.FunctionType
# with __globals__ bound to the target module. Variables like DATA_OFFSETS_KEY,
# _read_tensor_data_mmap, etc. are resolved at runtime from torch.distributed.checkpoint.

"""Patch for PyTorch DCP (Distributed Checkpoint) safetensors consolidation.

This module provides a monkey patch to fix compatibility issues with distributed
file systems (e.g., HDFS via FUSE) that do not support r+b (read-write binary)
file mode for random write access.
"""

import types


_dcp_consolidation_patch_applied = False

# Fixed torch version for this patch - update when upgrading torch
_REQUIRED_TORCH_VERSION = "2.9"


def apply_dcp_consolidation_patch():
    """Patch DCP safetensors consolidation to use append mode for HDFS FUSE compatibility.

    The original implementation in PyTorch uses r+b mode for random write access:
        with open(output_file, "r+b") as output_stream:
            output_stream.seek(0, os.SEEK_END)
            ...

    This is not supported by some append only file systems (e.g., HDFS via FUSE).
    This patch replaces the function to use append mode instead:

        with open(output_file, "ab") as output_stream:
            ...

    Note: Append mode requires tensors to be processed in offset order, which is
    already ensured by sorting tensors before writing.

    The patch uses types.FunctionType to create a new function with __globals__
    bound to the target module, enabling access to internal functions like
    _read_tensor_data_mmap and _write_sub_tensor_to_file_optimized.
    """
    global _dcp_consolidation_patch_applied

    if _dcp_consolidation_patch_applied:
        return

    # Verify torch version is 2.9 or later
    import torch

    # Parse version to support 2.9, 2.10, 2.11, etc. and variants like 2.11.0+xpu
    torch_version_parts = torch.__version__.split("+")[0].split(".")
    try:
        torch_major, torch_minor = int(torch_version_parts[0]), int(torch_version_parts[1])
        is_compatible = (torch_major > 2) or (torch_major == 2 and torch_minor >= 9)
    except (ValueError, IndexError):
        is_compatible = False

    if not is_compatible:
        raise RuntimeError(
            f"DCP consolidation patch requires torch >= {_REQUIRED_TORCH_VERSION}, "
            f"but got {torch.__version__}. Please update PyTorch."
        )

    import torch.distributed.checkpoint._consolidate_hf_safetensors as hf_module

    if not hasattr(hf_module, "_process_output_file"):
        raise RuntimeError(
            f"torch.distributed.checkpoint._consolidate_hf_safetensors does not have "
            f"_process_output_file attribute. Please verify torch {_REQUIRED_TORCH_VERSION}.x compatibility."
        )

    # Define the replacement function logic
    # This is a modified version of torch.distributed.checkpoint._consolidate_hf_safetensors._process_output_file
    # Original: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/distributed/checkpoint/_consolidate_hf_safetensors.py
    # Key change: Use append mode ("ab") instead of read-write mode ("r+b") for HDFS FUSE compatibility
    def _process_output_file_impl(output_file, output_data, input_files_data):
        sorted_tensors = sorted(output_data.fqn_data.items(), key=lambda x: x[1].offset_in_file)

        with open(output_file, "ab") as output_stream:  # Changed from "r+b"
            for tensor_fqn, tensor_fqn_data in sorted_tensors:
                full_tensor_mv = memoryview(
                    bytearray(math.prod(tensor_fqn_data.shape_in_file) * tensor_fqn_data.dtype_size)
                )

                for safetensors_file in input_files_data:
                    file_metadata = input_files_data[safetensors_file].metadata
                    input_metadata_size = input_files_data[safetensors_file].metadata_size

                    if tensor_fqn not in file_metadata:
                        continue

                    metadata = file_metadata[tensor_fqn]
                    data_offsets = metadata[DATA_OFFSETS_KEY]

                    # These functions are resolved from hf_module's globals at runtime
                    data_to_write = _read_tensor_data_mmap(
                        safetensors_file,
                        data_offsets[0],
                        data_offsets[1],
                        input_metadata_size,
                    )

                    fqn_custom_metadata = _get_dcp_custom_metadata(file_metadata)[tensor_fqn]
                    offsets_of_tensor_being_read = fqn_custom_metadata[SAVED_OFFSETS_KEY]

                    _write_sub_tensor_to_file_optimized(
                        full_tensor_mv,
                        data_to_write,
                        tensor_fqn_data.dtype_size,
                        tensor_fqn_data.shape_in_file,
                        offsets_of_tensor_being_read,
                        metadata[SHAPE_KEY],
                    )

                output_stream.write(full_tensor_mv)

    # Create a new function with the target module's globals
    # This ensures that internal functions like _read_tensor_data_mmap are resolved correctly
    patched_func = types.FunctionType(
        _process_output_file_impl.__code__,
        hf_module.__dict__,  # Use target module's globals for symbol resolution
        _process_output_file_impl.__name__,
        _process_output_file_impl.__defaults__,
        _process_output_file_impl.__closure__,
    )

    hf_module._process_output_file = patched_func
    _dcp_consolidation_patch_applied = True
