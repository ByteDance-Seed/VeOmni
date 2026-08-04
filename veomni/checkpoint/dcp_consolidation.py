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

"""Patches for PyTorch DCP (Distributed Checkpoint) safetensors consolidation.

This module provides monkey patches for:
- Distributed file systems (e.g., HDFS via FUSE) that do not support r+b
  (read-write binary) file mode for random write access.
- Integer and boolean tensors, whose byte sizes PyTorch's consolidator computes
  with ``torch.finfo`` even though it only accepts floating-point dtypes.
"""

import hashlib
import inspect
import types


_dcp_consolidation_patch_applied = False

# Fixed torch versions for this patch - update when upgrading torch
_SUPPORTED_TORCH_VERSION_PREFIXES = ("2.9", "2.10", "2.11")
_EXPECTED_PROCESS_OUTPUT_FILE_ARGS = ("output_file", "output_data", "input_files_data")
_EXPECTED_PARSE_INPUT_METADATA_ARGS = ("input_files_data", "output_files_data")
_SUPPORTED_PROCESS_OUTPUT_FILE_SHA256 = {
    # torch 2.9.1
    "0837813477b4ca319890ef671b954f83bbe966f21a751875606b74e4e8e30ea8",
    # torch 2.10.0
    "794961d5aab4bcc419b08e9a41fbb4c80cb70ff19b07847c96f00cc193cee3a2",
    # torch 2.11.0 upstream source
    "ff25a85cc52018707334f1206760fe186146771e5357388f0b4d6bc19bdf61c1",
    # torch 2.11.0+cu130 CI wheel
    "433c9d026092f48f5ba02631975294de1a8ae98e020d5cb6ffd0f5db760476fe",
}
_SUPPORTED_PARSE_INPUT_METADATA_SHA256 = {
    # torch 2.9.1, 2.10.0, and 2.11.0 (upstream source and cu130 CI wheel)
    "f6c476c9467a32928c8b29c211988c78a14a934bc4cbc5763a3882a7fc3b11f0",
}


def apply_dcp_consolidation_patch():
    """Patch DCP safetensors consolidation for HDFS FUSE and non-floating tensors.

    The original implementation in PyTorch uses r+b mode for random write access:
        with open(output_file, "r+b") as output_stream:
            output_stream.seek(0, os.SEEK_END)
            ...

    This is not supported by some append only file systems (e.g., HDFS via FUSE).
    This patch replaces the function to use append mode instead:

        with open(output_file, "ab") as output_stream:
            ...

    PyTorch also computes every tensor's byte size with ``torch.finfo`` during
    consolidation. This fails for valid safetensors dtypes such as int64 and
    bool. The patched metadata parser uses ``Tensor.element_size()`` instead,
    which handles every torch dtype supported by safetensors.

    Note: Append mode requires tensors to be processed in offset order, which
    is already ensured by sorting tensors before writing.

    The patch uses types.FunctionType to create a new function with __globals__
    bound to the target module, enabling access to internal functions like
    _read_tensor_data_mmap and _write_sub_tensor_to_file_optimized.
    """
    global _dcp_consolidation_patch_applied

    if _dcp_consolidation_patch_applied:
        return

    # Verify torch version matches a known-compatible implementation.
    import torch

    if not torch.__version__.startswith(_SUPPORTED_TORCH_VERSION_PREFIXES):
        raise RuntimeError(
            f"DCP consolidation patch requires torch {_SUPPORTED_TORCH_VERSION_PREFIXES}, "
            f"but got {torch.__version__}. Please update the patch or verify compatibility."
        )

    import torch.distributed.checkpoint._consolidate_hf_safetensors as hf_module

    if not hasattr(hf_module, "_process_output_file"):
        raise RuntimeError(
            f"torch.distributed.checkpoint._consolidate_hf_safetensors does not have "
            f"_process_output_file attribute. Please verify torch {_SUPPORTED_TORCH_VERSION_PREFIXES} compatibility."
        )

    if not hasattr(hf_module, "_parse_input_metadata"):
        raise RuntimeError(
            "torch.distributed.checkpoint._consolidate_hf_safetensors does not have "
            f"_parse_input_metadata attribute. Please verify torch {_SUPPORTED_TORCH_VERSION_PREFIXES} compatibility."
        )

    process_output_file = hf_module._process_output_file
    process_output_file_args = tuple(inspect.signature(process_output_file).parameters)
    if process_output_file_args != _EXPECTED_PROCESS_OUTPUT_FILE_ARGS:
        raise RuntimeError(
            "torch.distributed.checkpoint._consolidate_hf_safetensors._process_output_file "
            f"signature changed from {_EXPECTED_PROCESS_OUTPUT_FILE_ARGS} to {process_output_file_args}. "
            "Please update the DCP consolidation patch."
        )

    process_output_file_source = inspect.getsource(process_output_file)
    process_output_file_hash = hashlib.sha256(process_output_file_source.encode()).hexdigest()
    if process_output_file_hash not in _SUPPORTED_PROCESS_OUTPUT_FILE_SHA256:
        raise RuntimeError(
            "torch.distributed.checkpoint._consolidate_hf_safetensors._process_output_file "
            f"source hash {process_output_file_hash} is not in the verified set. "
            "Please update the DCP consolidation patch."
        )

    parse_input_metadata = hf_module._parse_input_metadata
    parse_input_metadata_args = tuple(inspect.signature(parse_input_metadata).parameters)
    if parse_input_metadata_args != _EXPECTED_PARSE_INPUT_METADATA_ARGS:
        raise RuntimeError(
            "torch.distributed.checkpoint._consolidate_hf_safetensors._parse_input_metadata "
            f"signature changed from {_EXPECTED_PARSE_INPUT_METADATA_ARGS} to {parse_input_metadata_args}. "
            "Please update the DCP consolidation patch."
        )

    parse_input_metadata_source = inspect.getsource(parse_input_metadata)
    parse_input_metadata_hash = hashlib.sha256(parse_input_metadata_source.encode()).hexdigest()
    if parse_input_metadata_hash not in _SUPPORTED_PARSE_INPUT_METADATA_SHA256:
        raise RuntimeError(
            "torch.distributed.checkpoint._consolidate_hf_safetensors._parse_input_metadata "
            f"source hash {parse_input_metadata_hash} is not in the verified set. "
            "Please update the DCP consolidation patch."
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

    def _parse_input_metadata_impl(input_files_data, output_files_data):
        from safetensors.torch import _getdtype

        fqn_to_size_mapping = {}

        for file_data in input_files_data.values():
            safetensors_metadata = file_data.metadata
            dcp_sharding_info = _get_dcp_custom_metadata(safetensors_metadata)
            if not dcp_sharding_info:
                raise ValueError(
                    "No DCP custom metadata found in safetensors file. "
                    "The file must be saved with DCP to be consolidated."
                )

            for key, val in safetensors_metadata.items():
                if key == DEFAULT_EXTRA_METADATA_KEY:
                    continue

                sizes = val[SHAPE_KEY]
                offsets = dcp_sharding_info[key][SAVED_OFFSETS_KEY]

                if key not in fqn_to_size_mapping:
                    cur_size = [size + offset for size, offset in zip(sizes, offsets)]
                    fqn_to_size_mapping[key] = (cur_size, val[DTYPE_KEY])
                else:
                    cur_size = fqn_to_size_mapping[key][0]
                    for i in range(len(sizes)):
                        cur_size[i] = max(cur_size[i], sizes[i] + offsets[i])

        for fqn, tensor_info in fqn_to_size_mapping.items():
            tensor_size, dtype_str = tensor_info
            dtype_size = torch.empty((), dtype=_getdtype(dtype_str)).element_size()
            for output_data in output_files_data.values():
                if fqn in output_data.fqn_data:
                    output_data.fqn_data[fqn] = _FqnData(
                        shape_in_file=tensor_size,
                        dtype_size=dtype_size,
                        dtype_str=dtype_str,
                    )

    # Create a new function with the target module's globals
    # This ensures that internal functions like _read_tensor_data_mmap are resolved correctly
    patched_func = types.FunctionType(
        _process_output_file_impl.__code__,
        hf_module.__dict__,  # Use target module's globals for symbol resolution
        _process_output_file_impl.__name__,
        _process_output_file_impl.__defaults__,
        _process_output_file_impl.__closure__,
    )

    patched_parse_input_metadata = types.FunctionType(
        _parse_input_metadata_impl.__code__,
        hf_module.__dict__,
        _parse_input_metadata_impl.__name__,
        _parse_input_metadata_impl.__defaults__,
        _parse_input_metadata_impl.__closure__,
    )

    hf_module._process_output_file = patched_func
    hf_module._parse_input_metadata = patched_parse_input_metadata
    _dcp_consolidation_patch_applied = True
