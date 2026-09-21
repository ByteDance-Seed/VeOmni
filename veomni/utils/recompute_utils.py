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

from typing import Any, List

import torch

from veomni.utils import helper


logger = helper.create_logger(__name__)


def string_to_op(op_string: str) -> Any:
    """
    Convert a single operation string to PyTorch operation object

    Args:
        op_string: e.g. "aten.addmm.default" or "torch.ops.flash_attn._flash_attn_forward.default"

    Returns:
        PyTorch operation object
    """
    global torch
    # Clean the string
    clean_string = op_string.strip()

    # Remove torch.ops. prefix (if exists)
    if clean_string.startswith("torch.ops."):
        clean_string = clean_string[len("torch.ops.") :]

    # Split path and access level by level
    parts = clean_string.split(".")

    # Check if torch.ops is available
    if not hasattr(torch, "ops"):
        raise AttributeError("torch.ops not available in this PyTorch version")

    current = torch.ops

    # Special handling: ensure accessing aten operations by first trying to trigger registration
    if parts[0] == "aten":
        try:
            # Try to access a basic aten operation to trigger module loading
            _ = torch.ops.aten.add
        except AttributeError:
            # If cannot access aten, may need to import related modules
            try:
                import torch._C._dispatch
            except ImportError:
                pass

    for i, part in enumerate(parts):
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            # More detailed error information, including current path
            current_path = ".".join(parts[:i])
            available_attrs = dir(current) if hasattr(current, "__dict__") else []
            raise AttributeError(
                f"Operation '{op_string}' not found. "
                f"Missing attribute: '{part}' at path 'torch.ops.{current_path}'. "
                f"Available attributes: {available_attrs[:10]}{'...' if len(available_attrs) > 10 else ''}"
            )

    return current


def convert_ops_to_objects(ops_strings: List[str]) -> List[Any]:
    """
    Convert operation string list to operation object list
    Args:
        ops_strings: String list

    Returns:
        PyTorch operation object list
    """
    ops_objects = []
    failed_ops = []

    # First perform environment check
    _check_torch_ops_availability()

    for op_str in ops_strings:
        try:
            op_obj = string_to_op(op_str)
            ops_objects.append(op_obj)
            logger.info_rank0(f"✓ Conversion successful: {op_str}")
            assert isinstance(op_obj, torch._ops.OpOverload), "Please check if the ops is end with .default"
        except (AttributeError, TypeError) as e:
            logger.info_rank0(f"✗ Conversion failed: {op_str} - {e}")
            failed_ops.append(op_str)
        except Exception as e:
            logger.info_rank0(f"✗ Conversion failed: {op_str} - {e}")
            raise e

    if failed_ops:
        logger.info_rank0(f"\nWarning: {len(failed_ops)} operations failed to convert")
        logger.info_rank0("Possible reasons:")
        logger.info_rank0("1. PyTorch version does not support certain operations")
        logger.info_rank0("2. Missing related extension modules (e.g. flash_attn)")
        logger.info_rank0("3. Operation name spelling error")

    return ops_objects


def _check_torch_ops_availability():
    global torch
    # Check if torch.ops is available
    if not hasattr(torch, "ops"):
        raise RuntimeError("torch.ops is not available in current PyTorch version")

    # Check basic aten operations
    try:
        _ = torch.ops.aten.add
        logger.info_rank0("✓ torch.ops.aten available")
    except AttributeError as e:
        logger.info_rank0(f"✗ torch.ops.aten not available: {e}")
        logger.info_rank0("Trying to import necessary modules...")
        try:
            import torch._C._dispatch

            logger.info_rank0("✓ Successfully imported torch._C._dispatch")
        except ImportError as e:
            logger.info_rank0(f"✗ Cannot import torch._C._dispatch: {e}")
