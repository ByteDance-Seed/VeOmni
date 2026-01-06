#!/usr/bin/env python
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

"""
Utilities for verifying checkpoint conversions between DCP and HuggingFace formats.
"""

import json
import logging
import os
from typing import Dict, Optional

import torch
from torch.distributed.checkpoint import FileSystemReader, load
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME


try:
    from safetensors.torch import load_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_dcp_checkpoint(dcp_checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load a DCP (Distributed Checkpoint) checkpoint from disk.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.

    Returns:
        State dict loaded from the checkpoint.
    """
    import gc
    from collections import OrderedDict

    from torch.distributed.checkpoint.metadata import Metadata

    logger.info(f"Loading DCP checkpoint from {dcp_checkpoint_dir}")

    # Step 1: Get all keys from metadata
    reader = FileSystemReader(dcp_checkpoint_dir)
    metadata = reader.read_metadata()

    all_keys = []
    if isinstance(metadata, Metadata):
        for key in metadata.state_dict_metadata.keys():
            if key.startswith("model."):
                all_keys.append(key)

    logger.info(f"Found {len(all_keys)} model keys in DCP checkpoint")

    # Step 2: Pre-initialize state_dict with placeholder tensors
    state_dict = OrderedDict()
    for key in all_keys:
        tensor_metadata = metadata.state_dict_metadata[key]
        state_dict[key] = torch.empty(
            tensor_metadata.size,
            dtype=tensor_metadata.properties.dtype if hasattr(tensor_metadata.properties, "dtype") else torch.float32,
        )

    # Step 3: Load the checkpoint
    logger.info("Loading tensors from DCP (this may take a while)...")
    load(
        state_dict,
        checkpoint_id=dcp_checkpoint_dir,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        no_dist=True,
    )

    logger.info(f"Loaded {len(state_dict)} tensors")

    # Step 4: Process tensors
    logger.info("Processing tensors...")
    loaded_state_dict = {}
    total_keys = len(state_dict)

    for idx, (key, tensor) in enumerate(state_dict.items(), 1):
        if not torch.is_tensor(tensor):
            continue

        # Handle DTensor (distributed tensor)
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()

        # Convert DCP key to HuggingFace format:
        # - "model.model.*" -> "model.*" (remove first "model." prefix)
        # - "model.lm_head.weight" -> "lm_head.weight" (special case)
        if key.startswith("model.model."):
            hf_key = key[6:]  # Remove first "model." prefix
        elif key == "model.lm_head.weight":
            hf_key = "lm_head.weight"
        else:
            # Keep other keys as-is after removing "model." prefix
            hf_key = key[6:]
        loaded_state_dict[hf_key] = tensor.detach().cpu()

        # Show progress
        if idx % max(10, total_keys // 10) == 0 or idx == total_keys:
            logger.info(f"  Processed {idx}/{total_keys} tensors ({idx * 100 // total_keys}%)")

        # Periodic garbage collection
        if idx % 10 == 0:
            gc.collect()

    logger.info(f"✓ Successfully loaded {len(loaded_state_dict)} tensors from DCP checkpoint")
    return loaded_state_dict


def load_hf_checkpoint(hf_checkpoint_dir: str, safe_serialization: bool = True) -> Dict[str, torch.Tensor]:
    """
    Load a HuggingFace checkpoint from disk.

    Args:
        hf_checkpoint_dir: Directory containing the HF checkpoint.
        safe_serialization: Whether the checkpoint uses safetensors format.

    Returns:
        State dict loaded from the checkpoint.
    """
    if safe_serialization:
        weight_files = [f for f in os.listdir(hf_checkpoint_dir) if f.endswith(".safetensors")]
        index_file = SAFE_WEIGHTS_INDEX_NAME
    else:
        weight_files = [f for f in os.listdir(hf_checkpoint_dir) if f.endswith(".bin")]
        index_file = WEIGHTS_INDEX_NAME

    loaded_state_dict = {}

    if len(weight_files) == 1:
        # Single file checkpoint
        weight_file = os.path.join(hf_checkpoint_dir, weight_files[0])
        if safe_serialization:
            if not SAFETENSORS_AVAILABLE:
                raise ImportError("safetensors is not available. Please install it with: pip install safetensors")
            loaded_state_dict = load_file(weight_file)
        else:
            loaded_state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
    else:
        # Sharded checkpoint - load from index
        index_path = os.path.join(hf_checkpoint_dir, index_file)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        with open(index_path) as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        files_to_load = set(weight_map.values())

        for file_name in files_to_load:
            file_path = os.path.join(hf_checkpoint_dir, file_name)
            if safe_serialization:
                if not SAFETENSORS_AVAILABLE:
                    raise ImportError("safetensors is not available. Please install it with: pip install safetensors")
                shard_dict = load_file(file_path)
            else:
                shard_dict = torch.load(file_path, map_location="cpu", weights_only=True)
            loaded_state_dict.update(shard_dict)

    return loaded_state_dict


def verify_hf_checkpoint_structure(hf_checkpoint_dir: str, safe_serialization: bool = True) -> bool:
    """
    Verify that the HuggingFace checkpoint has the correct file structure.

    Args:
        hf_checkpoint_dir: Directory containing the saved HF checkpoint.
        safe_serialization: Whether the checkpoint uses safetensors format.

    Returns:
        True if structure verification passes, False otherwise.
    """
    logger.info(f"Verifying HuggingFace checkpoint structure at {hf_checkpoint_dir}")

    # Check if directory exists
    if not os.path.exists(hf_checkpoint_dir):
        logger.error(f"Checkpoint directory does not exist: {hf_checkpoint_dir}")
        return False

    # Check for weight files
    if safe_serialization:
        weight_files = [f for f in os.listdir(hf_checkpoint_dir) if f.endswith(".safetensors")]
        index_file = SAFE_WEIGHTS_INDEX_NAME
    else:
        weight_files = [f for f in os.listdir(hf_checkpoint_dir) if f.endswith(".bin")]
        index_file = WEIGHTS_INDEX_NAME

    if len(weight_files) == 0:
        logger.error(f"No weight files found in {hf_checkpoint_dir}")
        return False

    logger.info(f"✓ Found {len(weight_files)} weight file(s): {weight_files}")

    # Check for index file if sharded
    if len(weight_files) > 1:
        index_path = os.path.join(hf_checkpoint_dir, index_file)
        if not os.path.exists(index_path):
            logger.error(f"Index file not found for sharded checkpoint: {index_path}")
            return False
        logger.info(f"✓ Found index file: {index_file}")

    logger.info("✓ Checkpoint structure verification passed!")
    return True


def verify_hf_checkpoint_weights(
    hf_checkpoint_dir: str,
    original_state_dict: Dict[str, torch.Tensor],
    safe_serialization: bool = True,
    num_keys_to_check: Optional[int] = None,
    rtol: float = 1e-3,
    atol: float = 5e-4,
) -> bool:
    """
    Verify that the HuggingFace checkpoint weights match the original state dict.

    Args:
        hf_checkpoint_dir: Directory containing the saved HF checkpoint.
        original_state_dict: Original state dict to compare against.
        safe_serialization: Whether the checkpoint uses safetensors format.
        num_keys_to_check: Number of keys to verify (None = all keys). For large models, checking a subset is faster.
        rtol: Relative tolerance for value comparison.
        atol: Absolute tolerance for value comparison.

    Returns:
        True if verification passes, False otherwise.
    """
    logger.info(f"Verifying HuggingFace checkpoint weights at {hf_checkpoint_dir}")

    try:
        # Load the saved weights back
        loaded_state_dict = load_hf_checkpoint(hf_checkpoint_dir, safe_serialization)

        # Compare keys
        original_keys = set(original_state_dict.keys())
        loaded_keys = set(loaded_state_dict.keys())

        if original_keys != loaded_keys:
            missing_keys = original_keys - loaded_keys
            extra_keys = loaded_keys - original_keys
            logger.error("Key mismatch detected!")
            if missing_keys:
                logger.error(f"Missing keys ({len(missing_keys)}): {list(missing_keys)[:10]}...")
            if extra_keys:
                logger.error(f"Extra keys ({len(extra_keys)}): {list(extra_keys)[:10]}...")
            return False

        logger.info(f"✓ All {len(original_keys)} keys match between original and loaded checkpoints")

        # Compare tensor values
        if num_keys_to_check is None:
            keys_to_check = list(original_keys)
        else:
            num_keys_to_check = min(num_keys_to_check, len(original_keys))
            keys_to_check = list(original_keys)[:num_keys_to_check]

        logger.info(f"Verifying {len(keys_to_check)} tensor(s)...")

        mismatches = []
        for key in keys_to_check:
            original_tensor = original_state_dict[key]
            loaded_tensor = loaded_state_dict[key]

            # Check shape
            if original_tensor.shape != loaded_tensor.shape:
                logger.error(f"Shape mismatch for key '{key}': {original_tensor.shape} vs {loaded_tensor.shape}")
                return False

            # Check values (with tolerance for dtype conversion)
            # Convert to float for comparison to handle different dtypes
            original_float = original_tensor.cpu().float()
            loaded_float = loaded_tensor.cpu().float()

            if not torch.allclose(original_float, loaded_float, rtol=rtol, atol=atol):
                max_diff = (original_float - loaded_float).abs().max().item()
                mismatches.append((key, max_diff))
                logger.warning(f"Value mismatch for key '{key}', max diff: {max_diff}")

        if mismatches:
            logger.error(f"Found {len(mismatches)} tensor(s) with value mismatches:")
            for key, max_diff in mismatches[:5]:  # Show first 5
                logger.error(f"  - {key}: max_diff={max_diff}")
            return False

        logger.info(f"✓ Verified {len(keys_to_check)} tensor(s) - all values match (rtol={rtol}, atol={atol})")
        logger.info("✓ HuggingFace checkpoint weight verification passed!")
        return True

    except Exception as e:
        logger.error(f"Verification failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_hf_checkpoint(
    hf_checkpoint_dir: str,
    original_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    safe_serialization: bool = True,
    verify_structure: bool = True,
    verify_weights: bool = True,
    num_keys_to_check: Optional[int] = 10,
    rtol: float = 1e-3,
    atol: float = 5e-4,
) -> bool:
    """
    Comprehensive verification of a HuggingFace checkpoint.

    Args:
        hf_checkpoint_dir: Directory containing the saved HF checkpoint.
        original_state_dict: Original state dict to compare against (required if verify_weights=True).
        safe_serialization: Whether the checkpoint uses safetensors format.
        verify_structure: Whether to verify file structure.
        verify_weights: Whether to verify weight values against original state dict.
        num_keys_to_check: Number of keys to verify (None = all keys).
        rtol: Relative tolerance for value comparison.
        atol: Absolute tolerance for value comparison.

    Returns:
        True if all requested verifications pass, False otherwise.
    """
    logger.info("=" * 80)
    logger.info("Starting HuggingFace checkpoint verification")
    logger.info("=" * 80)

    # Verify structure
    if verify_structure:
        if not verify_hf_checkpoint_structure(hf_checkpoint_dir, safe_serialization):
            logger.error("Structure verification failed!")
            return False

    # Verify weights
    if verify_weights:
        if original_state_dict is None:
            logger.error("Cannot verify weights without original_state_dict!")
            return False

        if not verify_hf_checkpoint_weights(
            hf_checkpoint_dir, original_state_dict, safe_serialization, num_keys_to_check, rtol, atol
        ):
            logger.error("Weight verification failed!")
            return False

    logger.info("=" * 80)
    logger.info("✓ All verifications passed!")
    logger.info("=" * 80)
    return True


def verify_dcp_to_hf_conversion(
    dcp_checkpoint_dir: str,
    hf_checkpoint_dir: str,
    safe_serialization: bool = True,
    verify_structure: bool = True,
    verify_weights: bool = True,
    num_keys_to_check: Optional[int] = None,
    rtol: float = 1e-3,
    atol: float = 5e-4,
) -> bool:
    """
    Verify DCP to HuggingFace checkpoint conversion by comparing weights.

    This function loads weights from the DCP checkpoint and compares them
    with the converted HuggingFace checkpoint.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.
        hf_checkpoint_dir: Directory containing the HF checkpoint.
        safe_serialization: Whether the HF checkpoint uses safetensors format.
        verify_structure: Whether to verify HF file structure.
        verify_weights: Whether to verify weight values.
        num_keys_to_check: Number of keys to verify (None = all keys).
        rtol: Relative tolerance for value comparison.
        atol: Absolute tolerance for value comparison.

    Returns:
        True if all requested verifications pass, False otherwise.
    """
    logger.info("=" * 80)
    logger.info("Starting DCP to HuggingFace conversion verification")
    logger.info("=" * 80)

    # Load original DCP checkpoint if weight verification is needed
    original_state_dict = None
    if verify_weights:
        try:
            original_state_dict = load_dcp_checkpoint(dcp_checkpoint_dir)
        except Exception as e:
            logger.error(f"Failed to load DCP checkpoint: {e}")
            import traceback

            traceback.print_exc()
            return False

    # Verify the HF checkpoint
    return verify_hf_checkpoint(
        hf_checkpoint_dir=hf_checkpoint_dir,
        original_state_dict=original_state_dict,
        safe_serialization=safe_serialization,
        verify_structure=verify_structure,
        verify_weights=verify_weights,
        num_keys_to_check=num_keys_to_check,
        rtol=rtol,
        atol=atol,
    )
