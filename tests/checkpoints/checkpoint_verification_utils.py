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


def _normalize_key(key: str) -> Optional[str]:
    """
    Convert DCP key to HuggingFace format. Returns None for non-model weights.

    This function mirrors the key normalization logic in scripts/merge_dcp_to_hf.py
    to ensure consistent behavior between conversion and verification.

    Conversion rules:
    - "model.model.*" -> "model.*" (remove first "model." prefix)
    - "model.lm_head.weight" -> "lm_head.weight" (special case)
    - Other "model.*" keys -> log warning and strip "model." prefix
    - Keys without "model." prefix -> None (non-model weights)
    """
    if not key.startswith("model."):
        return None

    if key.startswith("model.model."):
        # Standard case: model.model.* -> model.*
        return key[6:]  # Remove first "model." prefix
    elif key == "model.lm_head.weight":
        # Special case: model.lm_head.weight -> lm_head.weight
        return "lm_head.weight"
    else:
        # Other keys with single "model." prefix - log and strip prefix
        logger.warning(
            f"Found key with single 'model.' prefix that doesn't match expected patterns: '{key}'. "
            f"Converting to '{key[6:]}' by stripping 'model.' prefix."
        )
        return key[6:]


def load_dcp_checkpoint(dcp_checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load a DCP (Distributed Checkpoint) checkpoint from disk and extract model weights.

    This function uses a DIFFERENT approach than merge_dcp_to_hf.py for cross-validation:
    1. Read metadata and get ALL keys (including optimizer states, etc.)
    2. Pre-allocate tensors for ALL keys
    3. Load everything in one go
    4. Filter out non-model weights after loading
    5. Normalize keys to HuggingFace format

    This independent implementation provides cross-validation for the conversion script.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.

    Returns:
        State dict with model weights in HuggingFace format (normalized keys).
    """
    from collections import OrderedDict

    from torch.distributed.checkpoint.metadata import Metadata

    logger.info(f"Loading DCP checkpoint from {dcp_checkpoint_dir}")
    logger.info("Reading metadata and loading ALL tensors, then filtering for model weights")

    # Step 1: Read metadata to get all keys
    reader = FileSystemReader(dcp_checkpoint_dir)
    metadata = reader.read_metadata()

    if not isinstance(metadata, Metadata):
        raise ValueError(f"Invalid metadata format in {dcp_checkpoint_dir}")

    # Step 2: Pre-allocate placeholder tensors for ALL keys (not just model weights)
    # Note: Some keys may have BytesStorageMetadata (non-tensor data), skip those
    state_dict = OrderedDict()
    skipped_keys = []

    for dcp_key, tensor_metadata in metadata.state_dict_metadata.items():
        # Check if this is tensor metadata (has 'properties' attribute)
        if not hasattr(tensor_metadata, "properties"):
            skipped_keys.append(dcp_key)
            continue

        # Check if dtype is available
        if not hasattr(tensor_metadata.properties, "dtype"):
            logger.warning(f"Skipping key '{dcp_key}': no dtype information in metadata")
            skipped_keys.append(dcp_key)
            continue

        state_dict[dcp_key] = torch.empty(
            tensor_metadata.size,
            dtype=tensor_metadata.properties.dtype,
        )

    logger.info(f"Found {len(state_dict)} tensor keys in DCP checkpoint")
    if skipped_keys:
        logger.info(f"Skipped {len(skipped_keys)} non-tensor keys (e.g., optimizer config)")

    # Step 3: Load ALL tensors from checkpoint
    load(
        state_dict,
        checkpoint_id=dcp_checkpoint_dir,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        no_dist=True,
    )

    logger.info(f"Loaded {len(state_dict)} total tensors from DCP")

    # Step 4: Filter for model weights and normalize keys to HuggingFace format
    loaded_state_dict = {}
    non_model_count = 0

    for dcp_key, tensor in state_dict.items():
        hf_key = _normalize_key(dcp_key)

        if hf_key is None:
            # Skip non-model keys (optimizer states, etc.)
            non_model_count += 1
            continue

        if not torch.is_tensor(tensor):
            logger.warning(f"Skipping non-tensor key: {dcp_key}")
            continue

        # Handle DTensor (distributed tensor)
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()

        loaded_state_dict[hf_key] = tensor.detach().cpu()

    logger.info(f"✓ Extracted {len(loaded_state_dict)} model weight tensors")
    logger.info(f"✓ Filtered out {non_model_count} non-model tensors")

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
) -> bool:
    """
    Verify that the HuggingFace checkpoint weights match the original state dict.

    Args:
        hf_checkpoint_dir: Directory containing the saved HF checkpoint.
        original_state_dict: Original state dict to compare against (with HF-format keys).
            Should be in the same dtype as the HF checkpoint (typically bf16).
        safe_serialization: Whether the checkpoint uses safetensors format.
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
                logger.error(f"Missing keys ({len(missing_keys)}): {sorted(missing_keys)[:10]}...")
            if extra_keys:
                logger.error(f"Extra keys ({len(extra_keys)}): {sorted(extra_keys)[:10]}...")
            return False

        logger.info(f"✓ All {len(original_keys)} keys match between original and loaded checkpoints")

        # Compare ALL tensor values
        logger.info(f"Verifying all {len(original_keys)} tensors...")

        mismatches = []
        for key in sorted(original_keys):
            original_tensor = original_state_dict[key]
            loaded_tensor = loaded_state_dict[key]

            # Check shape
            if original_tensor.shape != loaded_tensor.shape:
                logger.error(f"Shape mismatch for key '{key}': {original_tensor.shape} vs {loaded_tensor.shape}")
                return False

            # Check dtype matches (both should be bf16 now)
            if original_tensor.dtype != loaded_tensor.dtype:
                logger.error(f"Dtype mismatch for key '{key}': {original_tensor.dtype} vs {loaded_tensor.dtype}")
                return False

            # Direct comparison since both tensors should be in the same dtype (bf16)
            if not torch.equal(original_tensor.cpu(), loaded_tensor.cpu()):
                diff = (original_tensor.cpu().float() - loaded_tensor.cpu().float()).abs().max().item()
                mismatches.append((key, diff))
                logger.warning(f"Value mismatch for key '{key}', max diff: {diff}")

        if mismatches:
            logger.error(f"Found {len(mismatches)} tensor(s) with value mismatches:")
            for key, max_diff in mismatches[:10]:  # Show first 10
                logger.error(f"  - {key}: max_diff={max_diff}")
            return False

        logger.info(f"✓ Verified {len(original_keys)} tensor(s) - all values match exactly")
        logger.info("✓ HuggingFace checkpoint weight verification passed!")
        return True

    except Exception as e:
        logger.error(f"Verification failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_hf_checkpoint(
    hf_checkpoint_dir: str,
    original_state_dict: Dict[str, torch.Tensor],
    safe_serialization: bool = True,
) -> bool:
    """
    Comprehensive verification of a HuggingFace checkpoint.

    Args:
        hf_checkpoint_dir: Directory containing the saved HF checkpoint.
        original_state_dict: Original state dict to compare against (with HF-format keys).
            Should be in the same dtype as the HF checkpoint for exact comparison.
        safe_serialization: Whether the checkpoint uses safetensors format.
        rtol: Relative tolerance for value comparison (kept for API compatibility, not used).
        atol: Absolute tolerance for value comparison (kept for API compatibility, not used).

    Returns:
        True if all verifications pass, False otherwise.
    """
    logger.info("=" * 80)
    logger.info("Starting HuggingFace checkpoint verification")
    logger.info("=" * 80)

    # Verify structure
    if not verify_hf_checkpoint_structure(hf_checkpoint_dir, safe_serialization):
        logger.error("Structure verification failed!")
        return False

    # Verify weights (exact match, rtol/atol not used when both are bf16)
    if not verify_hf_checkpoint_weights(hf_checkpoint_dir, original_state_dict, safe_serialization):
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
) -> bool:
    """
    Verify DCP to HuggingFace checkpoint conversion by comparing weights.

    This function:
    1. Loads ALL tensors from DCP checkpoint (different approach than merge_dcp_to_hf.py)
    2. Filters for model weights and normalizes keys to HF format
    3. Converts DCP weights (fp32) to bf16 to match HF checkpoint dtype
    4. Compares with the converted HuggingFace checkpoint using exact match (rtol=0, atol=0)

    Since HF checkpoints are saved in bf16, we convert the DCP fp32 weights to bf16
    before comparison. This ensures exact match verification without tolerance.

    This provides independent cross-validation of the conversion script.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.
        hf_checkpoint_dir: Directory containing the HF checkpoint.
        safe_serialization: Whether the HF checkpoint uses safetensors format.

    Returns:
        True if verification passes, False otherwise.
    """
    logger.info("=" * 80)
    logger.info("Starting DCP to HuggingFace conversion verification")
    logger.info("=" * 80)

    try:
        # Load DCP checkpoint using simplified approach (load all, then filter)
        dcp_state_dict = load_dcp_checkpoint(dcp_checkpoint_dir)
    except Exception as e:
        logger.error(f"Failed to load DCP checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Convert DCP state dict from fp32 to bf16 to match HF checkpoint dtype
    logger.info("Converting DCP weights from fp32 to bf16 for exact comparison...")
    dcp_state_dict_bf16 = {key: tensor.to(torch.bfloat16) for key, tensor in dcp_state_dict.items()}
    logger.info(f"✓ Converted {len(dcp_state_dict_bf16)} tensors to bf16")

    # Verify the HF checkpoint against DCP state dict (bf16 vs bf16, exact match)
    return verify_hf_checkpoint(
        hf_checkpoint_dir=hf_checkpoint_dir,
        original_state_dict=dcp_state_dict_bf16,
        safe_serialization=safe_serialization,
    )
