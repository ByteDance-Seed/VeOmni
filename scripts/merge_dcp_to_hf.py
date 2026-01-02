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
Self-contained script to merge DCP (Distributed Checkpoint) format to HuggingFace format.

This script loads a DCP checkpoint and converts it to HuggingFace format (.safetensors or .bin).
It can optionally include model config and processor from a model assets directory.

Usage:
    python merge_dcp_to_hf.py --load-dir <dcp_checkpoint_path> [--save-dir <output_path>] [--model-assets-dir <assets_path>]
"""

import argparse
import json
import logging
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from transformers import AutoConfig, AutoProcessor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size in bytes of a given dtype."""
    size_map = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
    }
    return size_map.get(dtype, 4)


def dcp_to_torch_state_dict(save_checkpoint_path: Union[str, os.PathLike]) -> STATE_DICT_TYPE:
    """
    Convert a DCP checkpoint directory into a Torch state_dict.

    This function only loads model weights (ignoring optimizer states, etc.) to reduce memory usage.

    Args:
        save_checkpoint_path: Directory containing the DCP checkpoint.

    Returns:
        state_dict: Dictionary containing model weights.

    Warning:
        To avoid OOM, it's recommended to only run this function on a single rank.
    """
    logger.info(f"Loading DCP checkpoint from {save_checkpoint_path}")

    # Load the state_dict from the DCP checkpoint
    state_dict: STATE_DICT_TYPE = {}

    # Only load model weights (excluding optimizer states, etc.)
    # keys=["model"] will only load model state dict
    # See _EmptyStateDictLoadPlanner._should_include_key() in torch's default_planner.py
    keys = ["model"]

    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(save_checkpoint_path),
        planner=_EmptyStateDictLoadPlanner(keys=keys),
        no_dist=True,
    )

    # Handle flattened state dicts
    if "state" in state_dict:
        # This happens when the model state dicts are flattened during saving
        state_dict = state_dict["state"]

    # Extract model state dict
    model_state_dict = state_dict.get("model", state_dict)

    logger.info(f"Loaded {len(model_state_dict)} keys from DCP checkpoint")

    # Filter out keys that have no actual tensor data (empty model scenario)
    filtered_state_dict = {}
    for key, value in model_state_dict.items():
        if torch.is_tensor(value) and value.numel() > 0:
            filtered_state_dict[key] = value
        else:
            logger.warning(f"Skipping key '{key}' with no tensor data")

    logger.info(f"After filtering, {len(filtered_state_dict)} keys remain")

    return filtered_state_dict


def _get_shard_info(
    state_dict: Dict[str, torch.Tensor],
    save_dtype: Optional[Union[str, torch.dtype]],
    shard_size: int,
    safe_serialization: bool,
) -> tuple[bool, int, Dict[str, str]]:
    """
    Calculate shard information for splitting large state dicts.

    Args:
        state_dict: The state dict to shard.
        save_dtype: Target dtype for saving.
        shard_size: Maximum size per shard in bytes.
        safe_serialization: Whether using safetensors format.

    Returns:
        Tuple of (is_sharded, total_size, weight_map)
    """
    current_size, total_size = 0, 0
    current_shard, shard_list = [], []

    for name, tensor in state_dict.items():
        if isinstance(save_dtype, str):
            dtype = getattr(torch, save_dtype)
        elif isinstance(save_dtype, torch.dtype):
            dtype = save_dtype
        else:
            dtype = tensor.dtype

        tensor_size = tensor.numel() * get_dtype_size(dtype)

        if current_size != 0 and current_size + tensor_size > shard_size:
            total_size += current_size
            shard_list.append(current_shard)
            current_size = 0
            current_shard = []

        current_size += tensor_size
        current_shard.append(name)

    if current_size != 0:
        total_size += current_size
        shard_list.append(current_shard)

    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    num_shards = len(shard_list)
    weight_map = OrderedDict()

    if num_shards == 1:
        is_sharded = False
        for name in shard_list[0]:
            weight_map[name] = weights_name
    else:
        is_sharded = True
        for shard_idx, shard in enumerate(shard_list):
            prefix, extension = weights_name.rsplit(".", maxsplit=1)
            file_name = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"
            for name in shard:
                weight_map[name] = file_name

    return is_sharded, total_size, weight_map


def _save_state_dict(
    state_dict: Dict[str, torch.Tensor],
    path_to_save: os.PathLike,
    safe_serialization: bool,
) -> None:
    """
    Save state dict to disk.

    Args:
        state_dict: State dict to save.
        path_to_save: File path to save to.
        safe_serialization: If True, use safetensors; otherwise use pickle.
    """
    if safe_serialization:
        save_file(state_dict, path_to_save, metadata={"format": "pt"})
    else:
        torch.save(state_dict, path_to_save)


@torch.no_grad()
def save_model_weights(
    output_dir: Union[str, os.PathLike],
    state_dict: Dict[str, torch.Tensor],
    save_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    shard_size: int = 5_000_000_000,
    safe_serialization: bool = True,
    model_assets: Optional[Sequence["ModelAssets"]] = None,
) -> None:
    """
    Save model weights in HuggingFace format.

    Args:
        output_dir: Directory to save weights.
        state_dict: Model state dict to save.
        save_dtype: Target dtype for saving (default: bfloat16).
        shard_size: Maximum shard size in bytes (default: 5GB).
        safe_serialization: Whether to use safetensors format (default: True).
        model_assets: Optional model assets (config, processor, etc.) to save.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving model weights to {output_dir}")
    logger.info(f"Format: {'safetensors' if safe_serialization else 'pytorch'}, dtype: {save_dtype}")

    is_sharded, total_size, weight_map = _get_shard_info(state_dict, save_dtype, shard_size, safe_serialization)

    full_state_dict = OrderedDict()
    prev_file_name = None

    for name, tensor in state_dict.items():
        # Handle DTensor (distributed tensor)
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()

        # Convert dtype if specified
        if save_dtype:
            target_dtype = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype
            tensor = tensor.to(dtype=target_dtype)

        # Save previous shard if moving to next file
        if prev_file_name is not None and weight_map[name] != prev_file_name:
            _save_state_dict(full_state_dict, os.path.join(output_dir, prev_file_name), safe_serialization)
            full_state_dict = OrderedDict()

        full_state_dict[name] = tensor.detach().cpu()
        prev_file_name = weight_map[name]
        del tensor

    # Save final shard
    if len(full_state_dict):
        _save_state_dict(full_state_dict, os.path.join(output_dir, prev_file_name), safe_serialization)

    # Save index file for sharded checkpoints
    if is_sharded:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        with open(os.path.join(output_dir, index_file), "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)

        logger.info(f"Model weight splits saved in {output_dir}")
    else:
        logger.info(f"Model weights saved at {os.path.join(output_dir, prev_file_name)}")

    # Save model assets (config, processor, etc.)
    if model_assets is not None:
        for model_asset in model_assets:
            if hasattr(model_asset, "save_pretrained"):
                model_asset.save_pretrained(output_dir)
                logger.info(f"Saved model asset: {type(model_asset).__name__}")
            else:
                logger.warning(f"Model asset {model_asset} does not implement `save_pretrained`")


def merge_to_hf_pt(load_dir: str, save_path: str, model_assets_dir: Optional[str] = None) -> None:
    """
    Merge DCP checkpoint to HuggingFace format.

    Args:
        load_dir: Directory containing DCP checkpoint.
        save_path: Output directory for HuggingFace format checkpoint.
        model_assets_dir: Optional directory containing model config and processor.
    """
    # Load state dict from DCP checkpoint (model weights only)
    state_dict = dcp_to_torch_state_dict(save_checkpoint_path=load_dir)

    # Load model assets if provided
    model_assets = None
    if model_assets_dir is not None:
        logger.info(f"Loading model assets from {model_assets_dir}")
        try:
            config = AutoConfig.from_pretrained(model_assets_dir)
            processor = AutoProcessor.from_pretrained(model_assets_dir, trust_remote_code=True)
            model_assets = [config, processor]
        except Exception as e:
            logger.warning(f"Failed to load model assets: {e}")
            model_assets = None

    # Save in HuggingFace format
    save_model_weights(save_path, state_dict, model_assets=model_assets)


def main():
    parser = argparse.ArgumentParser(
        description="Merge DCP checkpoint to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--load-dir", type=str, required=True, help="Directory containing DCP checkpoint")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory for HuggingFace format checkpoint (default: <load-dir>/hf_ckpt)",
    )
    parser.add_argument(
        "--model-assets-dir",
        type=str,
        default=None,
        help="Directory containing model config and processor (optional)",
    )
    args = parser.parse_args()

    load_dir = args.load_dir
    save_dir = os.path.join(load_dir, "hf_ckpt") if args.save_dir is None else args.save_dir
    model_assets_dir = args.model_assets_dir

    logger.info(f"Merge Args: load_dir={load_dir}, save_dir={save_dir}, model_assets_dir={model_assets_dir}")

    merge_to_hf_pt(load_dir, save_dir, model_assets_dir)

    logger.info(f"Merge to HF format success! Saved to: {save_dir}")


if __name__ == "__main__":
    main()
