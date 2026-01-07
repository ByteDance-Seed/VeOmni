import argparse
import gc
import json
import logging
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader, load
from torch.distributed.checkpoint.metadata import Metadata
from transformers import AutoConfig, AutoProcessor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_dtype_size(dtype: torch.dtype) -> int:
    """Return size in bytes for a given dtype."""
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


def _normalize_key(key: str) -> Optional[str]:
    """
    Convert DCP key to HuggingFace format. Returns None for non-model weights.

    Conversion rules:
    - "model.model.*" -> "model.*" (remove first "model." prefix)
    - "model.lm_head.weight" -> "lm_head.weight" (special case)
    - Other "model.*" keys -> log warning and strip "model." prefix
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


def _get_sharding_plan(
    checkpoint_path: Union[str, os.PathLike],
    shard_size: int,
    save_dtype: Optional[Union[str, torch.dtype]],
) -> Tuple[List[Dict[str, str]], Dict[str, str], int, List[str]]:
    """
    Create sharding plan from checkpoint metadata without loading weights.

    Returns:
        shards: List of {hf_key: dcp_key} dicts per shard
        total_size: Total size in bytes
        all_dcp_keys: All valid DCP model keys
    """
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    if not isinstance(metadata, Metadata):
        raise ValueError(f"Invalid metadata format in {checkpoint_path}")

    # Collect model tensors and calculate sizes
    tensor_infos = []
    all_dcp_keys = []

    for key, tensor_meta in metadata.state_dict_metadata.items():
        hf_key = _normalize_key(key)
        if hf_key:
            # Determine dtype for size calculation
            if save_dtype:
                dtype = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype
            else:
                if not hasattr(tensor_meta.properties, "dtype"):
                    raise ValueError(
                        f"Cannot determine dtype for tensor '{key}': metadata does not contain dtype information"
                    )
                dtype = tensor_meta.properties.dtype

            # Calculate tensor size in bytes
            numel = 1
            for dim in tensor_meta.size:
                numel *= dim

            byte_size = numel * get_dtype_size(dtype)

            tensor_infos.append({"dcp_key": key, "hf_key": hf_key, "size": byte_size, "metadata": tensor_meta})
            all_dcp_keys.append(key)

    # Sort by key name for deterministic output
    tensor_infos.sort(key=lambda x: x["hf_key"])

    # Pack tensors into shards
    shards = []
    current_shard = {}
    current_shard_size = 0
    total_size = 0

    for info in tensor_infos:
        size = info["size"]
        total_size += size

        # Start new shard if adding this tensor exceeds shard_size (unless current shard is empty)
        if current_shard and (current_shard_size + size > shard_size):
            shards.append(current_shard)
            current_shard = {}
            current_shard_size = 0

        current_shard[info["hf_key"]] = info["dcp_key"]
        current_shard_size += size

    if current_shard:
        shards.append(current_shard)

    return shards, total_size, all_dcp_keys


def _process_shard(
    shard_idx: int,
    num_shards: int,
    shard_keys: Dict[str, str],  # hf_key -> dcp_key
    checkpoint_path: str,
    output_dir: str,
    save_dtype: Optional[Union[str, torch.dtype]],
    safe_serialization: bool,
) -> str:
    """Load, cast, and save a single shard. Returns the shard filename."""
    # Determine filename
    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    if num_shards == 1:
        filename = weights_name
    else:
        prefix, extension = weights_name.rsplit(".", maxsplit=1)
        filename = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"

    save_path = os.path.join(output_dir, filename)
    logger.info(f"Processing shard {shard_idx + 1}/{num_shards}: {filename} ({len(shard_keys)} tensors)")

    # Create placeholder state_dict with correct shapes for loading
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    state_dict = OrderedDict()
    dcp_keys_to_load = list(shard_keys.values())

    for dcp_key in dcp_keys_to_load:
        tensor_metadata = metadata.state_dict_metadata[dcp_key]
        if not hasattr(tensor_metadata.properties, "dtype"):
            raise ValueError(
                f"Cannot determine dtype for tensor '{dcp_key}': metadata does not contain dtype information"
            )
        state_dict[dcp_key] = torch.empty(
            tensor_metadata.size,
            dtype=tensor_metadata.properties.dtype,
        )

    # Load partial checkpoint
    load(
        state_dict,
        checkpoint_id=checkpoint_path,
        storage_reader=FileSystemReader(checkpoint_path),
        no_dist=True,
    )

    # Cast and rename tensors
    processed_dict = OrderedDict()
    target_dtype = None
    if save_dtype:
        target_dtype = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype

    for hf_key, dcp_key in shard_keys.items():
        tensor = state_dict[dcp_key]

        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()

        if target_dtype:
            tensor = tensor.to(dtype=target_dtype)

        processed_dict[hf_key] = tensor.detach().cpu()

    del state_dict
    gc.collect()

    # Save shard
    if safe_serialization:
        save_file(processed_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(processed_dict, save_path)

    del processed_dict
    gc.collect()

    return filename


@torch.no_grad()
def save_model_weights(
    output_dir: Union[str, os.PathLike],
    checkpoint_path: Union[str, os.PathLike],
    save_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    shard_size: int = 2_000_000_000,
    safe_serialization: bool = True,
    model_assets: Optional[Sequence["ModelAssets"]] = None,
) -> None:
    """Convert DCP checkpoint to HuggingFace format with shard-by-shard processing (memory-efficient)."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model weights to {output_dir}")
    logger.info(
        f"Format: {'safetensors' if safe_serialization else 'pytorch'}, dtype={save_dtype}, shard_size={shard_size}"
    )

    # Plan shards from metadata
    logger.info("Analyzing DCP metadata and planning shards...")
    shards, total_size, all_dcp_keys = _get_sharding_plan(checkpoint_path, shard_size, save_dtype)

    logger.info(f"Found {len(all_dcp_keys)} model tensors, total size: ~{total_size / 1e9:.2f}GB")
    logger.info(f"Split into {len(shards)} shards")

    if len(shards) == 0:
        logger.warning("No model weights found! Check if checkpoint path is correct and contains 'model.' keys.")
        return

    # Process each shard
    weight_map = OrderedDict()
    num_shards = len(shards)

    for idx, shard_keys in enumerate(shards):
        filename = _process_shard(
            idx, num_shards, shard_keys, checkpoint_path, output_dir, save_dtype, safe_serialization
        )

        for hf_key in shard_keys.keys():
            weight_map[hf_key] = filename

    # Save index file for multi-shard checkpoints
    if num_shards > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        with open(os.path.join(output_dir, index_file), "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        logger.info(f"Saved index file to {index_file}")

    logger.info("Weight conversion complete.")

    # Save model assets (config, tokenizer, processor)
    if model_assets is not None:
        for model_asset in model_assets:
            if hasattr(model_asset, "save_pretrained"):
                model_asset.save_pretrained(output_dir)
                logger.info(f"Saved model asset: {type(model_asset).__name__}")
            else:
                logger.warning(f"Model asset {model_asset} does not implement `save_pretrained`")


def merge_to_hf_pt(
    load_dir: str, save_path: str, model_assets_dir: Optional[str] = None, shard_size: int = 2_000_000_000
) -> None:
    """Main conversion function: load DCP from load_dir and save HF format to save_path."""
    model_assets = None
    if model_assets_dir is not None:
        logger.info(f"Loading model assets from {model_assets_dir}")
        model_assets = []
        try:
            config = AutoConfig.from_pretrained(model_assets_dir)
            model_assets.append(config)
        except Exception as e:
            logger.warning(f"Failed to load AutoConfig: {e}")

        try:
            processor = AutoProcessor.from_pretrained(model_assets_dir, trust_remote_code=True)
            model_assets.append(processor)
        except Exception as e:
            logger.warning(f"Failed to load AutoProcessor: {e}")

        if not model_assets:
            model_assets = None

    save_model_weights(save_path, load_dir, shard_size=shard_size, model_assets=model_assets)


def main():
    parser = argparse.ArgumentParser(
        description="Merge DCP checkpoint to HuggingFace format (streaming optimized)",
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
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2_000_000_000,
        help="Maximum shard size in bytes (default: 2GB)",
    )
    args = parser.parse_args()

    load_dir = args.load_dir
    save_dir = os.path.join(load_dir, "hf_ckpt") if args.save_dir is None else args.save_dir
    model_assets_dir = args.model_assets_dir
    shard_size = args.shard_size

    merge_to_hf_pt(load_dir, save_dir, model_assets_dir, shard_size=shard_size)


if __name__ == "__main__":
    main()
