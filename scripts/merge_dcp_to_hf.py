import argparse
import gc
import json
import os
import shutil
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from veomni.checkpoint.dcp_checkpointer import _get_sharding_plan, _process_shard
from veomni.utils import helper


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logger = helper.create_logger(__name__)


# PEFT LoRA adapter key markers. The training DCP keeps PEFT's wrapped FQNs intact
# (e.g. ``base_model.model.<...>.lora_A.default.weight``), so a substring check on
# the HF-normalized keys is enough to spot a LoRA checkpoint.
_LORA_KEY_MARKERS = (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B.")


def _is_lora_key(hf_key: str) -> bool:
    return any(marker in hf_key for marker in _LORA_KEY_MARKERS)


def _detect_lora(all_hf_keys: Sequence[str]) -> bool:
    """Return True if any of the HF-normalized keys looks like a PEFT LoRA adapter."""
    return any(_is_lora_key(k) for k in all_hf_keys)


@torch.no_grad()
def save_lora_adapter_weights(
    output_dir: Union[str, os.PathLike],
    checkpoint_path: Union[str, os.PathLike],
    save_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    adapter_config_path: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Convert a DCP checkpoint that contains a PEFT LoRA adapter to ``adapter_model.safetensors``.

    Only ``*.lora_A.*`` / ``*.lora_B.*`` keys are exported, mirroring what
    ``veomni.utils.save_safetensor_utils.save_lora_adapter_with_dcp`` writes during
    training. The base model weights present in the DCP (frozen during LoRA fine-tuning)
    are intentionally dropped: at inference time they must come from the original
    base model path the LoRA was trained against.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving LoRA adapter to {output_dir}")

    # ``shard_size=None`` forces a single shard so we get ``adapter_model.safetensors`` directly.
    all_keys, _total_size, _all_dcp_keys = _get_sharding_plan(checkpoint_path, shard_size=None, save_dtype=save_dtype)
    lora_keys = {hf_k: dcp_k for hf_k, dcp_k in all_keys.items() if _is_lora_key(hf_k)}
    if not lora_keys:
        raise RuntimeError(
            f"LoRA conversion requested but no LoRA keys (.lora_A./.lora_B./...) found under {checkpoint_path}"
        )

    logger.info(f"Found {len(lora_keys)} LoRA tensors; loading and re-saving as adapter_model.safetensors")
    processed_dict = _process_shard(lora_keys, checkpoint_path, save_dtype)

    save_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(processed_dict, save_path, metadata={"format": "pt"})
    del processed_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if adapter_config_path is not None:
        adapter_config_path = str(adapter_config_path)
        if not os.path.isfile(adapter_config_path):
            raise FileNotFoundError(f"--adapter-config-path does not exist: {adapter_config_path}")
        shutil.copyfile(adapter_config_path, os.path.join(output_dir, "adapter_config.json"))
        logger.info(f"Copied adapter_config.json from {adapter_config_path}")
    else:
        logger.warning(
            "No --adapter-config-path provided. ``adapter_model.safetensors`` was written, but you must drop "
            "``adapter_config.json`` (from the matching training run's ``output_dir/global_step_*/``) next to it "
            "before the adapter can be loaded by peft / diffusers."
        )

    logger.info("LoRA adapter conversion complete.")


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

    for shard_idx, shard_keys in enumerate(shards):
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        if num_shards == 1:
            filename = weights_name
        else:
            prefix, extension = weights_name.rsplit(".", maxsplit=1)
            filename = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"

        save_path = os.path.join(output_dir, filename)
        logger.info(f"Processing shard {shard_idx + 1}/{num_shards}: {filename} ({len(shard_keys)} tensors)")

        processed_dict = _process_shard(shard_keys, checkpoint_path, save_dtype)

        # Save shard
        if safe_serialization:
            save_file(processed_dict, save_path, metadata={"format": "pt"})
        else:
            torch.save(processed_dict, save_path)

        del processed_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    parser.add_argument(
        "--mode",
        choices=("auto", "full", "lora"),
        default="auto",
        help=(
            "Conversion mode. 'auto' (default) inspects DCP keys: writes adapter_model.safetensors when the "
            "checkpoint contains PEFT LoRA keys, otherwise writes a full sharded HF safetensors dump. "
            "'full' / 'lora' force the corresponding mode."
        ),
    )
    parser.add_argument(
        "--adapter-config-path",
        type=str,
        default=None,
        help=(
            "Path to the matching adapter_config.json produced during LoRA training "
            "(usually under <output_dir>/global_step_*/adapter_config.json). Only used in 'lora' mode; "
            "copied next to adapter_model.safetensors so the adapter is loadable as-is."
        ),
    )
    args = parser.parse_args()

    load_dir = args.load_dir
    save_dir = os.path.join(load_dir, "hf_ckpt") if args.save_dir is None else args.save_dir
    model_assets_dir = args.model_assets_dir
    shard_size = args.shard_size

    mode = args.mode
    if mode == "auto":
        _shards_for_detection, _size, _dcp_keys = _get_sharding_plan(load_dir, shard_size=None, save_dtype="bfloat16")
        # _shards_for_detection is a single {hf_key: dcp_key} dict when shard_size is None
        detected_lora = _detect_lora(_shards_for_detection.keys())
        mode = "lora" if detected_lora else "full"
        logger.info(
            f"Auto-detected mode: {mode} "
            f"({'LoRA keys present' if detected_lora else 'no LoRA keys; treating as full checkpoint'})"
        )

    if mode == "lora":
        save_lora_adapter_weights(
            output_dir=save_dir,
            checkpoint_path=load_dir,
            adapter_config_path=args.adapter_config_path,
        )
    else:
        if args.adapter_config_path is not None:
            logger.warning("--adapter-config-path is only used in 'lora' mode; ignoring.")
        merge_to_hf_pt(load_dir, save_dir, model_assets_dir, shard_size=shard_size)


if __name__ == "__main__":
    main()
