import argparse
import os
import shutil
import sys
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
from transformers import AutoConfig, AutoProcessor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from veomni.checkpoint.conversion import CachedFileSystemReader, export_shards, hf_tensors, write_index
from veomni.checkpoint.dcp_checkpointer import _get_sharding_plan
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
    export_shards(
        [("adapter_model.safetensors", lora_keys)],
        CachedFileSystemReader(checkpoint_path),
        output_dir,
        lambda keys, source: hf_tensors(keys, source, save_dtype),
    )

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

    groups = []
    num_shards = len(shards)
    for shard_idx, shard_keys in enumerate(shards):
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        if num_shards == 1:
            filename = weights_name
        else:
            prefix, extension = weights_name.rsplit(".", maxsplit=1)
            filename = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"
        groups.append((filename, shard_keys))

    weight_map, total_size = export_shards(
        groups,
        CachedFileSystemReader(checkpoint_path),
        output_dir,
        lambda keys, source: hf_tensors(keys, source, save_dtype),
        safe_serialization=safe_serialization,
    )
    if num_shards > 1:
        index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        write_index(output_dir, weight_map, total_size, index_file)

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
    parser.add_argument(
        "--format",
        choices=("hf", "v4-flash", "v4-flash-base"),
        default="hf",
        help="Output format (default: hf preserves the existing HF/LoRA export)",
    )
    v4 = parser.add_argument_group("DeepSeek V4 export options")
    v4.add_argument("--device", default="cuda:0", help="Device for V4 quantization kernels")
    v4.add_argument("--workers", type=int, default=1, help="Number of V4 shard conversion workers")
    v4.add_argument(
        "--ep-size",
        type=int,
        default=1,
        help="EP size for muon_expert_zero_comm checkpoints; leave at 1 for hidden-dimension FSDP shards",
    )
    v4.add_argument("--layers", help="Layer subset for smoke tests, e.g. 0,2-3 (no assets or index)")
    v4.add_argument("--custom-template", help="Training-time chat template: file path or inline Jinja")
    v4.add_argument("--skip-assets", action="store_true", help="Skip V4 config/tokenizer assets, still write index")
    v4.add_argument("--overwrite", action="store_true", help="Rewrite complete V4 output shards")
    v4.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    v4.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    load_dir = args.load_dir
    save_dir = os.path.join(load_dir, "hf_ckpt") if args.save_dir is None else args.save_dir
    if args.format != "hf":
        if args.mode == "lora" or args.adapter_config_path is not None or args.model_assets_dir is not None:
            parser.error("V4 formats do not accept --mode lora, --adapter-config-path or --model-assets-dir")
        if args.shard_size != 2_000_000_000:
            parser.error("V4 formats use release shard layouts; --shard-size applies only to hf")
        from veomni.models.transformers.deepseek_v4.checkpoint_export import merge_checkpoint

        args.save_dir = save_dir
        merge_checkpoint(args, parser, entrypoint=__file__, argv=sys.argv[1:])
        return
    v4_only = (
        args.workers != 1
        or args.ep_size != 1
        or args.layers is not None
        or args.custom_template is not None
        or args.skip_assets
        or args.overwrite
        or args.worker_index is not None
        or args.worker_count != 1
        or args.device != "cuda:0"
    )
    if v4_only:
        parser.error("V4 export options require --format v4-flash or v4-flash-base")

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
